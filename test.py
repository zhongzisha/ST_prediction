
import glob,os,pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import get_args
from model import STModel
from dataset import PatchDataset

import idr_torch


def draw_figures(args):

    ckpt_dir = args.ckpt_dir
    gene_names = args.gene_names.split(',')

    with open(os.path.join(args.ckpt_dir, 'args_rank0.pkl'), 'rb') as fp:
        args = pickle.load(fp)['args']

    df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    with open(os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth), 'gene_infos.pkl'), 'rb') as fp:
        trained_gene_names = pickle.load(fp)['gene_names']
    
    gene_names = trained_gene_names if len(gene_names) == 0 else [v for v in gene_names if v in trained_gene_names]
    if len(gene_names) == 0: gene_names = trained_gene_names

    ckpt_paths = glob.glob(os.path.join(ckpt_dir, 'snapshot_*.pt'))
    num_epochs = len(ckpt_paths)

    bins = [-np.nan] + np.arange(0, 1, 0.1).tolist() + [np.nan]
    keys = ['total_loss', 'r2score', 'spearmanr_corr', 'spearmanr_pvalue', 'pearsonr_corr', 'pearsonr_pvalue']
    for rowid, row in sub_df.iterrows():
        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])

        results = {k: [] for k in keys}
        for epoch in range(num_epochs):

            save_filename = os.path.join(ckpt_dir, f'snapshot_{epoch}_results', save_prefix + '_results.pkl')
            with open(save_filename, 'rb') as fp:
                scores = pickle.load(fp)
            for k in keys:
                results[k].append(scores[k])

        for k in keys:
            res_df = pd.DataFrame(results[k], columns=[k] if k == 'total_loss' else trained_gene_names)
            res_df.to_csv(os.path.join(ckpt_dir, '{}_{}.csv'.format(save_prefix, k)), float_format='%.9f' if 'pvalue' in k else '%.3f')

            if k in ['r2score', 'spearmanr_corr', 'pearsonr_corr']:
                plt.hist(res_df.values[-1, :], bins=bins)
                plt.savefig(os.path.join(ckpt_dir, '{}_{}_hist.png'.format(save_prefix, k)))
                plt.close('all')
            
        print(save_prefix)


def test_main(args):
    
    ckpt_paths = []
    for ckpt_path in glob.glob(os.path.join(args.ckpt_dir, 'snapshot_*.pt')):
        save_dir = ckpt_path.replace('.pt', '_results')
        if not os.path.exists(os.path.join(save_dir, '.ALL_DONE')):
            ckpt_paths.append(ckpt_path)

    if len(ckpt_paths) == 0:
        print('all done, draw figures')
        draw_figures(args)
        return

    indices = np.arange(len(ckpt_paths))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    ckpt_paths = [ckpt_paths[i] for i in index_splits[idr_torch.rank]]

    if len(ckpt_paths) == 0:
        return

    with open(os.path.join(args.ckpt_dir, 'args_rank0.pkl'), 'rb') as fp:
        args = pickle.load(fp)['args']

    df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    with open(os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth), 'gene_infos.pkl'), 'rb') as fp:
        gene_names = pickle.load(fp)['gene_names']
    
    ### model related
    model = STModel(backbone=args.backbone, num_outputs=len(gene_names))
    snapshot = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(snapshot["MODEL_STATE"])
    model.cuda()
    model.eval()

    ### data realted
    if True:  # use imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    for ckpt_path in ckpt_paths:
        save_dir = ckpt_path.replace('.pt', '_results')
        os.makedirs(save_dir, exist_ok=True)

        for rowid, row in df.iterrows():
            save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
            save_filename = os.path.join(save_dir, save_prefix + '_results.pkl')
            if os.path.exists(save_filename):
                continue
            coord_df = pd.read_csv(os.path.join(args.data_root, save_prefix+'_coord.csv'))
            count_pt = torch.load(os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth), save_prefix+'_gene_count.pth'), weights_only=True)
            coord_df.index = np.arange(len(coord_df))

            dataset = PatchDataset(coords_df=coord_df, counts=count_pt, transform=val_transform, is_train=False, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))

            dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, pin_memory=True, shuffle=False, drop_last=False)

            preds = []
            for batch_idx, (images_batch, labels_batch) in enumerate(dataloader):
                images_batch = images_batch.cuda()
                with torch.no_grad():
                    preds_batch = model(images_batch)
                    preds.append(preds_batch.cpu().numpy())
            preds = np.concatenate(preds)  # num_spots x num_genes

            labels = count_pt.numpy()
            total_loss = mean_squared_error(labels, preds)

            r2scores = np.zeros(preds.shape[1])
            spearmanr_corrs = np.zeros(preds.shape[1])
            spearmanr_pvals = np.zeros(preds.shape[1])
            pearsonr_corrs = np.zeros(preds.shape[1])
            pearsonr_pvals = np.zeros(preds.shape[1])
            for j in range(preds.shape[1]):
                r2scores[j] = r2_score(labels[:, j], preds[:, j])

                res = spearmanr(preds[:, j], labels[:, j])
                spearmanr_corrs[j] = res.statistic
                spearmanr_pvals[j] = res.pvalue

                res = pearsonr(preds[:, j], labels[:, j])
                pearsonr_corrs[j] = res.statistic
                pearsonr_pvals[j] = res.pvalue

            scores = {
                'total_loss': total_loss,
                'r2score': r2scores, 
                'spearmanr_corr': spearmanr_corrs, 
                'spearmanr_pvalue': spearmanr_pvals, 
                'pearsonr_corr': pearsonr_corrs, 
                'pearsonr_pvalue': pearsonr_pvals
            }

            with open(save_filename, 'wb') as fp:
                pickle.dump(scores, fp)

        with open(os.path.join(save_dir, '.ALL_DONE'), 'w') as fp:
            pass  

if __name__ == '__main__':
    args = get_args()
    print('args: ', args)

    if args.action == 'test':
        test_main(args=args)
    else:
        print('wrong action')
