
import sys,os,shutil,json,h5py,glob,argparse
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pyarrow.parquet as pq
import openslide
import pickle
import random
import io
import tarfile
import time
import idr_torch
from urllib.parse import urlsplit
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
torch.set_printoptions(sci_mode=False)
# torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torcheval.metrics.functional import r2_score as r2_score_pytorch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import set_layer_config
from timm.models import is_model, model_entrypoint, load_checkpoint
from transformers import CLIPModel, CLIPProcessor


BACKBONE_DICT = {
    'resnet50': 2048,
    'CLIP': 512,
    'PLIP': 512,
    'MobileNetV3': 1280,
    'mobilenetv3': 1280,
    'ProvGigaPath': 1536,
    # 'CONCH': 512,
    'CONCH': 768,
    'UNI': 1024
}


def load_cfg_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        text = reader.read()
    return json.loads(text)

def load_model_config_from_hf(model_id: str):
    cached_file = './backbones/ProvGigaPath/config.json'

    hf_config = load_cfg_from_json(cached_file)
    if 'pretrained_cfg' not in hf_config:
        # old form, pull pretrain_cfg out of the base dict
        pretrained_cfg = hf_config
        hf_config = {}
        hf_config['architecture'] = pretrained_cfg.pop('architecture')
        hf_config['num_features'] = pretrained_cfg.pop('num_features', None)
        if 'labels' in pretrained_cfg:  # deprecated name for 'label_names'
            pretrained_cfg['label_names'] = pretrained_cfg.pop('labels')
        hf_config['pretrained_cfg'] = pretrained_cfg

    # NOTE currently discarding parent config as only arch name and pretrained_cfg used in timm right now
    pretrained_cfg = hf_config['pretrained_cfg']
    pretrained_cfg['hf_hub_id'] = model_id  # insert hf_hub id for pretrained weight load during model creation
    pretrained_cfg['source'] = 'hf-hub'

    # model should be created with base config num_classes if its exist
    if 'num_classes' in hf_config:
        pretrained_cfg['num_classes'] = hf_config['num_classes']

    # label meta-data in base config overrides saved pretrained_cfg on load
    if 'label_names' in hf_config:
        pretrained_cfg['label_names'] = hf_config.pop('label_names')
    if 'label_descriptions' in hf_config:
        pretrained_cfg['label_descriptions'] = hf_config.pop('label_descriptions')

    model_args = hf_config.get('model_args', {})
    model_name = hf_config['architecture']
    return pretrained_cfg, model_name, model_args


def split_model_name_tag(model_name: str, no_tag: str = ''):
    model_name, *tag_list = model_name.split('.', 1)
    tag = tag_list[0] if tag_list else no_tag
    return model_name, tag


def parse_model_name(model_name: str):
    if model_name.startswith('hf_hub'):
        # NOTE for backwards compat, deprecate hf_hub use
        model_name = model_name.replace('hf_hub', 'hf-hub')
    parsed = urlsplit(model_name)
    assert parsed.scheme in ('', 'timm', 'hf-hub')
    if parsed.scheme == 'hf-hub':
        # FIXME may use fragment as revision, currently `@` in URI path
        return parsed.scheme, parsed.path
    else:
        model_name = os.path.split(parsed.path)[-1]
        return 'timm', model_name


def create_model():
    model_name = 'hf_hub:prov-gigapath/prov-gigapath'
    model_source, model_name = parse_model_name(model_name)
    pretrained_cfg, model_name, model_args = load_model_config_from_hf(model_name)
    kwargs = {}
    if model_args:
        for k, v in model_args.items():
            kwargs.setdefault(k, v)
    create_fn = model_entrypoint(model_name)
    with set_layer_config(scriptable=None, exportable=None, no_jit=None):
        model = create_fn(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            **kwargs
        )
    load_checkpoint(model, './backbones/ProvGigaPath/pytorch_model.bin')

    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


def ddp_setup():
    dist.init_process_group(backend="nccl")


def collect_results_gpu(part_tensor, size, world_size):
    shape = part_tensor.shape
    shape_tensor = torch.tensor(shape[0], device=part_tensor.device)
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    shape_max = torch.tensor(shape_list).max()

    if len(shape) == 1:
        part_send = torch.zeros(shape_max, dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    if len(shape) == 2:
        part_send = torch.zeros((shape_max, shape[1]), dtype=part_tensor.dtype, device=part_tensor.device)
        part_send[:shape_tensor] = part_tensor
        part_recv_list = [
            part_tensor.new_zeros(shape_max, shape[1]) for _ in range(world_size)
        ]
        dist.all_gather(part_recv_list, part_send)
    return torch.cat(part_recv_list, axis=0)[:size]


class Trainer:
    def __init__(
            self,
            model,
            dataloader,
            optimizer,
            save_root='./outputs',
            accum_iter=1,
            gene_names=[]
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.model = model.to(self.gpu_id)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = os.path.join(save_root, 'snapshot.pt') 
        self.accum_iter = accum_iter
        self.save_root = save_root
        self.gene_names = gene_names

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.HuberLoss()
        
        if self.gpu_id == 0:
            self.losses = []

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

    def _run_epoch(self, epoch):

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        b_sz = len(next(iter(self.dataloader))[0])
        self.dataloader.sampler.set_epoch(epoch)
        dataset = self.dataloader.dataset

        total_loss = torch.tensor([0.], dtype=torch.float32, device=self.gpu_id)
        for batch_idx, (images_batch, labels_batch) in enumerate(self.dataloader):
            images_batch = images_batch.to(self.gpu_id)
            labels_batch = labels_batch.to(self.gpu_id)

            preds_batch = self.model(images_batch)
            mask = torch.isnan(labels_batch)
            loss = self.loss_fn(preds_batch[~mask], labels_batch[~mask])

            loss = loss / self.accum_iter
            loss.backward()

            if (batch_idx + 1) % self.accum_iter == 0:
                self.optimizer.step()
                self.model.zero_grad()
                self.optimizer.zero_grad()

            total_loss[0] += loss

        total_loss = collect_results_gpu(total_loss, 1, world_size=self.world_size)
        total_loss = total_loss.sum()

        if self.gpu_id == 0:
            self.losses.append([epoch, total_loss.item()])
            log_df = pd.DataFrame(self.losses, columns=['epoch', 'total_loss'])
            log_df.to_csv(os.path.join(self.save_root, f'loss.csv'), float_format='%.3f')
        dist.barrier()

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path.replace('.pt', '_{}.pt'.format(epoch)))
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0:
                self._save_snapshot(epoch)


class STModel(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.25, num_outputs=24665):
        super().__init__()
        self.backbone = backbone
        if backbone == 'resnet50':
            self.backbone_model = torchvision.models.resnet50(pretrained=True)
            self.backbone_model.fc = nn.Identity()
            self.transform = None
            self.image_processor = None
        elif backbone == 'CONCH':
            # from conch.open_clip_custom import create_model_from_pretrained
            # self.backbone_model, self.image_processor = create_model_from_pretrained('conch_ViT-B-16','./backbones/CONCH_weights_pytorch_model.bin')
            # self.transform = None
            from timm.models.vision_transformer import VisionTransformer
            self.backbone_model = VisionTransformer(embed_dim=768, 
                                                    depth=12, 
                                                    num_heads=12, 
                                                    mlp_ratio=4,
                                                    img_size=448, 
                                                    patch_size=16,
                                                    num_classes=0,
                                                    dynamic_img_size=True)
            self.backbone_model.load_state_dict(torch.load('./backbones/CONCH_vision_weights_pytorch_model.bin', weights_only=True))
        elif backbone == 'UNI':
            self.backbone_model = timm.create_model(
                "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
            )
            self.backbone_model.load_state_dict(torch.load("./backbones/UNI_pytorch_model.bin", map_location="cpu", weights_only=True), strict=True)
            self.transform = create_transform(**resolve_data_config(self.backbone_model.pretrained_cfg, model=self.backbone_model))
        elif backbone == 'ProvGigaPath':
            self.backbone_model = create_model()  # timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif backbone == 'CLIP':
            self.backbone_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        elif backbone == 'PLIP':
            self.backbone_model = CLIPModel.from_pretrained("./backbones/vinid_plip")
            self.image_processor = CLIPProcessor.from_pretrained("./backbones/vinid_plip")
        else:
            raise ValueError('error')

        self.rho = nn.Sequential(*[
            nn.Linear(BACKBONE_DICT[backbone], BACKBONE_DICT[backbone]), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        ])

        self.fc = nn.Linear(BACKBONE_DICT[backbone], num_outputs)

        # self.initialize_weights()
        self.rho.apply(self._init_weights)
        self.fc.apply(self._init_weights)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        
        if self.backbone in ['PLIP', 'CLIP']:
            h = self.backbone_model.get_image_features(x)
        elif self.backbone in ['resnet50', 'UNI', 'ProvGigaPath', 'CONCH']:
            h = self.backbone_model(x)

        h = self.rho(h)

        h = self.fc(h)

        # h = torch.tanh(h)

        return h


class PatchDataset(Dataset):
    def __init__(self, coords_df, counts, transform, is_train=False, cache_root='./'):
        super().__init__()
        self.coords_df = coords_df
        self.counts = counts

        self.transform = transform 
        self.is_train = is_train
        self.cache_root = cache_root

    def __len__(self):
        return len(self.coords_df)

    def __getitem__(self, idx): 
        patch = Image.open(os.path.join(self.cache_root, self.coords_df.loc[idx, 'patch_filename']))
        if self.is_train:
            if np.random.rand() < 0.5:
                patch = patch.rotate(np.random.choice([90, 180, 270]))
            # if np.random.rand() < 0.2:
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=np.random.randint(low=1,high=50)/100.)) 
        label = self.counts[idx]
        return self.transform(patch), label


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--trigger_eval_every', type=int, default=10)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--fixed_backbone', action='store_true')
    parser.add_argument('--use_smooth', action='store_true')
    parser.add_argument('--use_stain', action='store_true')
    parser.add_argument('--train_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx')
    parser.add_argument('--val_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx')
    parser.add_argument('--data_root', type=str, default='/data/zhongz2/temp29/ST_prediction_data')
    parser.add_argument('--gene_names', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_0.05_smooth_stain/valid_genes.pkl')
    parser.add_argument('--ckpt_path', type=str, default='')

    return parser.parse_args()


def build_dataloaders(args):
    train_df = pd.read_excel(args.train_csv) if 'xlsx' in args.train_csv else pd.read_csv(args.train_csv)
    val_df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    ### data realted
    if True:  # use imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ElasticTransform(alpha=50.),
        # transforms.ColorJitter(brightness=.3, hue=.2),
        # transforms.GaussianBlur(kernel_size=(5, 9)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_coords_df = []
    train_counts = []
    for rowid, row in train_df.iterrows():
        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        coord_df = pd.read_csv(os.path.join(args.data_root, save_prefix+'_coord.csv'))
        count_pt = torch.load(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_data', save_prefix+'_gene_count.pth'), weights_only=True)
        train_coords_df.append(coord_df)
        train_counts.append(count_pt)
    train_coords_df = pd.concat(train_coords_df)
    train_coords_df.index = np.arange(len(train_coords_df))
    train_counts = torch.cat(train_counts)

    train_dataset = PatchDataset(coords_df=train_coords_df, counts=train_counts, transform=train_transform, is_train=True, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))


    val_coords_df = []
    val_counts = []
    for rowid, row in val_df.iterrows():
        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        coord_df = pd.read_csv(os.path.join(args.data_root, save_prefix+'_coord.csv'))
        count_pt = torch.load(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_data', save_prefix+'_gene_count.pth'), weights_only=True)
        val_coords_df.append(coord_df)
        val_counts.append(count_pt)
    val_coords_df = pd.concat(val_coords_df)
    val_coords_df.index = np.arange(len(val_coords_df))
    val_counts = torch.cat(val_counts)

    train_dataset = PatchDataset(coords_df=train_coords_df, counts=train_counts, transform=train_transform, is_train=True, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))
    val_dataset = PatchDataset(coords_df=val_coords_df, counts=val_counts, transform=val_transform, is_train=False, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))
        

    dataloaders = {
        'train':
            DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)),
        'val':
            DataLoader(val_dataset, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False))
    }

    return dataloaders

def train_main(args):

    ddp_setup()

    cache_root = os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth))
    save_root = f'{cache_root}/results/gpus{args.num_gpus}/backbone{args.backbone}_fixed{args.fixed_backbone}/lr{args.lr}_b{args.batch_size}_e{args.max_epochs}_accum{args.accum_iter}_v0_smooth{args.use_smooth}_stain{args.use_stain}'
    os.makedirs(save_root, exist_ok=True)

    if os.path.exists(os.path.join(save_root, 'snapshot_{}.pt'.format(args.max_epochs - 1))):
        print('done')
        sys.exit(-1)

    if os.path.isfile(args.gene_names):
        with open(args.gene_names, 'rb') as fp:
            gene_names = pickle.load(fp)['valid_genes']
    else:
        gene_names = [v.upper() for v in args.gene_names.split(',')]
    
    train_df = pd.read_excel(args.train_csv) if 'xlsx' in args.train_csv else pd.read_csv(args.train_csv)

    ### data realted
    if True:  # use imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ElasticTransform(alpha=50.),
        # transforms.ColorJitter(brightness=.3, hue=.2),
        # transforms.GaussianBlur(kernel_size=(5, 9)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((224, 224)), 
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])

    train_coords_df = []
    train_counts = []
    for rowid, row in train_df.iterrows():
        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        coord_df = pd.read_csv(os.path.join(args.data_root, save_prefix+'_coord.csv'))
        count_pt = torch.load(os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth), save_prefix+'_gene_count.pth'), weights_only=True)
        train_coords_df.append(coord_df)
        train_counts.append(count_pt)
    train_coords_df = pd.concat(train_coords_df)
    train_coords_df.index = np.arange(len(train_coords_df))
    train_counts = torch.cat(train_counts)

    train_dataset = PatchDataset(coords_df=train_coords_df, counts=train_counts, transform=train_transform, is_train=True, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))

    train_dataloader = \
        DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False))

    ### model related
    model = STModel(backbone=args.backbone, num_outputs=len(gene_names))
    if args.fixed_backbone:
        for param in model.backbone_model.parameters():
            param.requires_grad = False

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    backbone_params = [param for name, param in model.named_parameters() if 'backbone_model' in str(name)]
    other_params = [param for name, param in model.named_parameters() if 'backbone_model' not in str(name)]
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr}, 
        {'params': other_params, 'lr': args.lr*10}
    ], lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(model, train_dataloader, optimizer, save_root=save_root, accum_iter=args.accum_iter, gene_names=gene_names)
    trainer.train(max_epochs=args.max_epochs)
    dist.destroy_process_group()

    if False: # idr_torch.rank == 0:

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        r2 = pd.read_csv(os.path.join(save_root, 'val_r2score.csv'), index_col=0)
        pearson = pd.read_csv(os.path.join(save_root, 'val_pearsonr_corr.csv'), index_col=0)
        spearman = pd.read_csv(os.path.join(save_root, 'val_spearmanr_corr.csv'), index_col=0)

        s = pearson.sum(axis=0)
        sorted_names = s.sort_values(ascending=False).index
        r2 = r2[sorted_names]
        pearson = pearson[sorted_names]
        spearman = spearman[sorted_names]

        all_vst_df = {
            'train': pd.DataFrame(train_counts, columns=gene_names),
            'val': pd.DataFrame(val_counts, columns=gene_names)
        }
        bins = np.arange(0, 1, 0.1)

        for j,gene_name in enumerate(sorted_names[:min(5000, r2.shape[1])]):
            fig, axes = plt.subplots(nrows=1, ncols=2)
            ax = axes[0]
            r2[gene_name].plot(ax=ax)
            spearman[gene_name].plot(ax=ax)
            pearson[gene_name].plot(ax=ax)
            ax.legend(['r2', 'spearman', 'pearson'])
            ax.set_title('{}, rank = {}'.format(gene_name, j+1))
            # plt.savefig(os.path.join(d, 'val_{:02d}_{}.jpg'.format(j+1, gene_name)))

            if all_vst_df is not None:
                ax = axes[1]
                all_vst_df['train'][gene_name].hist(bins=bins, ax=ax)
                all_vst_df['val'][gene_name].hist(bins=bins, ax=ax)

            im_buffer = io.BytesIO()
            plt.savefig(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=os.path.join('val_{:02d}_{}.jpg'.format(j+1, gene_name)))
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)

            plt.close()

        tar_fp.close()

        save_filename = os.path.join(save_root, 'figures.tar.gz')
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())


def test_main(args):
    assert os.path.isfile(args.ckpt_path)

    if os.path.isfile(args.gene_names):
        with open(args.gene_names, 'rb') as fp:
            gene_names = pickle.load(fp)['valid_genes']
    else:
        gene_names = [v.upper() for v in args.gene_names.split(',')]
    
    val_df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    ### data realted
    if True:  # use imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    # train_transform = transforms.Compose([
    #     transforms.Resize((224, 224)), 
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     # transforms.ElasticTransform(alpha=50.),
    #     # transforms.ColorJitter(brightness=.3, hue=.2),
    #     # transforms.GaussianBlur(kernel_size=(5, 9)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    for rowid, row in train_df.iterrows():
        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        coord_df = pd.read_csv(os.path.join(args.data_root, save_prefix+'_coord.csv'))
        count_pt = torch.load(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'cache_data', save_prefix+'_gene_count.pth'), weights_only=True)

    train_coords_df = pd.concat(train_coords_df)
    train_coords_df.index = np.arange(len(train_coords_df))
    train_counts = torch.cat(train_counts)

    train_dataset = PatchDataset(coords_df=train_coords_df, counts=train_counts, transform=train_transform, is_train=True, cache_root=os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'images'))

    train_dataloader = \
        DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False))

    ### model related
    model = STModel(backbone=args.backbone, num_outputs=len(gene_names))
    if args.fixed_backbone:
        for param in model.backbone_model.parameters():
            param.requires_grad = False



if __name__ == '__main__':
    args = get_args()
    print('args: ', args)
    setup_seed(2024)

    if args.action == 'train':
        train_main(args=args) 
    elif args.action == 'test':
        test_main(args=args)
    else:
        print('wrong action')
