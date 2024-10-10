
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
            dataloaders,
            optimizer,
            save_root='./outputs',
            save_every=5,
            accum_iter=1,
            gene_names=[]
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.device = torch.device('cuda:{}'.format(self.gpu_id))
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.model = model.to(self.gpu_id)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = os.path.join(save_root, 'snapshot.pt') 
        self.accum_iter = accum_iter
        self.save_every = save_every
        self.save_root = save_root
        self.gene_names = gene_names

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.HuberLoss()

        if self.gpu_id == 0:
            save_dirs = {}
            for subset in self.dataloaders.keys():
                save_dirs[subset] = os.path.join(save_root, subset)
                os.makedirs(save_dirs[subset], exist_ok=True)
            self.save_dirs = save_dirs

            self.loss_dicts = {subset: {'loss': [], 'r2score': [], 'spearmanr_corr': [], 'spearmanr_pvalue': [], 'pearsonr_corr': [], 'pearsonr_pvalue': []} for subset in self.dataloaders.keys()}

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        # print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch, subset='train'):

        is_train = subset == 'train'
        if is_train:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        b_sz = len(next(iter(self.dataloaders[subset]))[0])
        # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloaders[subset])}")
        self.dataloaders[subset].sampler.set_epoch(epoch)
        dataset = self.dataloaders[subset].dataset

        labels = []
        preds = []
        total_loss = torch.tensor([0.], dtype=torch.float32, device=self.gpu_id)
        for batch_idx, (images_batch, labels_batch) in enumerate(self.dataloaders[subset]):
            images_batch = images_batch.to(self.gpu_id)
            labels_batch = labels_batch.to(self.gpu_id)

            if is_train:
                preds_batch = self.model(images_batch)
            else:
                with torch.no_grad():
                    preds_batch = self.model(images_batch)
            mask = torch.isnan(labels_batch)
            loss = self.loss_fn(preds_batch[~mask], labels_batch[~mask])

            if is_train:
                loss = loss / self.accum_iter
                loss.backward()

                if (batch_idx + 1) % self.accum_iter == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
            else:
                labels.append(labels_batch)
                preds.append(preds_batch)

            total_loss[0] += loss

        total_loss = collect_results_gpu(total_loss, 1, world_size=self.world_size)
        total_loss = total_loss.sum()
        if is_train:
            if self.gpu_id == 0:
                self.loss_dicts[subset]['loss'].append([epoch, total_loss.item()])
        else:
            labels = torch.cat(labels)
            preds = torch.cat(preds)

            all_labels = collect_results_gpu(labels, len(dataset), world_size=self.world_size)
            all_preds = collect_results_gpu(preds, len(dataset), world_size=self.world_size)
            
            if self.gpu_id == 0:
                scores = r2_score_pytorch(all_preds, all_labels, multioutput='raw_values')
                all_preds = all_preds.detach().cpu().numpy()
                all_labels = all_labels.detach().cpu().numpy()
                self.loss_dicts[subset]['loss'].append([epoch, total_loss.item()])
                self.loss_dicts[subset]['r2score'].append(scores.detach().cpu().numpy().tolist())

                spearmanr_corrs = []
                spearmanr_pvals = [] 
                pearsonr_corrs = []
                pearsonr_pvals = []
                for j in range(all_preds.shape[1]):
                    res = spearmanr(all_preds[:, j], all_labels[:, j])
                    spearmanr_corrs.append(res.statistic)
                    spearmanr_pvals.append(res.pvalue)

                    res = pearsonr(all_preds[:, j], all_labels[:, j])
                    pearsonr_corrs.append(res.statistic)
                    pearsonr_pvals.append(res.pvalue)

                self.loss_dicts[subset]['spearmanr_corr'].append(spearmanr_corrs)
                self.loss_dicts[subset]['spearmanr_pvalue'].append(spearmanr_pvals)
                self.loss_dicts[subset]['pearsonr_corr'].append(pearsonr_corrs)
                self.loss_dicts[subset]['pearsonr_pvalue'].append(pearsonr_pvals)

        if self.gpu_id == 0:
            for subset, v in self.loss_dicts.items():
                for name, vv in v.items():
                    if len(vv) == 0:
                        continue
                    if name == 'loss':
                        column_names = ['_epoch', '_total_loss']
                    else:
                        column_names = self.gene_names
                
                    log_df = pd.DataFrame(vv, columns=column_names)
                    # if subset == 'val' and 'r2' in name:
                    #     print(log_df.values[-1, :])
                    log_df.to_csv(os.path.join(self.save_root, f'{subset}_{name}.csv'), float_format='%.9f' if 'pvalue' in name else '%.3f')
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

            for subset in self.dataloaders.keys():
                if self.dataloaders[subset] is not None:
                    self._run_epoch(epoch, subset)

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

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
            nn.Linear(BACKBONE_DICT[backbone], 512), 
            nn.ReLU(), 
            nn.Dropout(dropout)
        ])

        self.fc = nn.Linear(512, num_outputs)
        # self.fc = nn.Linear(BACKBONE_DICT[backbone], num_outputs)

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

        # h = 8 * torch.tanh(h)  # [-8, 8]

        return h





class PatchDataset(Dataset):
    def __init__(self, svs_prefixes, inds, coord_df, vst_df, transform, is_train=False, cache_root='./'):
        super().__init__()
        self.svs_prefixes = svs_prefixes
        self.inds = inds
        self.coord_df = coord_df
        self.vst_df = vst_df

        self.transform = transform 
        self.is_train = is_train
        self.cache_root = cache_root

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx): 
        svs_prefix = self.svs_prefixes[self.inds[idx]]
        x, y = self.coord_df.iloc[idx]
        vst = self.vst_df.iloc[idx]
        patch = Image.open(os.path.join(self.cache_root, svs_prefix, f'x{x}_y{y}.jpg'))
        if self.is_train:
            if np.random.rand() < 0.5:
                patch = patch.rotate(np.random.choice([90, 180, 270]))
            # if np.random.rand() < 0.2:
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=np.random.randint(low=1,high=50)/100.)) 
        label = torch.tensor(vst.values)
        return self.transform(patch), label


def neighbor_smoothing_vst(vst_df, use_10x=False):
    spot_names = vst_df.index.values
    new_vst_df = vst_df.copy()
        
    for s in spot_names:
        r, c = [int(v) for v in s.split('x')]
        if use_10x:
            ns = ['{}x{}'.format(rr, cc) for rr, cc in [
                (r+1, c+1), (r+1, c-1), (r, c-2), (r, c), (r, c+2), (r-1, c-1), (r-1, c+1)
            ]]
        else:
            ns = ['{}x{}'.format(rr, cc) for rr in [r-1, r, r+1] for cc in [c-1, c, c+1]]
        ns = [v for v in ns if v in spot_names]
        new_vst_df.loc[s] = vst_df.loc[ns].mean()
    return new_vst_df

def clean_vst(vst_df, gene_names):
    new_vst_df = pd.DataFrame({col: vst_df[col] if col in vst_df.columns else np.nan for col in gene_names})
    return new_vst_df

def convert_index_names_10x(coord_df, vst_df):
    coord_df = coord_df.copy()
    vst_df = vst_df.copy()
    mappings = {rowind: '{}x{}'.format(row['array_row'], row['array_col']) for rowind, row in coord_df.iterrows()}
    coord_df.index = coord_df.index.map(mappings)
    vst_df.index = vst_df.index.map(mappings)
    return coord_df, vst_df

def load_train_objs(data_root='./data', backbone='resnet50', lr=1e-4, fixed_backbone=False, val_ind=0, cache_root='./', use_vst_smooth=False, save_root='./'): 

    with open(os.path.join(data_root, 'meta.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        alldata = tmp['alldata']
        gene_names = tmp['gene_names']
        low_thres, high_thres = tmp['gene_thres']

    if False:
        all_svs_prefixes = []
        all_inds = {'train': [], 'val': []}
        all_coord_df = {'train': [], 'val': []}
        all_vst_df = {'train': [], 'val': []}
                
        mean, std = [], []
        svs_prefix_id = 0
        for ind, (patient, data) in enumerate(alldata.items()):
            if ind == val_ind:
                for svs_prefix, item in data.items():
                    all_svs_prefixes.append(svs_prefix)
                    all_inds['val'].append(svs_prefix_id*np.ones((item['coord_df'].shape[0],),dtype=np.uint8))
                    all_coord_df['val'].append(item['coord_df'])
                    vst_df = clean_vst(item['vst_df'], gene_names)
                    if use_vst_smooth:
                        all_vst_df['val'].append(neighbor_smoothing_vst(vst_df))
                    else:
                        all_vst_df['val'].append(vst_df)
                    svs_prefix_id += 1
            else:
                for svs_prefix, item in data.items():
                    all_svs_prefixes.append(svs_prefix)
                    all_inds['train'].append(svs_prefix_id*np.ones((item['coord_df'].shape[0],),dtype=np.uint8))
                    all_coord_df['train'].append(item['coord_df'])
                    vst_df = clean_vst(item['vst_df'], gene_names)
                    if use_vst_smooth:
                        all_vst_df['train'].append(neighbor_smoothing_vst(vst_df))
                    else:
                        all_vst_df['train'].append(vst_df)
                    mean.append(item['mean'])
                    std.append(item['std'])
                    svs_prefix_id += 1
        mean = np.stack(mean, axis=0).mean(axis=0).tolist()
        std = np.stack(std, axis=0).mean(axis=0).tolist()

        for subset in ['train', 'val']:
            all_inds[subset] = np.concatenate(all_inds[subset], axis=0)
            all_coord_df[subset] = pd.concat(all_coord_df[subset], axis=0).astype(np.int32)
            all_vst_df[subset] = pd.concat(all_vst_df[subset], axis=0).astype(np.float32)
    else:

        patient = '10x_Visium_FFPE_Human_Breast_Cancer_1.3.0'
        pdata = tmp['alldata'][patient][patient]
        coord_df = pdata['coord_df']
        vst_df = clean_vst(pdata['vst_df'], gene_names)

        X_col_name, Y_col_name = 'pxl_col_in_fullres', 'pxl_row_in_fullres'
        xs, ys = coord_df[X_col_name], coord_df[Y_col_name]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        xcuts = np.arange(xmin, xmax, step=100)
        for i, xcut in enumerate(xcuts):
            left_inds = np.where(xs<xcut-50)[0]
            right_inds = np.where(xs>xcut+50)[0]
            # print(len(left_inds), len(right_inds))
            ratio = len(left_inds) / (len(left_inds) + len(right_inds))
            if ratio > 0.7:
                break
        
        coord_df_l = coord_df[coord_df[X_col_name]<xcut-50]
        coord_df_r = coord_df[coord_df[X_col_name]>xcut+50]
        vst_df_l = vst_df.loc[coord_df_l.index.values]
        vst_df_r = vst_df.loc[coord_df_r.index.values]

        coord_df_l, vst_df_l = convert_index_names_10x(coord_df_l, vst_df_l)
        coord_df_r, vst_df_r = convert_index_names_10x(coord_df_r, vst_df_r)

        coord_df_l = coord_df_l[[X_col_name, Y_col_name]]
        coord_df_r = coord_df_r[[X_col_name, Y_col_name]]

        if use_vst_smooth:
            vst_df_l = neighbor_smoothing_vst(vst_df_l, use_10x=True)
            vst_df_r = neighbor_smoothing_vst(vst_df_r, use_10x=True)
        
        all_svs_prefixes = [patient]
        all_inds = {
            'train': [0 for _ in range(len(vst_df_l))], 
            'val': [0 for _ in range(len(vst_df_r))]
        }
        all_coord_df = {
            'train': coord_df_l.astype(np.int32), 
            'val': coord_df_r.astype(np.int32)
        }
        all_vst_df = {
            'train': vst_df_l.astype(np.float32), 
            'val': vst_df_r.astype(np.float32)
        }

        if idr_torch.rank == 0:
            with open(os.path.join(save_root, 'train_val_data.pkl'), 'wb') as fp:
                pickle.dump({'all_vst_df': all_vst_df, 'all_coord_df': all_coord_df, 'patient': patient}, fp)
    
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

    train_dataset = PatchDataset(all_svs_prefixes, all_inds['train'], all_coord_df['train'], all_vst_df['train'], transform=train_transform, is_train=True, cache_root=cache_root)
    val_dataset = PatchDataset(all_svs_prefixes, all_inds['val'], all_coord_df['val'], all_vst_df['val'], transform=val_transform, is_train=False, cache_root=cache_root)
    
    model = STModel(backbone=backbone, num_outputs=len(gene_names))

    if fixed_backbone:
        for param in model.backbone_model.parameters():
            param.requires_grad = False

    # lr = 5e-4
    weight_decay = 1e-5
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    backbone_params = [param for name, param in model.named_parameters() if 'backbone_model' in str(name)]
    other_params = [param for name, param in model.named_parameters() if 'backbone_model' not in str(name)]
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': lr}, 
        {'params': other_params, 'lr': lr*10}
    ], lr=lr, weight_decay=weight_decay)

    return train_dataset, val_dataset, model, optimizer, gene_names


def test_main(args):
    num_gpus = args.num_gpus
    data_root = args.data_root
    backbone = args.backbone
    lr = args.lr
    batch_size = args.batch_size
    fixed_backbone = args.fixed_backbone == 'True'
    use_vst_smooth = args.use_vst_smooth == 'True'
    val_ind = args.val_ind

    max_epochs = args.max_epochs
    save_every = args.save_every
    accum_iter = args.accum_iter

    save_root = f'{data_root}/results/val_{val_ind}/gpus{num_gpus}/backbone{backbone}_fixed{fixed_backbone}/lr{lr}_b{batch_size}_e{max_epochs}_accum{accum_iter}_v0_smooth{use_vst_smooth}'
    
    ckpt_path = f'{save_root}/snapshot_99.pt'

    if not os.path.exists(ckpt_path):
        print('no ckpt file')
        sys.exit(-1)
    else:
        print(ckpt_path)
    save_dir = os.path.dirname(ckpt_path)

    data_root = '/home/zhongz2/ST_prediction/data/10x/cache_data/data_20241003'
    cache_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'ST_prediction_data_10x')
    X_col_name = 'pxl_col_in_fullres'
    Y_col_name = 'pxl_row_in_fullres'
    gpu_id = 0

    with open(os.path.join(data_root, 'meta.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        alldata = tmp['alldata']
        gene_names = tmp['gene_names']
        low_thres, high_thres = tmp['gene_thres']
    
    if len(glob.glob(os.path.join(save_dir, '10x*_result.csv'))) == len(alldata):
        print('already done')
        sys.exit(0)

    model = STModel(backbone=backbone, num_outputs=len(gene_names))
    state_dict = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(state_dict['MODEL_STATE'])
    model.to(gpu_id)
    model.eval()

    if True:  # use imagenet mean and std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    for ind, (patient, data) in enumerate(alldata.items()):
        save_filename = os.path.join(save_dir, patient+'_result.csv')
        if os.path.exists(save_filename):
            continue

        print('begin ', patient)
        svs_prefix_id = 0
        all_svs_prefixes = []
        all_inds = {'train': [], 'val': []}
        all_coord_df = {'train': [], 'val': []}
        all_vst_df = {'train': [], 'val': []}

        for svs_prefix, item in data.items():
            all_svs_prefixes.append(svs_prefix)
            all_inds['val'].append(svs_prefix_id*np.ones((item['coord_df'].shape[0],),dtype=np.uint8))
            all_coord_df['val'].append(item['coord_df'][[X_col_name, Y_col_name]])
            vst_df = clean_vst(item['vst_df'], gene_names)
            if False: # use_vst_smooth:
                all_vst_df['val'].append(neighbor_smoothing_vst(vst_df))
            else:
                all_vst_df['val'].append(vst_df)
            svs_prefix_id += 1
        
        for subset in ['val']:
            all_inds[subset] = np.concatenate(all_inds[subset], axis=0)
            all_coord_df[subset] = pd.concat(all_coord_df[subset], axis=0).astype(np.int32)
            all_vst_df[subset] = pd.concat(all_vst_df[subset], axis=0).astype(np.float32)

        val_dataset = PatchDataset(all_svs_prefixes, all_inds['val'], all_coord_df['val'], all_vst_df['val'], transform=val_transform, is_train=False, cache_root=cache_root)

        val_dataloader = DataLoader(val_dataset, num_workers=2, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=False)

        preds = []
        for batch_idx, (images_batch, labels_batch) in enumerate(val_dataloader):
            images_batch = images_batch.to(gpu_id)
            with torch.no_grad():
                preds_batch = model(images_batch)
            preds.append(preds_batch)
        preds = torch.cat(preds).detach().cpu().numpy()
        gt = all_vst_df['val']
        pred = gt.copy()
        pred.iloc[:, :] = preds

        columns_with_nan = gt.columns[gt.isna().any()].tolist()
        valid_columns = [col for col in gt.columns if col not in columns_with_nan]
        gt = gt[valid_columns]
        pred = pred[valid_columns]
        all_labels = gt.values
        all_preds = pred.values

        r2_scores = []
        spearmanr_corrs = []
        spearmanr_pvals = [] 
        pearsonr_corrs = []
        pearsonr_pvals = []
        for j in range(all_preds.shape[1]):
            r2_scores.append(r2_score(all_labels[:, j], all_preds[:, j]))
            res = spearmanr(all_preds[:, j], all_labels[:, j])
            spearmanr_corrs.append(res.statistic)
            spearmanr_pvals.append(res.pvalue)

            res = pearsonr(all_preds[:, j], all_labels[:, j])
            pearsonr_corrs.append(res.statistic)
            pearsonr_pvals.append(res.pvalue)

        res_df = pd.DataFrame({
            'gene_name': gt.columns,
            'r2': r2_scores, 
            'spearman_corr': spearmanr_corrs, 
            'spearman_pval': spearmanr_pvals,
            'pearsonr_corr': pearsonr_corrs,
            'pearsonr_pval': pearsonr_pvals
        })

        res_df.sort_values('pearsonr_corr', ascending=False).to_csv(save_filename, index=False)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--fixed_backbone', type=str, default='False')
    parser.add_argument('--use_vst_smooth', type=str, default='False')
    parser.add_argument('--val_ind', type=int, default=0)

    return parser.parse_args()


def train_main(args):

    num_gpus = args.num_gpus
    data_root = args.data_root
    backbone = args.backbone
    lr = args.lr
    batch_size = args.batch_size
    fixed_backbone = args.fixed_backbone == 'True'
    use_vst_smooth = args.use_vst_smooth == 'True'
    val_ind = args.val_ind

    max_epochs = args.max_epochs
    save_every = args.save_every
    accum_iter = args.accum_iter

    save_root = f'{data_root}/results/val_{val_ind}/gpus{num_gpus}/backbone{backbone}_fixed{fixed_backbone}/lr{lr}_b{batch_size}_e{max_epochs}_accum{accum_iter}_v0_smooth{use_vst_smooth}'
    
    os.makedirs(save_root, exist_ok=True)

    if os.path.exists(os.path.join(save_root, 'snapshot_99.pt')):
        print('done')
        sys.exit(-1)

    # local directories
    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        cache_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'ST_prediction_data')  # first decompress data into these directories
    else:
        cache_root = os.path.join('/tmp/zhongz2', os.environ['SLURM_JOB_ID'], 'ST_prediction_data')
    os.makedirs(cache_root, exist_ok=True)

    ddp_setup()

    train_dataset, val_dataset, model, optimizer, gene_names = \
        load_train_objs(data_root=data_root, backbone=backbone, lr=lr, fixed_backbone=fixed_backbone, val_ind=val_ind, cache_root=cache_root, 
        use_vst_smooth=use_vst_smooth, save_root=save_root)

    dataloaders = {
        'train':
            DataLoader(train_dataset, num_workers=2, batch_size=batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)),
        'val':
            DataLoader(val_dataset, num_workers=2, batch_size=batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False)),
    }
    trainer = Trainer(model, dataloaders, optimizer, save_root=save_root, save_every=save_every, accum_iter=accum_iter, gene_names=gene_names)
    trainer.train(max_epochs=max_epochs)
    dist.destroy_process_group()


def plot_curves(args):
    val_ind = idr_torch.rank

    debug = False
    if not debug and val_ind<0 or val_ind>22:
        sys.exit(0)

    root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240925_secreted/results/val_0/gpus2'
    data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240925_secreted'

    # root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_all/data_images_He2020_224_20240920_all/results/val_0/gpus2'
    # data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_all/data_images_He2020_224_20240920_all/'

    # root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_v1/results/val_0/gpus2'
    # data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_v1'

    root = './data/He2020/cache_data/data_224_20240927_v2/results/val_0/gpus2/'
    data_root = './data/He2020/cache_data/data_224_20240927_v2/'

    root = '/home/zhongz2/ST_prediction/data/He2020/cache_data/data_224_20240927_v3/results/val_0/gpus2'
    data_root = '/home/zhongz2/ST_prediction/data/He2020/cache_data/data_224_20240927_v3/'

    data_root = './data/He2020/cache_data/data_224_20241002/'

    # check the data, plot hist
    with open(os.path.join(data_root, 'meta.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        alldata = tmp['alldata']
        gene_names = tmp['gene_names'] if 'gene_names' in tmp else tmp['selected_gene_names']
        low_thres, high_thres = tmp['gene_thres']
    bins = np.arange(low_thres, high_thres, 1)

    root = f'./data/He2020/cache_data/data_224_20241002/results/val_{val_ind}/gpus2/'
    save_filename = os.path.join(root, '..', '..', f'val_{val_ind}_figures_sum_new.tar.gz')
    if not debug and os.path.exists(save_filename):
        sys.exit(0)

    print(root)

    all_svs_prefixes = []
    all_inds = {'train': [], 'val': []}
    all_coord_df = {'train': [], 'val': []}
    all_vst_df = {'train': [], 'val': []}
    
    mean, std = [], []
    svs_prefix_id = 0
    for ind, (patient, data) in enumerate(alldata.items()):
        if ind == val_ind:
            for svs_prefix, item in data.items():
                all_svs_prefixes.append(svs_prefix)
                all_inds['val'].append(svs_prefix_id*np.ones((item['coord_df'].shape[0],),dtype=np.uint8))
                all_coord_df['val'].append(item['coord_df'])
                all_vst_df['val'].append(item['vst_df'])
                svs_prefix_id += 1
        else:
            for svs_prefix, item in data.items():
                all_svs_prefixes.append(svs_prefix)
                all_inds['train'].append(svs_prefix_id*np.ones((item['coord_df'].shape[0],),dtype=np.uint8))
                all_coord_df['train'].append(item['coord_df'])
                all_vst_df['train'].append(item['vst_df'])
                mean.append(item['mean'])
                std.append(item['std'])
                svs_prefix_id += 1
    mean = np.stack(mean, axis=0).mean(axis=0).tolist()
    std = np.stack(std, axis=0).mean(axis=0).tolist()

    for subset in ['train', 'val']:
        all_inds[subset] = np.concatenate(all_inds[subset], axis=0)
        all_coord_df[subset] = pd.concat(all_coord_df[subset], axis=0).astype(np.int32)
        all_vst_df[subset] = pd.concat(all_vst_df[subset], axis=0).astype(np.float32)

    trn_sum = all_vst_df['train'].sum(axis=0)
    trn_mean = all_vst_df['train'].mean(axis=0)

    backbones = ['resnet50']
    learning_rates = ['0.0001', '1e-05', '1e-06', '5e-05', '5e-06']
    batch_sizes = [32, 64]
    if debug:
        backbones = ['resnet50']
        learning_rates = ['1e-05']
        batch_sizes = [32]

    index_map = {
    '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1':'Colorectal',
       '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1':'Glioblastoma',
       '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1': 'Kidney',
       '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1': 'Lung',
       '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0': 'OvarianCarci',
       '10x_CytAssist_FFPE_Human_Colon_Rep1_2.1.0': 'Colon_Rep1',
       '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0': 'LSCC',
       '10x_Visium_FFPE_Human_Breast_Cancer_1.3.0': 'Breast',
       '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0':'Cervical',
       '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0':'Intestinal',
       '10x_Visium_FFPE_Human_Normal_Prostate_1.3.0':'Prostate',
       '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0':'OvarianCancer',
       '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0':'PACC',
       '10x_Visium_FFPE_Human_Prostate_Cancer_1.3.0':'ProstateCancer'
    }


    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

    for lr in learning_rates:
        for bs in batch_sizes:
            for backbone in backbones:

                save_prefix = f'val_{val_ind}/backbone{backbone}_lr{lr}_bs{bs}'
                results = {}
                for fixed in ['True', 'False']:
                    for smooth in ['True', 'False']:
                        d = os.path.join(root, f'backbone{backbone}_fixed{fixed}', f'lr{lr}_b{bs}_e100_accum1_v0_smooth{smooth}')
                        if not os.path.exists(os.path.join(d, 'snapshot_99.pt')):
                            continue
                        r2 = pd.read_csv(os.path.join(d, 'val_r2score.csv'), index_col=0)
                        pearson = pd.read_csv(os.path.join(d, 'val_pearsonr_corr.csv'), index_col=0)
                        spearman = pd.read_csv(os.path.join(d, 'val_spearmanr_corr.csv'), index_col=0)
                        results_10x = {}
                        for filename in glob.glob(os.path.join(d, '10x_*result.csv')):
                            results_10x[os.path.basename(filename).replace('_result.csv', '')] = pd.read_csv(filename, index_col=0)
                        if debug: 
                            print(d, len(results_10x))
                        if len(results_10x)>0:
                            results[f'fixed{fixed}_smooth{smooth}'] = {
                                'r2': r2, 'pearson': pearson, 'spearman': spearman,
                                'results_10x': results_10x
                            }

                if len(results) != 4:
                    continue
                elif debug: 
                    print(save_prefix, len(results))
                
                r2 = results['fixedFalse_smoothTrue']['r2']
                s = r2.sum(axis=0)
                # s = r2.max(axis=0)
                sorted_names = s.sort_values(ascending=False).index

                for j,gene_name in enumerate(sorted_names[:min(50, r2.shape[1])]):

                    fig = plt.figure(figsize=(2*4, (1+len(results))*4))
                    gs = GridSpec(nrows=5, ncols=2, figure=fig)

                    ax = fig.add_subplot(gs[0, :])
                    all_vst_df['train'][gene_name].hist(bins=bins, ax=ax)
                    all_vst_df['val'][gene_name].hist(bins=bins, ax=ax)
                    ax.set_title('{}, rank = {}'.format(gene_name, j+1))

                    for i, (k, v) in enumerate(results.items()):
                        # rr, cc = (i+2)//2, (i+2)%2
                        ax = fig.add_subplot(gs[i+1, 0])
                        v['r2'][gene_name].plot(ax=ax)
                        v['spearman'][gene_name].plot(ax=ax)
                        v['pearson'][gene_name].plot(ax=ax)
                        ax.legend(['r2', 'spearman', 'pearson'])
                        ax.set_title(k)
                    
                        result_10x = []
                        for kk, vv in v['results_10x'].items():
                            try:
                                result_10x.append([kk] + vv.loc[gene_name].values.tolist())
                            except:
                                if debug: 
                                    print(gene_name, ' not in {}'.format(k))
                                pass
                        if len(result_10x) == 0:
                            if debug: 
                                print(i, k)
                            continue

                        result_10x_df = pd.DataFrame(result_10x, columns=['svs_prefix'] + vv.columns.values.tolist())
                        result_10x_df = result_10x_df.set_index('svs_prefix')[['r2',  'spearman_corr',  'pearsonr_corr']]
                        result_10x_df.index = result_10x_df.index.map(index_map)
                        ax = fig.add_subplot(gs[i+1, 1])

                        # Plot each column's data with a different marker and label
                        ax.scatter(result_10x_df.index, result_10x_df['r2'], label='r2', marker='o', color='r')
                        ax.scatter(result_10x_df.index, result_10x_df['spearman_corr'], label='spearman', marker='s', color='g')
                        ax.scatter(result_10x_df.index, result_10x_df['pearsonr_corr'], label='pearson', marker='^', color='b')
                        ax.tick_params('x', labelrotation=90)
                        ax.legend(['r2', 'spearman', 'pearson'])

                    plt.tight_layout()

                    im_buffer = io.BytesIO()
                    plt.savefig(im_buffer, format='JPEG')
                    info = tarfile.TarInfo(name=os.path.join(save_prefix, 'val_{:02d}_{}.jpg'.format(j+1, gene_name)))
                    info.size = im_buffer.getbuffer().nbytes
                    info.mtime = time.time()
                    im_buffer.seek(0)
                    tar_fp.addfile(info, im_buffer)

                    plt.close('all')

    tar_fp.close()
    with open(save_filename, 'wb') as fp:
        fp.write(fh.getvalue())

    if False:

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        files = glob.glob(f'{root}/**/**/*_99.pt')   

        with open('/home/zhongz2/ST_prediction/data/10x/cache_data/data_20241003/meta.pkl', 'rb') as fp:
            tmp = pickle.load(fp)
            bins = tmp['bins'] 
            del tmp
          
        for f in files:
            d = os.path.dirname(f)

            try:
                with open(os.path.join(d, 'train_val_data.pkl'), 'rb') as fp:
                    tmp = pickle.load(fp)
                    all_vst_df = tmp['all_vst_df']
                    del tmp
            except:
                all_vst_df = None

            r2 = pd.read_csv(os.path.join(d, 'val_r2score.csv'), index_col=0)
            pearson = pd.read_csv(os.path.join(d, 'val_pearsonr_corr.csv'), index_col=0)
            spearman = pd.read_csv(os.path.join(d, 'val_spearmanr_corr.csv'), index_col=0)

            s = r2.sum(axis=0)
            # s = r2.max(axis=0)
            sorted_names = s.sort_values(ascending=False).index
            r2 = r2[sorted_names]
            pearson = pearson[sorted_names]
            spearman = spearman[sorted_names]

            for j,gene_name in enumerate(sorted_names): #[:min(50, r2.shape[1])]):
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
                info = tarfile.TarInfo(name=os.path.join(d, 'val_{:02d}_{}.jpg'.format(j+1, gene_name)))
                info.size = im_buffer.getbuffer().nbytes
                info.mtime = time.time()
                im_buffer.seek(0)
                tar_fp.addfile(info, im_buffer)

                plt.close()

        tar_fp.close()

        save_filename = '/data/zhongz2/temp29/10x_res.tar.gz'
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())



if __name__ == '__main__':
    args = get_args()
    setup_seed(2024)

    if args.action == 'train':
        train_main(args=args) 
    elif args.action == 'test':
        test_main(args=args)
    elif args.action == 'plot':
        plot_curves(args=args)
