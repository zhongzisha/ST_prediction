
import sys,os,shutil,json,h5py,glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import openslide
import pickle
import random
import io
import tarfile
import time
from urllib.parse import urlsplit
from sklearn.metrics import r2_score, confusion_matrix, roc_auc_score
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
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torcheval.metrics.functional import r2_score as r2_score_pytorch
from torcheval.metrics.functional import auc as auc_pytorch
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


def weighted_mse_loss(input, target, weight):
    return (weight * (input - target) ** 2)

def calculate_topk_accuracy(y_pred, y, k = 5):
    with torch.no_grad():
        batch_size, num = y.shape[:2]
        acc_1s, acc_ks = [], []
        for j in range(num):
            _, top_pred = y_pred[:, j, :].topk(k, 1)
            top_pred = top_pred.t()
            correct = top_pred.eq(y[:, j].view(1, -1).expand_as(top_pred))
            correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim = True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim = True)
            acc_1s.append(correct_1.item() / batch_size)
            acc_ks.append(correct_k.item() / batch_size)
    return acc_1s, acc_ks

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
        self.num_classes = model.num_classes
        self.num_genes = model.num_genes
        self.bin_labels = np.array([-2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.class_weights = torch.tensor([
            100.0,
            5.902011951294695,
            1.4845647080816549,
            13.242422708556559,
            22.401221928314527,
            50.51746899723922,
            61.26961281548879
        ], dtype=torch.float32).to(self.gpu_id)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

        # self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.HuberLoss()

        if self.gpu_id == 0:
            save_dirs = {}
            for subset in self.dataloaders.keys():
                save_dirs[subset] = os.path.join(save_root, subset)
                os.makedirs(save_dirs[subset], exist_ok=True)
            self.save_dirs = save_dirs

            self.loss_dicts = {subset: {'acc_1': [], 'acc_5': [], 'auc': [], 'loss': [], 'r2score': [], 'spearmanr_corr': [], 'spearmanr_pvalue': [], 'pearsonr_corr': [], 'pearsonr_pvalue': []} for subset in self.dataloaders.keys()}

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
        reg_labels = []
        total_loss = torch.tensor([0.], dtype=torch.float32, device=self.gpu_id)
        for batch_idx, (images_batch, labels_batch, labels_batch_reg) in enumerate(self.dataloaders[subset]):
            images_batch = images_batch.to(self.gpu_id)
            labels_batch = labels_batch.to(self.gpu_id)
            labels_batch_reg = labels_batch_reg.to(self.gpu_id)

            if is_train:
                preds_batch = self.model(images_batch)
            else:
                with torch.no_grad():
                    preds_batch = self.model(images_batch)

            preds_batch = preds_batch.reshape(-1, self.num_classes)
            labels_batch = labels_batch.reshape(-1)
            loss = self.loss_fn(preds_batch, labels_batch)

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
                reg_labels.append(labels_batch_reg)

            total_loss[0] += loss

        total_loss = collect_results_gpu(total_loss, 1, world_size=self.world_size)
        total_loss = total_loss.sum()
        if is_train:
            if self.gpu_id == 0:
                self.loss_dicts[subset]['loss'].append([epoch, total_loss.item()])
        else:
            labels = torch.cat(labels)
            preds = torch.cat(preds)
            reg_labels = torch.cat(reg_labels)


            all_labels = collect_results_gpu(labels, self.num_genes*len(dataset), world_size=self.world_size)
            all_logits = collect_results_gpu(preds, self.num_genes*len(dataset), world_size=self.world_size)
            all_reg_labels = collect_results_gpu(reg_labels, len(dataset), world_size=self.world_size)
            
            if self.gpu_id == 0:

                self.loss_dicts[subset]['loss'].append([epoch, total_loss.item()])
                if False:

                    scores = r2_score_pytorch(all_logits, all_labels, multioutput='raw_values')
                    all_logits = all_logits.detach().cpu().numpy()
                    all_labels = all_labels.detach().cpu().numpy()
                    self.loss_dicts[subset]['r2score'].append(scores.detach().cpu().numpy().tolist())

                    spearmanr_corrs = []
                    spearmanr_pvals = [] 
                    pearsonr_corrs = []
                    pearsonr_pvals = []
                    for j in range(all_logits.shape[1]):
                        res = spearmanr(all_logits[:, j], all_labels[:, j])
                        spearmanr_corrs.append(res.statistic)
                        spearmanr_pvals.append(res.pvalue)

                        res = pearsonr(all_logits[:, j], all_labels[:, j])
                        pearsonr_corrs.append(res.statistic)
                        pearsonr_pvals.append(res.pvalue)

                    self.loss_dicts[subset]['spearmanr_corr'].append(spearmanr_corrs)
                    self.loss_dicts[subset]['spearmanr_pvalue'].append(spearmanr_pvals)
                    self.loss_dicts[subset]['pearsonr_corr'].append(pearsonr_corrs)
                    self.loss_dicts[subset]['pearsonr_pvalue'].append(pearsonr_pvals)
                else:
                    all_logits = all_logits.reshape(-1, self.num_genes, self.num_classes)
                    all_labels = all_labels.reshape(-1, self.num_genes) # B, G
                    # acc_1, acc_5 = calculate_topk_accuracy(all_logits, all_labels)
                    all_probs = F.softmax(all_logits, dim=-1)  # B, G, C
                    all_preds = all_probs.argmax(-1).detach().cpu().numpy()  # B, G
                    # self.loss_dicts[subset]['acc_1'].append(acc_1)
                    # self.loss_dicts[subset]['acc_5'].append(acc_5)
                    all_labels = all_labels.detach().cpu().numpy()
                    all_probs = all_probs.detach().cpu().numpy()
                    cmat = confusion_matrix(y_true=all_labels.reshape(-1), y_pred=all_preds.reshape(-1))
                    print(cmat)
                    # aucs = np.zeros((self.num_genes,),dtype=np.float32)
                    # for j in range(self.num_genes):
                    #     auc = roc_auc_score(y_true=all_labels[:, j], y_score=all_probs[:, j, :], average='weighted', multi_class='ovo',
                    #                         labels=np.arange(len(self.bin_labels)))
                    #     aucs[j] = auc
                    # self.loss_dicts[subset]['auc'].append(aucs)
                    all_reg_preds = torch.from_numpy(self.bin_labels[all_preds]).to(self.gpu_id)

                    print('all_reg_preds', all_reg_preds)
                    print('all_reg_labels', all_reg_labels)
                    
                    scores = r2_score_pytorch(all_reg_preds, all_reg_labels, multioutput='raw_values')
                    all_reg_preds = all_reg_preds.detach().cpu().numpy()
                    all_reg_labels = all_reg_labels.detach().cpu().numpy()
                    self.loss_dicts[subset]['r2score'].append(scores.detach().cpu().numpy().tolist())

                    spearmanr_corrs = []
                    spearmanr_pvals = [] 
                    pearsonr_corrs = []
                    pearsonr_pvals = []
                    for j in range(all_reg_preds.shape[1]):
                        res = spearmanr(all_reg_preds[:, j], all_reg_labels[:, j])
                        spearmanr_corrs.append(res.statistic)
                        spearmanr_pvals.append(res.pvalue)

                        res = pearsonr(all_reg_preds[:, j], all_reg_labels[:, j])
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
                    if subset == 'val' and 'r2' in name:
                        print(log_df.values[-1, :])
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
    def __init__(self, backbone='resnet50', dropout=0.25, num_genes=10000, num_classes=7):
        super().__init__()
        self.backbone = backbone
        self.num_genes = num_genes
        self.num_classes = num_classes

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

        self.fc = nn.Linear(512, num_genes*num_classes)
        # self.fc = nn.Linear(BACKBONE_DICT[backbone], num_genes)

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

        h = h.reshape(-1, self.num_genes, self.num_classes)

        return h





class PatchDataset(Dataset):
    def __init__(self, data, transform, is_train=False, cache_root='./'):
        super().__init__()
        self.data = data
        self.transform = transform 
        self.is_train = is_train
        self.cache_root = cache_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        img_path, txt_path = self.data[idx]
        patch = Image.open(os.path.join(self.cache_root, img_path))
        if self.is_train:
            if np.random.rand() < 0.5:
                patch = patch.rotate(np.random.choice([90, 180, 270]))
            # if np.random.rand() < 0.2:
            #     patch = patch.filter(ImageFilter.GaussianBlur(radius=np.random.randint(low=1,high=50)/100.)) 
        with open(os.path.join(self.cache_root, txt_path), 'r') as fp:
            cls_label = torch.tensor([int(float(v)) for v in fp.readline().split(',')])
        with open(os.path.join(self.cache_root, txt_path.replace('_cls', '')), 'r') as fp:
            reg_label = torch.tensor([float(v) for v in fp.readline().split(',')])
        return self.transform(patch), cls_label, reg_label


def load_train_objs(data_root='./data', backbone='resnet50', lr=1e-4, fixed_backbone=False, val_ind=0, cache_root='./'): 

    with open(os.path.join(data_root, 'meta.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        alldata = tmp['alldata']
        gene_names = tmp['gene_names']
        low_thres, high_thres = tmp['gene_thres']
        bins = tmp['bins']
        del tmp

    train_data = []
    val_data = []
    mean, std = [], []
    for ind, (patient, data) in enumerate(alldata.items()):
        if ind == val_ind:
            for item in data.values():
                val_data.extend(item['data'])
        else:
            for item in data.values():
                train_data.extend(item['data'])
                mean.append(item['mean'])
                std.append(item['std'])
    mean = np.stack(mean, axis=0).mean(axis=0).tolist()
    std = np.stack(std, axis=0).mean(axis=0).tolist()

    if False:  # use imagenet mean and std
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


    train_dataset = PatchDataset(train_data, transform=train_transform, is_train=True, cache_root=cache_root)
    val_dataset = PatchDataset(val_data, transform=val_transform, is_train=False, cache_root=cache_root)
    
    model = STModel(backbone=backbone, num_genes=len(gene_names), num_classes=len(bins)-1)

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


def train_main():
    num_gpus = int(sys.argv[1])
    data_root = sys.argv[2]
    backbone = sys.argv[3]
    lr = float(sys.argv[4])
    batch_size = int(sys.argv[5])
    fixed_backbone = sys.argv[6] == 'True'
    val_ind = int(sys.argv[7])
    max_epochs = 100
    save_every = 200
    accum_iter = 1
    save_root = f'{data_root}/results/val_{val_ind}/gpus{num_gpus}/backbone{backbone}_fixed{fixed_backbone}/lr{lr}_b{batch_size}_e{max_epochs}_accum{accum_iter}_v1'
    os.makedirs(save_root, exist_ok=True)

    # local directories
    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        cache_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'ST_prediction_data')  # first decompress data into these directories
    else:
        cache_root = os.path.join('/tmp/zhongz2', os.environ['SLURM_JOB_ID'], 'ST_prediction_data')
    os.makedirs(cache_root, exist_ok=True)

    ddp_setup()
    train_dataset, val_dataset, model, optimizer, gene_names = \
        load_train_objs(data_root=data_root, backbone=backbone, lr=lr, fixed_backbone=fixed_backbone, val_ind=val_ind, cache_root=cache_root)

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


if __name__ == '__main__':
    setup_seed(2024)
    train_main() 


"""

NUM_GPUS=2
torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29898 \
    ST_prediction_exps_v4.py ${NUM_GPUS} ./data/He2020/cache_data/data_224_20240926 resnet50 5e-6 64 True 0

"""

def plot_curves():

    import glob,os,io,tarfile,time,pickle
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240925_secreted/results/val_0/gpus2'
    data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240925_secreted'

    # root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_all/data_images_He2020_224_20240920_all/results/val_0/gpus2'
    # data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_all/data_images_He2020_224_20240920_all/'

    # root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_v1/results/val_0/gpus2'
    # data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240920_v1'

    root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240926_secreted/results/val_0/gpus2'
    data_root = '/data/zhongz2/temp29/ST_prediction/data/He2020/cache_data/data_224_20240926_secreted'

    # check the data, plot hist
    with open(os.path.join(data_root, 'meta.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        alldata = tmp['alldata']
        gene_names = tmp['gene_names'] if 'gene_names' in tmp else tmp['selected_gene_names']
        low_thres, high_thres = tmp['gene_thres']
    bins = np.arange(low_thres, high_thres, 1)

    val_ind = 0
    train_data = []
    val_data = []
    mean, std = [], []
    for ind, (patient, data) in enumerate(alldata.items()):
        if ind == val_ind:
            for item in data.values():
                val_data.extend(item['data'])
        else:
            for item in data.values():
                train_data.extend(item['data'])
                mean.append(item['mean'])
                std.append(item['std'])
    mean = np.stack(mean, axis=0).mean(axis=0).tolist()
    std = np.stack(std, axis=0).mean(axis=0).tolist()

    trnY, valY = [], []
    for item in train_data:
        with open(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'ST_prediction_data', item[1]), 'r') as fp:
            trnY.append([float(v) for v in fp.readline().split(',')])
    for item in val_data:
        with open(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'ST_prediction_data', item[1]), 'r') as fp:
            valY.append([float(v) for v in fp.readline().split(',')])
    # trnY = [item[1] for item in train_data]
    # valY = [item[1] for item in val_data]
    trn_df = pd.DataFrame(np.array(trnY), columns=gene_names)
    val_df = pd.DataFrame(np.array(valY), columns=gene_names)

    trn_sum = trn_df.sum(axis=0)
    trn_mean = trn_df.mean(axis=0)

    files = glob.glob(f'{root}/**/**/*_99.pt')
    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
    
    for f in files:
        d = os.path.dirname(f)

        r2 = pd.read_csv(os.path.join(d, 'val_r2score.csv'), index_col=0)
        pearson = pd.read_csv(os.path.join(d, 'val_pearsonr_corr.csv'), index_col=0)
        spearman = pd.read_csv(os.path.join(d, 'val_spearmanr_corr.csv'), index_col=0)

        s = r2.sum(axis=0)
        # s = r2.max(axis=0)
        sorted_names = s.sort_values(ascending=False).index
        r2 = r2[sorted_names]
        pearson = pearson[sorted_names]
        spearman = spearman[sorted_names]

        for j,gene_name in enumerate(sorted_names[:min(50, r2.shape[1])]):
            fig, axes = plt.subplots(nrows=1, ncols=2)
            ax = axes[0]
            r2[gene_name].plot(ax=ax)
            spearman[gene_name].plot(ax=ax)
            pearson[gene_name].plot(ax=ax)
            ax.legend(['r2', 'spearman', 'pearson'])
            ax.set_title('{}, rank = {}'.format(gene_name, j+1))
            # plt.savefig(os.path.join(d, 'val_{:02d}_{}.jpg'.format(j+1, gene_name)))

            ax = axes[1]
            trn_df[gene_name].hist(bins=bins, ax=ax)
            val_df[gene_name].hist(bins=bins, ax=ax)

            im_buffer = io.BytesIO()
            plt.savefig(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=os.path.join(d, 'val_{:02d}_{}.jpg'.format(j+1, gene_name)))
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)

            plt.close()

    tar_fp.close()
    with open(os.path.join(root, 'val_figures_sum.tar.gz'), 'wb') as fp:
        fp.write(fh.getvalue())



def main():

    device = torch.device('cuda:0')

    train_dataset, val_dataset, model, optimizer = load_train_objs()

    model.to(device)

    train_loader = DataLoader(train_dataset, num_workers=2, batch_size=4, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, num_workers=2, batch_size=4, shuffle=False, drop_last=False)

    loss_fn = nn.MSELoss()

    accum_iter = 2

    log_strs = [] 
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        preds = model(images)

        mask = torch.isnan(labels)
        
        total_loss = loss_fn(preds[~mask], labels[~mask])

        total_loss = total_loss / accum_iter
        total_loss.backward()

        if (batch_idx + 1) % accum_iter == 0:
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()

    model.eval()

    all_labels = []
    all_preds = []
    for batch_idx, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        
        preds = model(images)

        all_labels.append(labels)
        all_preds.append(preds)

