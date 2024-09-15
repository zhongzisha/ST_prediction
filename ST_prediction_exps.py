
import sys,os,shutil,json,h5py,glob
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import openslide
import pickle
import io
import tarfile
import time
from sklearn.metrics import r2_score
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
torch.set_printoptions(sci_mode=False)
torch.multiprocessing.set_sharing_strategy('file_system')
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


with open('ST_gene_list.pkl', 'rb') as fp:
    gene_map_dict = {v: str(i) for i, v in enumerate(pickle.load(fp)['gene_list'])}
BACKBONE_DICT = {
    'resnet50': 2048,
    'CLIP': 512,
    'PLIP': 512,
    'MobileNetV3': 1280,
    'mobilenetv3': 1280,
    'ProvGigaPath': 1536,
    'CONCH': 512,
    'UNI': 1024
}
GLOBAL_MEAN = [0.75225115, 0.5662438 , 0.72874427]
GLOBAL_STD = [0.12278537, 0.14380322, 0.10359251]


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

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch, subset='train'):

        is_train = subset == 'train'
        if is_train:
            self.model.train()
            self.model.zero_grad()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        b_sz = len(next(iter(self.dataloaders[subset]))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloaders[subset])}")
        self.dataloaders[subset].sampler.set_epoch(epoch)
        dataset = self.dataloaders[subset].dataset

        labels = []
        preds = []
        train_loss = 0
        for batch_idx, (images_batch, labels_batch) in enumerate(self.dataloaders[subset]):
            images_batch = images_batch.to(self.gpu_id)
            labels_batch = labels_batch.to(self.gpu_id)

            if is_train:
                preds_batch = self.model(images_batch)
            else:
                with torch.no_grad():
                    preds_batch = self.model(images_batch)

            if is_train:
                mask = torch.isnan(labels_batch)
                total_loss = self.loss_fn(preds_batch[~mask], labels_batch[~mask])
                total_loss = total_loss / self.accum_iter
                total_loss.backward()

                train_loss += total_loss.item()

                if (batch_idx + 1) % self.accum_iter == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                    self.optimizer.zero_grad()
                
                if self.gpu_id == 0 and (batch_idx + 1) % 10 == 0:
                    print(total_loss.item())
            else:
                labels.append(labels_batch)
                preds.append(preds_batch)

        if is_train:
            print('train loss: ', train_loss)
        else:
            labels = torch.cat(labels)
            preds = torch.cat(preds)

            all_labels = collect_results_gpu(labels, len(dataset), world_size=self.world_size)
            all_preds = collect_results_gpu(preds, len(dataset), world_size=self.world_size)
            
            if self.gpu_id == 0:
                scores = r2_score_pytorch(all_preds, all_labels, multioutput='raw_values')
                scores = scores[~torch.isnan(scores)]
                scores, inds = torch.sort(scores)
                print('Bottom 10: ', scores[:10])
                print('Top 10: ', scores[-10:])
                print('Count(>0.2): ', len(torch.where(scores>0.2)[0]))

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
            from conch.open_clip_custom import create_model_from_pretrained
            self.backbone_model, self.image_processor = create_model_from_pretrained('conch_ViT-B-16','./backbones/CONCH_weights_pytorch_model.bin')
            self.transform = None
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
        
        if self.backbone in ['CONCH']:
            h = self.backbone_model.encode_image(x, proj_contrast=False, normalize=False)
        elif self.backbone in ['PLIP', 'CLIP']:
            h = self.backbone_model.get_image_features(x)
        elif self.backbone in ['resnet50', 'UNI', 'ProvGigaPath']:
            h = self.backbone_model(x)

        h = self.rho(h)

        h = self.fc(h)

        h = 8 * torch.tanh(h)  # [-8, 8]

        return h



class PatchDataset1(Dataset):
    def __init__(self, data, transform, is_train=False):
        super().__init__()
        self.data = data
        self.transform = transform 
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        patch = Image.open(self.data[idx])
        if self.is_train:
            if np.random.rand() < 0.5:
                patch = patch.rotate(np.random.choice([90, 180, 270]))
            if np.random.rand() < 0.2:
                patch = patch.filter(ImageFilter.GaussianBlur(radius=np.random.randint(low=1,high=50)/100.)) 

        with open(self.data[idx].replace('.jpg', '.txt'), 'r') as fp:
            label = torch.tensor([float(v) for v in fp.readline().split(',')])
        return self.transform(patch), label


def load_train_objs(val_ind=0, data_root='./data', lr=1e-4): 

    human_slide_ids = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Protein_Expression_Human_Tonsil_2.1.0',
        '10x_CytAssist_Fresh_Frozen_Human_Breast_Cancer_2.0.1',
        '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0',
        '10x_V1_Breast_Cancer_Block_A_Section_1_1.1.0',
        '10x_V1_Breast_Cancer_Block_A_Section_2_1.1.0',
        '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        '10x_Visium_Human_Breast_Cancer_1.3.0',
        'ST1K4M_Human_Breast_10X_06092021_Visium',
        'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep1',
        'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep2',
        'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep1',
        'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep2',
        'ST1K4M_Human_Prostate_10X_06092021_Visium_cancer',
        'ST1K4M_Human_Prostate_10X_06092021_Visium_normal',
        'ST1K4M_Human_Prostate_10X_07122022_Visium'
    ]

    invalid_prefixes = [
        'selected_gene_names', 'mean_std', 'meta'
    ]
    val_prefixes = [
        human_slide_ids[val_ind]
    ]
    train_data = []
    val_data = []
    # files = sorted(glob.glob(os.path.join(data_root, '*.pkl')))
    files = sorted(os.listdir(data_root))
    for d in files:
        if not os.path.isdir(os.path.join(data_root, d)):
            print(d, 'not dir')
            continue
        svs_prefix = d
        if svs_prefix in invalid_prefixes:
            continue
        samples = glob.glob(os.path.join(data_root, d, '*.jpg'))
        if svs_prefix in val_prefixes:
            val_data.extend(samples) 
        else:
            train_data.extend(samples) 

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    mean = GLOBAL_MEAN
    std = GLOBAL_STD
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

    train_dataset = PatchDataset1(train_data, transform=train_transform, is_train=True)
    val_dataset = PatchDataset1(val_data, transform=val_transform, is_train=False)

    model = STModel(backbone='resnet50')

    # for param in model.backbone_model.parameters():
    #     param.requires_grad = False

    # lr = 5e-4
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # backbone_params = [param for name, param in model.named_parameters() if 'backbone_model' in str(name)]
    # other_params = [param for name, param in model.named_parameters() if 'backbone_model' not in str(name)]
    # optimizer = torch.optim.Adam([
    #     {'params': backbone_params, 'lr': lr}, 
    #     {'params': other_params, 'lr': lr*10}
    # ], lr=lr, weight_decay=weight_decay)

    return train_dataset, val_dataset, model, optimizer


def train_main():
    val_ind = int(sys.argv[1])
    data_root = sys.argv[2]
    lr = float(sys.argv[3])
    batch_size = int(sys.argv[4])
    max_epochs = 200
    ddp_setup()
    train_dataset, val_dataset, model, optimizer = load_train_objs(val_ind=val_ind, data_root=data_root, lr=lr)

    dataloaders = {
        'train':
            DataLoader(train_dataset, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False)),
        'val':
            DataLoader(val_dataset, num_workers=4, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(val_dataset, shuffle=False, drop_last=False)),
    }
    trainer = Trainer(model, dataloaders, optimizer, save_root='./outputs', save_every=100, accum_iter=1)
    trainer.train(max_epochs=max_epochs)
    dist.destroy_process_group()


if __name__ == '__main__':
    train_main() 


"""

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29898 \
    ST_prediction_exps.py 0 ./data 1e-4 64

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29898 \
    ST_prediction_exps.py 1 /tmp/zhongz2/data 1e-4 32
"""

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

