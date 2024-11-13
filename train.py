
import sys,os
import numpy as np
import pandas as pd
import pickle
import random

import torch
torch.set_printoptions(sci_mode=False)
# torch.multiprocessing.set_sharing_strategy('file_system')
torch.multiprocessing.set_sharing_strategy('file_descriptor')
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from utils import get_args
from model import STModel
from dataset import PatchDataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)


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
            accum_iter=1
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
        self.model = DDP(self.model, device_ids=[self.gpu_id])

        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = nn.HuberLoss()
        
        if self.gpu_id == 0:
            self.losses = []

    def _run_epoch(self, epoch):

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        self.dataloader.sampler.set_epoch(epoch)

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
            log_df.to_csv(os.path.join(self.save_root, f'train_loss.csv'), float_format='%.3f', index=False)

        dist.barrier()

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, os.path.join(self.save_root, f'snapshot_{epoch}.pt'))

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0:
                self._save_snapshot(epoch)



def train_main(args):

    dist.init_process_group(backend="nccl")

    cache_root = os.path.join(args.data_root, 'exp_smooth{}'.format(args.use_smooth))
    save_root = f'{cache_root}/results/gpus{args.num_gpus}/backbone{args.backbone}_fixed{args.fixed_backbone}/lr{args.lr}_b{args.batch_size}_e{args.max_epochs}_accum{args.accum_iter}_v0_smooth{args.use_smooth}_stain{args.use_stain}'
    os.makedirs(save_root, exist_ok=True)

    with open(os.path.join(save_root, 'args_rank{}.pkl'.format(int(os.environ["LOCAL_RANK"]))), 'wb') as fp:
        pickle.dump({'args': args}, fp)
    if args.use_stain == 'True':
        with open(os.path.join(save_root, '.USE_STAIN'), 'w') as fp:
            pass

    if os.path.exists(os.path.join(save_root, 'snapshot_{}.pt'.format(args.max_epochs - 1))):
        print('done')
        sys.exit(-1)

    with open(os.path.join(cache_root, 'gene_infos.pkl'), 'rb') as fp:
        gene_names = pickle.load(fp)['gene_names']
    
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
        DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, pin_memory=True, shuffle=False, 
                sampler=DistributedSampler(train_dataset, shuffle=True, drop_last=False))

    ### model related
    model = STModel(backbone=args.backbone, num_outputs=len(gene_names))
    if args.fixed_backbone == "True":
        for param in model.backbone_model.parameters():
            param.requires_grad = False

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    backbone_params = [param for name, param in model.named_parameters() if 'backbone_model' in str(name)]
    other_params = [param for name, param in model.named_parameters() if 'backbone_model' not in str(name)]
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr}, 
        {'params': other_params, 'lr': args.lr*10}
    ], lr=args.lr, weight_decay=args.weight_decay)

    trainer = Trainer(model, train_dataloader, optimizer, save_root=save_root, accum_iter=args.accum_iter)
    trainer.train(max_epochs=args.max_epochs)
    dist.destroy_process_group()



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
