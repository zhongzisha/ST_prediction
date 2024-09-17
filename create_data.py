
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
from sklearn.metrics import r2_score, pairwise_distances
import idr_torch
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


def test_vst_vis(): # fine
    sc.logging.print_versions()
    sc.set_figure_params(facecolor="white", figsize=(8, 8))
    sc.settings.verbosity = 3

    adata = sc.datasets.visium_sge(sample_id="CytAssist_11mm_FFPE_Human_Lung_Cancer")
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    sc.pl.spatial(adata, color="log1p_n_genes_by_counts", cmap="hsv", save=True)


def get_gene_names():

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx') 
    human_slide_ids = {
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_CytAssist_FFPE_Protein_Expression_Human_Tonsil_2.1.0',
        '10x_CytAssist_Fresh_Frozen_Human_Breast_Cancer_2.0.1',
        # '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0',
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
    }
    df = df[df['slide_id'].isin(human_slide_ids)].reset_index(drop=True)

    # get common genes
    gene_names = {}
    invalid_svs_prefix = []
    for rowid, row in df.iterrows():
        svs_prefix = row['slide_id']
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        print(svs_prefix)
        if 'index' in existing_columns: 
            invalid_svs_prefix.append(svs_prefix)
        gene_names[svs_prefix] = [v for v in existing_columns if '__' != v[:2]]
    gene_names = list(gene_names.values())
    common_gene_names = sorted(list(set(gene_names[0]).intersection(*gene_names[1:])))
    all_gene_names = sorted(list(set(gene_names[0]).union(*gene_names[1:])))

    secreted_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', low_memory=False)
    secreted_gene_names = sorted(list(set([v.upper() for v in secreted_df['Gene'].values.tolist()])))

    print(len(common_gene_names), common_gene_names[:10])
    print(len(all_gene_names), all_gene_names[:10])
    print(len(secreted_gene_names), secreted_gene_names[:10])

    with open('gene_names.pkl', 'wb') as fp:
        pickle.dump({
            'common_gene_names': common_gene_names, 
            'all_gene_names': all_gene_names, 
            'secreted_gene_names': secreted_gene_names
        }, fp)


# version = 'v0' # just RGB patch, normalize with imagenet-mean/std
def create_data():

    version = sys.argv[1]

    version = 'v5'
    spot_scale = 1.3
    # data_root = os.path.join('/data/zhongz2/temp_ST_prediction', f'data_{version}')
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_images_{spot_scale}')
    os.makedirs(data_root, exist_ok=True)

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx')
    svs_dir = os.path.join(root, 'svs')
    patches_dir = os.path.join(root, 'patches')
    gene_vst_dir = os.path.join(root, 'gene_vst')

    human_slide_ids = {
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
    }
    df = df[df['slide_id'].isin(human_slide_ids)].reset_index(drop=True)

    with open('gene_names.pkl', 'rb') as fp:
        all_gene_names = pickle.load(fp)['all_gene_names']
        gene_map_dict = {v: str(i) for i, v in enumerate(all_gene_names)}
 
    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    for rowid, row in sub_df.iterrows():
        svs_prefix = row['slide_id']
        save_filename = '{}/{}.tar.gz'.format(data_root, svs_prefix)
        if os.path.exists(save_filename):
            continue
        svs_filename = os.path.join(root, 'svs', svs_prefix+'.svs')
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

        query_columns = ['__spot_X', '__spot_Y']
        xy_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        query_columns = [col for col in existing_columns if '__' != col[:2]]
        if False:
            vst_df = pd.read_parquet(vst_filename_db, columns=query_columns)
            vst_df = vst_df.clip(lower=-8, upper=8, axis=1)
            vst_df = vst_df.rename(columns=gene_map_dict)

        spot_size = row['spot_size']
        patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size
        slide = openslide.open_slide(svs_filename)

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        # for (_, row1), (_, row2) in zip(xy_df.iterrows(), vst_df.iterrows()):
        for _, row1 in xy_df.iterrows():
            x, y = int(row1['__spot_X'])-st_patch_size//2, int(row1['__spot_Y'])-st_patch_size//2  # left, top
            patch = slide.read_region(location=(x,y), level=0, size=(st_patch_size, st_patch_size)).convert('RGB')
            patch_filename = os.path.join(svs_prefix, f'x{x}_y{y}.jpg')
            # patch.save(save_filename)
            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=patch_filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)

            if False:
                labels_dict = {k:np.nan for k in list(gene_map_dict.values())}
                labels_dict.update(row2.to_dict())
                label = ','.join(['{:.3f}'.format(v) if v is not np.nan else 'nan' for k,v in labels_dict.items()])
                # with open(save_filename.replace('.jpg', '.txt'), 'w') as fp:
                #     fp.write(label)
                txt_buffer = io.StringIO(label)
                btxt_buffer = io.BytesIO(txt_buffer.read().encode())
                txt_filename = os.path.join(svs_prefix, f'x{x}_y{y}.txt')
                info = tarfile.TarInfo(name=txt_filename)
                info.size = btxt_buffer.getbuffer().nbytes
                info.mtime = time.time()
                btxt_buffer.seek(0)
                tar_fp.addfile(info, btxt_buffer)

        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())


def generate_train_val_data():

    # given a list of selected genes, generate the label files for each spot
    selected_gene_names = ['A2M', 'IFNG']
    selected_gene_names = []
    selected_gene_names = ['IFNG']; version = 'IFNG'
    selected_svs_prefixes = {
        'train': [
            # '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
            '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
            '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
            '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
            '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
            '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
            '10x_CytAssist_FFPE_Protein_Expression_Human_Tonsil_2.1.0',
            '10x_CytAssist_Fresh_Frozen_Human_Breast_Cancer_2.0.1',
            # '10x_Targeted_Visium_Human_BreastCancer_Immunology_1.2.0',
            '10x_V1_Breast_Cancer_Block_A_Section_1_1.1.0',
            '10x_V1_Breast_Cancer_Block_A_Section_2_1.1.0',
            '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
            '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
            '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
            '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
            '10x_Visium_Human_Breast_Cancer_1.3.0',
            'ST1K4M_Human_Breast_10X_06092021_Visium',
            'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep1',
            'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep1',
            'ST1K4M_Human_Prostate_10X_06092021_Visium_cancer',
            'ST1K4M_Human_Prostate_10X_06092021_Visium_normal',
            'ST1K4M_Human_Prostate_10X_07122022_Visium'
        ], 
        'val': [
            '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
            'ST1K4M_Human_Colon_10X_10052023_Visium_control_rep2',
            'ST1K4M_Human_Colon_10X_10052023_Visium_post_xenium_rep2',
        ]
    }

    spot_scale = 1.3
    use_gene_smooth = True
    data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_{version}_{spot_scale}_{use_gene_smooth}')
    os.makedirs(data_root, exist_ok=True)

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx')
    svs_dir = os.path.join(root, 'svs')
    patches_dir = os.path.join(root, 'patches')
    gene_vst_dir = os.path.join(root, 'gene_vst')

    with open('gene_names.pkl', 'rb') as fp:
        all_gene_names = pickle.load(fp)['all_gene_names']
    if len(selected_gene_names) == 0:
        selected_gene_names = sorted(all_gene_names)
    gene_map_dict = {v: str(i) for i, v in enumerate(selected_gene_names)}

    use_gene_tar = False

    data = {}
    for subset, svs_prefixes in selected_svs_prefixes.items():

        subset_save_root = os.path.join(data_root, subset)
        os.makedirs(subset_save_root, exist_ok=True)

        with open(os.path.join(subset_save_root, 'run.sh'), 'w') as fp:
            fp.write('#!/bin/bash\nfor f in `ls *.tar.gz`;do tar -xf $f; done')

        items = []
        for _, row in df.iterrows():
            svs_prefix = row['slide_id']
            if svs_prefix not in svs_prefixes:
                continue
            svs_filename = os.path.join(root, 'svs', svs_prefix+'.svs')
            save_filename = os.path.join(subset_save_root, svs_prefix+'.tar.gz')
            vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
            parquet_file = pq.ParquetFile(vst_filename_db)
            existing_columns = parquet_file.schema.names
            meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

            query_columns = ['__spot_X', '__spot_Y']
            xy_df = pd.read_parquet(vst_filename_db, columns=query_columns)
            if use_gene_smooth:
                distances = pairwise_distances(xy_df.values.astype(np.float32))
                sorted_inds = np.argsort(distances, axis=1)[:, 0:9] # 8 neighbors
            query_columns = [col for col in existing_columns if '__' != col[:2] and col in selected_gene_names]
            print('query_columns', query_columns)

            if len(query_columns) == 0:
                print('no selected genes in this case', svs_prefix)
                continue

            vst_df = pd.read_parquet(vst_filename_db, columns=query_columns)
            vst_df = vst_df.clip(lower=-8, upper=8, axis=1)
            vst_df = vst_df.rename(columns=gene_map_dict)

            spot_size = row['spot_size']
            patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
            st_patch_size = patch_size
            slide = openslide.open_slide(svs_filename)

            if use_gene_tar:
                fh = io.BytesIO()
                tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

            # for (_, row1), (_, row2) in zip(xy_df.iterrows(), vst_df.iterrows()):
            for rowind, (_, row1) in enumerate(xy_df.iterrows()):
                x, y = int(row1['__spot_X'])-st_patch_size//2, int(row1['__spot_Y'])-st_patch_size//2  # left, top

                if use_gene_smooth:
                    row2 = vst_df.iloc[sorted_inds[rowind]].mean()
                else:
                    row2 = vst_df.iloc[rowind]

                labels_dict = {k:np.nan for k in list(gene_map_dict.values())}
                labels_dict.update(row2.to_dict())

                if use_gene_tar:
                    label = ','.join(['{:.3f}'.format(v) if v is not np.nan else 'nan' for k,v in labels_dict.items()])
                    txt_buffer = io.StringIO(label)
                    btxt_buffer = io.BytesIO(txt_buffer.read().encode())
                    txt_filename = os.path.join(svs_prefix, f'x{x}_y{y}.txt')
                    info = tarfile.TarInfo(name=txt_filename)
                    info.size = btxt_buffer.getbuffer().nbytes
                    info.mtime = time.time()
                    btxt_buffer.seek(0)
                    tar_fp.addfile(info, btxt_buffer)

                if use_gene_tar:
                    items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), os.path.join(subset, svs_prefix, f'x{x}_y{y}.txt')))
                else:
                    items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), list(labels_dict.values())))

            if use_gene_tar:
                tar_fp.close()
                with open(save_filename, 'wb') as fp:
                    fp.write(fh.getvalue())

            print(svs_prefix)
        data[subset] = items

    with open(os.path.join(data_root, 'meta.pkl'), 'wb') as fp:
        pickle.dump({
            'selected_gene_names': selected_gene_names,
            'selected_svs_prefixes': selected_svs_prefixes,
            'data': data
        }, fp)






def get_all_df():

    import os
    import pyarrow.parquet as pq
    import pandas as pd

    root = '/data/zhongz2/ST_20240903'
    df = pd.read_excel(f'{root}/ST_20240907.xlsx')
    svs_dir = os.path.join(root, 'svs')
    patches_dir = os.path.join(root, 'patches')
    gene_vst_dir = os.path.join(root, 'gene_vst')

    human_slide_ids = {
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
    }
    df = df[df['slide_id'].isin(human_slide_ids)].reset_index(drop=True)


    all_vst_df = []
    for rowid, row in df.iterrows():
        svs_prefix = row['slide_id']

        svs_filename = os.path.join(root, 'svs', svs_prefix+'.svs')
        vst_filename_db = os.path.join(root, 'vst_dir_db', svs_prefix+'_original_VST.db')
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

        query_columns = ['__spot_X', '__spot_Y']
        xy_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        query_columns = [col for col in existing_columns if '__' != col[:2]]

        vst_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        vst_df = vst_df.clip(lower=-8, upper=8, axis=1)

        all_vst_df.append(vst_df)
        print(svs_prefix)
    
    all_vst_df = pd.concat(all_vst_df, axis=1)


def plot_hist():

    import os,pickle
    import pandas as pd
    import matplotlib.pyplot as plt 

    with open('meta.pkl', 'rb') as fp:
        meta = pickle.load(fp)
    
    trnY = [item[1] for item in meta['data']['train']]
    valY = [item[1] for item in meta['data']['val']]
    
    df = pd.DataFrame(trnY)
    df[0].hist()
    df = pd.DataFrame(valY)
    df[0].hist()
    plt.savefig('hist.png')
