
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
# import idr_torch
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
GLOBAL_MEAN = [0.79229504, 0.6218904 , 0.7667101]  #FRCE
GLOBAL_STD = [0.13924763, 0.15537892, 0.1152281]

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


def prepare_data(): # 10x on FRCE

    import os,h5py,glob,time,pickle
    import numpy as np
    import openslide
    import base64
    import pandas as pd
    import json
    import scanpy
    import matplotlib.pyplot as plt
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 12660162500
    from PIL import Image, ImageFile, ImageDraw
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    root = '/scratch/cluster_scratch/zhongz2/debug/data/10x'
    dirs = glob.glob(root+'/*')

    valid_svs_prefixes = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0',
        # '10x_CytAssist_FFPE_Human_Colon_Post_Xenium_Rep1_2.0.1',
        # '10x_CytAssist_FFPE_Human_Colon_Post_Xenium_Rep2_2.0.1',
        '10x_CytAssist_FFPE_Human_Colon_Rep1_2.1.0',
        # '10x_CytAssist_FFPE_Human_Colon_Rep2_2.1.0',
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_Visium_FFPE_Human_Breast_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Normal_Prostate_1.3.0',
        '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Cancer_1.3.0'
    ]

    items = []
    for d in dirs:
        if '_HD_' in d:
            continue

        svs_prefix = os.path.basename(d)
        if svs_prefix not in valid_svs_prefixes:
            continue

        name = '_'.join(svs_prefix.split('_')[1:-1])

        count_filename = os.path.join(d, name+'_filtered_feature_bc_matrix.h5')
        new_gene_count_filename = os.path.join(d, name+'_gene_count.csv')
        vst_filename = os.path.join(d, name+'_gene_vst.tsv')
        count_db_filename = os.path.join(d, name+'_gene_count.db')
        vst_db_filename = os.path.join(d, name+'_gene_vst.db')
        spot_vis_filename = os.path.join(d, name+'_spot_vis.jpg')
        with open('{}/spatial/scalefactors_json.json'.format(d), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])

        svs_filename = os.path.join(d, name + '_tissue_image.tif')
        if not os.path.exists(svs_filename):
            svs_filename = os.path.join(d, name + '_image.tif')
            if not os.path.exists(svs_filename):
                svs_filename = os.path.join(d, name + '_image.jpg')
                if not os.path.exists(svs_filename):
                    raise ValueError("error")
        coord_filename = os.path.join(d, 'spatial/tissue_positions.csv')

        if not os.path.exists(coord_filename):
            coord_filename = os.path.join(d, 'spatial/tissue_positions_list.csv')
            if not os.path.exists(coord_filename):
                raise ValueError("coord not existed")
            else:
                coord_df = pd.read_csv(coord_filename, header=None, index_col=0, low_memory=False)
                barcode_col_name = 0
                Y_col_name = 4
                X_col_name = 5
                in_tissue_col_name = 1
        else:
            coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False)
            barcode_col_name = 'barcode'
            Y_col_name = 'pxl_row_in_fullres'
            X_col_name = 'pxl_col_in_fullres'
            in_tissue_col_name = 'in_tissue'

        items.append([svs_prefix, svs_filename, coord_filename, count_filename, new_gene_count_filename, vst_filename, vst_db_filename, count_db_filename, spot_size])
        if os.path.exists(spot_vis_filename):
            continue

        coord_df = coord_df[coord_df[in_tissue_col_name]==1]

        if not os.path.exists(count_filename):
            raise ValueError("count not existed")
        else:
            counts_df = scanpy.read_10x_h5(count_filename).to_df().T
            counts_df = counts_df.astype(np.float32)
            counts_df = counts_df.fillna(0)
            counts_df = counts_df.groupby(counts_df.index).sum().T
            counts_df = counts_df.loc[[v for v in coord_df.index.values if v in counts_df.index.values]]
            counts_df.columns = [n.upper() for n in counts_df.columns]

            invalid_col_index = np.where(counts_df.sum(axis=0) == 0)[0]
            if len(invalid_col_index):# invalid genes 
                counts_df = counts_df.drop(columns=counts_df.columns[invalid_col_index])  

            invalid_row_index = np.where((counts_df != 0).sum(axis=1) < 100)[0]
            if len(invalid_row_index):# invalid spots 
                counts_df = counts_df.drop(index=counts_df.iloc[invalid_row_index].index)

            coord_df = coord_df.loc[[v for v in counts_df.index.values if v in coord_df.index.values]] # only keep those spots with gene counts
        
        if True:
            counts_df.T.to_csv(new_gene_count_filename, sep='\t')
            counts_df['__spot_X'] = coord_df[X_col_name].values.tolist()
            counts_df['__spot_Y'] = coord_df[Y_col_name].values.tolist()
            counts_df.index.name = '__barcode'
            counts_df.to_parquet(count_db_filename, engine='fastparquet')
            del counts_df

            if False:
                parquet_file = pq.ParquetFile(count_db_filename)
                existing_columns = parquet_file.schema.names
                meta_columns = ['__barcode', '__spot_X', '__spot_Y']

                query_columns = ['__spot_X', '__spot_Y']
                xy_df = pd.read_parquet(count_db_filename, columns=query_columns)
                query_columns = [col for col in existing_columns if '__' != col[:2]]


            joblines = [
                '#!/bin/bash\n\n',
                'Rscript --vanilla compute_vst.R "{}" "{}"\n\n\n'.format(new_gene_count_filename, vst_filename)
            ]

            temp_job_filename = f'./job_compute_vst_{name}.sh'
            with open(temp_job_filename, 'w') as fp:
                fp.writelines(joblines)
            time.sleep(0.5)
            os.system(f'bash "{temp_job_filename}"')      

        if True:
            vst = pd.read_csv(vst_filename, sep='\t', index_col=0, low_memory=False)
            vst = vst.astype('float32')
            vst = vst.subtract(vst.mean(axis=1), axis=0)
            vst = vst.T
            vst.columns = [n.upper() for n in vst.columns]
            vst['__spot_X'] = coord_df[X_col_name].values.tolist()
            vst['__spot_Y'] = coord_df[Y_col_name].values.tolist()
            vst.index.name = '__barcode'
            vst.to_parquet(vst_db_filename, engine='fastparquet')
            del vst

        if False:
            spot_scale = 1.3
            patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
            st_patch_size = patch_size

            slide = openslide.open_slide(svs_filename)
            # plot spot figure
            W, H = slide.level_dimensions[0]
            img = slide.read_region((0, 0), 0, (W, H)).convert('RGB')
            draw = ImageDraw.Draw(img)
            img2 = Image.fromarray(255*np.ones((H, W, 3), dtype=np.uint8))
            draw2 = ImageDraw.Draw(img2)
            circle_radius = int(spot_size * 0.5)
            # colors = np.concatenate([colors, 128*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
            for ind, (x,y) in enumerate(zip(coord_df[X_col_name].values.tolist(), coord_df[Y_col_name].values.tolist())):
                xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
                draw.ellipse(xy, outline=(255, 128, 0), width=8)
                x -= patch_size // 2
                y -= patch_size // 2
                xy = [x, y, x+patch_size, y+patch_size]
                draw2.rectangle(xy, fill=(144, 238, 144))
            img3 = Image.blend(img, img2, alpha=0.4)
            img3.save(spot_vis_filename)

            slide.close()
            del img, img2, img3, draw, draw2 
        
    df = pd.DataFrame(items, columns=['slide_id', 'svs_filename', 'coord_filename', 'count_filename', 'new_gene_count_filename', 'vst_filename', 'vst_db_filename', 'count_db_filename', 'spot_size'])
    df.to_csv(f'{root}/all.csv')









def get_gene_names_FRCE():

    root = '/scratch/cluster_scratch/zhongz2/debug/data/10x'
    df = pd.read_csv(f'{root}/all.csv', index_col=0, low_memory=False)
    # get common genes
    gene_names = {}
    invalid_svs_prefix = []
    for rowid, row in df.iterrows():
        svs_prefix = row['slide_id']
        parquet_file = pq.ParquetFile(row['vst_db_filename'])
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

    print(len(common_gene_names), common_gene_names[:20])
    print(len(all_gene_names), all_gene_names[:20])
    print(len(secreted_gene_names), secreted_gene_names[:20])

    with open('gene_names.pkl', 'wb') as fp:
        pickle.dump({
            'common_gene_names': common_gene_names, 
            'all_gene_names': all_gene_names, 
            'secreted_gene_names': secreted_gene_names
        }, fp)


# version = 'v0' # just RGB patch, normalize with imagenet-mean/std
def create_data_FRCE():

    spot_scale = 1.3

    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_images_{spot_scale}')
    else:
        data_root = os.path.join('/tmp/zhongz2/', f'data_images_{spot_scale}')
    os.makedirs(data_root, exist_ok=True)

    root = '/scratch/cluster_scratch/zhongz2/debug/data/10x'
    df = pd.read_csv(f'{root}/all.csv', index_col=0, low_memory=False)

    with open('gene_names.pkl', 'rb') as fp:
        all_gene_names = pickle.load(fp)['all_gene_names']
        gene_map_dict = {v: str(i) for i, v in enumerate(all_gene_names)}
 
    # indices = np.arange(len(df))
    # index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    # sub_df = df.iloc[index_splits[idr_torch.rank]]
    # sub_df = sub_df.reset_index(drop=True)
    sub_df = df

    global_mean = np.zeros((3, ), dtype=np.float32)
    global_std = np.zeros((3, ), dtype=np.float32)

    for rowid, row in sub_df.iterrows():
        svs_prefix = row['slide_id']
        save_filename = '{}/{}.tar.gz'.format(data_root, svs_prefix)
        if os.path.exists(save_filename):
            print('existed.')
            continue
        vst_filename_db = row['vst_db_filename']
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
        slide = openslide.open_slide(row['svs_filename'])

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        mean = np.zeros((3, ), dtype=np.float32)
        std = np.zeros((3, ), dtype=np.float32)
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

            patch = np.array(patch, dtype=np.float32) / 255
            mean += patch.mean((0, 1))
            std += patch.std((0, 1))

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

        mean /= len(xy_df)
        std /= len(xy_df)
        global_mean += mean
        global_std += std
    global_mean /= len(df)
    global_std /= len(df)

    print('global_mean', global_mean)
    print('global_std', global_std)
    with open(f'{root}/global_mean_std.pkl', 'wb') as fp:
        pickle.dump({'global_mean': global_mean, 'global_std': global_std}, fp)


def generate_train_val_data_FRCE():

    version = '_20240919_v1'

    # given a list of selected genes, generate the label files for each spot
    selected_gene_names = ['A2M', 'IFNG']
    selected_gene_names = []
    selected_gene_names = ['IFNG', 'FASN', 'A2M', 'SERPING1']
    selected_gene_names = sorted(selected_gene_names)

    all_svs_prefixes = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0', 
        '10x_CytAssist_FFPE_Human_Colon_Rep1_2.1.0', 
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_Visium_FFPE_Human_Breast_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Normal_Prostate_1.3.0',
        '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Cancer_1.3.0'
    ]
    val_prefixes = [
        '10x_Visium_FFPE_Human_Prostate_Cancer_1.3.0'
    ]

    spot_scale = 1.3
    use_gene_smooth = False

    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_genes_{spot_scale}_{use_gene_smooth}_{version}')
    else:
        data_root = os.path.join('/tmp/zhongz2/', f'data_genes_{spot_scale}_{use_gene_smooth}_{version}')
    hist_plot_save_dir = os.path.join(data_root, 'hists')
    os.makedirs(hist_plot_save_dir, exist_ok=True)

    root = '/scratch/cluster_scratch/zhongz2/debug/data/10x'
    df = pd.read_csv(f'{root}/all.csv', index_col=0, low_memory=False)

    with open('gene_names.pkl', 'rb') as fp:
        all_gene_names = pickle.load(fp)['all_gene_names']
    if len(selected_gene_names) == 0:
        selected_gene_names = sorted(all_gene_names)
    gene_map_dict = {v: str(i) for i, v in enumerate(selected_gene_names)}

    use_gene_tar = False
    bins = np.arange(-8, 8, 1)
    
    data = {'train': [], 'val': []}
    for _, row in df.iterrows():
        svs_prefix = row['slide_id']
        if svs_prefix not in all_svs_prefixes:
            continue
        svs_filename = row['svs_filename']
        save_filename = os.path.join(data_root, svs_prefix+'.tar.gz')
        vst_filename_db = row['vst_db_filename']
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y']

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

        for gene_name in query_columns: 
            vst_df[gene_name].hist(bins=bins)
            plt.savefig(f'{hist_plot_save_dir}/{svs_prefix}_{gene_name}_hist.jpg')
            plt.close()

        vst_df = vst_df.rename(columns=gene_map_dict)

        spot_size = row['spot_size']
        patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size
        slide = openslide.open_slide(svs_filename)

        if use_gene_tar:
            fh = io.BytesIO()
            tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        items = []
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
                items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), os.path.join(svs_prefix, f'x{x}_y{y}.txt')))
            else:
                items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), list(labels_dict.values())))

        if use_gene_tar:
            tar_fp.close()
            with open(save_filename, 'wb') as fp:
                fp.write(fh.getvalue())
        
        print(svs_prefix)
        if svs_prefix in val_prefixes:
            data['val'].extend(items)
        else:
            data['train'].extend(items)

    if use_gene_tar:
        with open(os.path.join(data_root, 'run.sh'), 'w') as fp:
            fp.write('#!/bin/bash\nfor f in `ls *.tar.gz`;do tar -xf $f; done')


    with open(os.path.join(data_root, 'meta.pkl'), 'wb') as fp:
        pickle.dump({
            'data': data,
            'selected_gene_names': selected_gene_names,
            'use_gene_smooth': use_gene_smooth,
            'val_prefixes': val_prefixes
        }, fp)











def prepare_data(): # He2020 on FRCE

    import os,h5py,glob,time,pickle
    import numpy as np
    import openslide
    import base64
    import pandas as pd
    import json
    import scanpy
    import matplotlib.pyplot as plt
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 12660162500
    from PIL import Image, ImageFile, ImageDraw
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
    df = pd.read_csv(f'{root}/metadata.csv', low_memory=False)

    items = []
    for _, row in df.iterrows(): 
        svs_prefix = os.path.basename(row['histology_image']).replace('.jpg', '')

        name = svs_prefix

        count_filename = os.path.join(root, row['count_matrix'])
        new_gene_count_filename = os.path.join(root, name+'_gene_count.csv')
        vst_filename = os.path.join(root, name+'_gene_vst.tsv')
        count_db_filename = os.path.join(root, name+'_gene_count.db')
        vst_db_filename = os.path.join(root, name+'_gene_vst.db')
        spot_vis_filename = os.path.join(root, name+'_spot_vis.jpg')
        # with open('{}/spatial/scalefactors_json.json'.format(d), 'r') as fp:
        #     spot_size = float(json.load(fp)['spot_diameter_fullres'])
        spot_size = 224

        svs_filename = os.path.join(root, row['histology_image'])
        coord_filename = os.path.join(root, row['spot_coordinates'])

        coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False)
        Y_col_name = 'Y'
        X_col_name = 'X' 

        items.append([svs_prefix, svs_filename, coord_filename, count_filename, new_gene_count_filename, vst_filename, vst_db_filename, count_db_filename, spot_size])
        if os.path.exists(spot_vis_filename):
            continue

        if not os.path.exists(count_filename):
            raise ValueError("count not existed")
        else:
            counts_df = pd.read_csv(count_filename, sep='\t', index_col=0, low_memory=False).T
            counts_df = counts_df.astype(np.float32)
            counts_df = counts_df.fillna(0)
            counts_df = counts_df.groupby(counts_df.index).sum().T
            counts_df = counts_df.loc[[v for v in coord_df.index.values if v in counts_df.index.values]]
            counts_df.columns = [n.upper() for n in counts_df.columns]
            valid_columns = [col for col in counts_df.columns if '__AMBIGUOUS' not in col]
            counts_df = counts_df[valid_columns] 

            invalid_col_index = np.where(counts_df.sum(axis=0) == 0)[0]
            if len(invalid_col_index):# invalid genes 
                counts_df = counts_df.drop(columns=counts_df.columns[invalid_col_index])  

            if True:
                invalid_row_index = np.where((counts_df != 0).sum(axis=1) < 100)[0]
                if len(invalid_row_index):# invalid spots 
                    counts_df = counts_df.drop(index=counts_df.iloc[invalid_row_index].index)

            coord_df = coord_df.loc[[v for v in counts_df.index.values if v in coord_df.index.values]] # only keep those spots with gene counts
        
        if True:
            counts_df.T.to_csv(new_gene_count_filename, sep='\t')
            counts_df['__spot_X'] = coord_df[X_col_name].values.tolist()
            counts_df['__spot_Y'] = coord_df[Y_col_name].values.tolist()
            counts_df.index.name = '__barcode'
            counts_df.to_parquet(count_db_filename, engine='fastparquet')
            del counts_df

            if False:
                parquet_file = pq.ParquetFile(count_db_filename)
                existing_columns = parquet_file.schema.names
                meta_columns = ['__barcode', '__spot_X', '__spot_Y']

                query_columns = ['__spot_X', '__spot_Y']
                xy_df = pd.read_parquet(count_db_filename, columns=query_columns)
                query_columns = [col for col in existing_columns if '__' != col[:2]]


            joblines = [
                '#!/bin/bash\n\n',
                'Rscript --vanilla compute_vst.R "{}" "{}"\n\n\n'.format(new_gene_count_filename, vst_filename)
            ]

            temp_job_filename = f'./job_compute_vst_{name}.sh'
            with open(temp_job_filename, 'w') as fp:
                fp.writelines(joblines)
            time.sleep(0.5)
            os.system(f'bash "{temp_job_filename}"')      

        if True:
            vst = pd.read_csv(vst_filename, sep='\t', index_col=0, low_memory=False)
            vst = vst.astype('float32')
            vst = vst.subtract(vst.mean(axis=1), axis=0)
            vst = vst.T
            vst.columns = [n.upper() for n in vst.columns]
            vst['__spot_X'] = coord_df[X_col_name].values.tolist()
            vst['__spot_Y'] = coord_df[Y_col_name].values.tolist()
            vst.index.name = '__barcode'
            vst.to_parquet(vst_db_filename, engine='fastparquet')
            del vst

        spot_scale = 1.3
        patch_size = 224  # int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size

        slide = openslide.open_slide(svs_filename)
        # plot spot figure
        W, H = slide.level_dimensions[0]
        img = slide.read_region((0, 0), 0, (W, H)).convert('RGB')
        draw = ImageDraw.Draw(img)
        img2 = Image.fromarray(255*np.ones((H, W, 3), dtype=np.uint8))
        draw2 = ImageDraw.Draw(img2)
        circle_radius = int(spot_size * 0.5)
        # colors = np.concatenate([colors, 128*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
        for ind, (x,y) in enumerate(zip(coord_df[X_col_name].values.tolist(), coord_df[Y_col_name].values.tolist())):
            xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
            draw.ellipse(xy, outline=(255, 128, 0), width=8)
            x -= patch_size // 2
            y -= patch_size // 2
            xy = [x, y, x+patch_size, y+patch_size]
            draw2.rectangle(xy, fill=(144, 238, 144))
        img3 = Image.blend(img, img2, alpha=0.4)
        img3.save(spot_vis_filename)

        slide.close()
        del img, img2, img3, draw, draw2 
    
    df = pd.DataFrame(items, columns=['slide_id', 'svs_filename', 'coord_filename', 'count_filename', 'new_gene_count_filename', 'vst_filename', 'vst_db_filename', 'count_db_filename', 'spot_size'])
    df.to_csv(f'{root}/all.csv')



# version = 'v0' # just RGB patch, normalize with imagenet-mean/std
def create_data_FRCE(): # He2020

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
    # import idr_torch
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 12660162500
    from PIL import Image, ImageFile, ImageDraw, ImageFilter
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    spot_scale = 1.3

    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_images_he2020_{spot_scale}')
    else:
        data_root = os.path.join('/tmp/zhongz2/', f'data_images_he2020_{spot_scale}')
    os.makedirs(data_root, exist_ok=True)

    root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
    df = pd.read_csv(f'{root}/all.csv', index_col=0, low_memory=False)
 
    # indices = np.arange(len(df))
    # index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    # sub_df = df.iloc[index_splits[idr_torch.rank]]
    # sub_df = sub_df.reset_index(drop=True)
    sub_df = df

    global_mean = np.zeros((3, ), dtype=np.float32)
    global_std = np.zeros((3, ), dtype=np.float32)

    for rowid, row in sub_df.iterrows():
        svs_prefix = row['slide_id']
        save_filename = '{}/{}.tar.gz'.format(data_root, svs_prefix)
        if os.path.exists(save_filename):
            print('existed.')
            continue
        vst_filename_db = row['vst_db_filename']
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y', '__upperleft_X', '__upperleft_Y']

        query_columns = ['__spot_X', '__spot_Y']
        xy_df = pd.read_parquet(vst_filename_db, columns=query_columns)
        query_columns = [col for col in existing_columns if '__' != col[:2]]

        spot_size = row['spot_size']
        patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size
        slide = openslide.open_slide(row['svs_filename'])

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        mean = np.zeros((3, ), dtype=np.float32)
        std = np.zeros((3, ), dtype=np.float32)
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

            patch = np.array(patch, dtype=np.float32) / 255
            mean += patch.mean((0, 1))
            std += patch.std((0, 1))

        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())

        mean /= len(xy_df)
        std /= len(xy_df)
        global_mean += mean
        global_std += std
    global_mean /= len(df)
    global_std /= len(df)

    print('global_mean', global_mean)
    print('global_std', global_std)
    with open(f'{root}/global_mean_std.pkl', 'wb') as fp:
        pickle.dump({'global_mean': global_mean, 'global_std': global_std}, fp)



def generate_train_val_data_FRCE():

    version = '_He2020_v1'

    # given a list of selected genes, generate the label files for each spot
    selected_gene_names = ['A2M', 'IFNG']
    selected_gene_names = []
    selected_gene_names = ['IFNG', 'FASN', 'A2M', 'SERPING1']
    selected_gene_names = ['ENSG00000111537', 'ENSG00000169710', 'ENSG00000175899', 'ENSG00000149131']
    selected_gene_names = ['ENSG00000169710']
    selected_gene_names = sorted(selected_gene_names)

    all_svs_prefixes = [
        '10x_CytAssist_11mm_FFPE_Human_Colorectal_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Glioblastoma_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Kidney_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Lung_Cancer_2.0.1',
        '10x_CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma_2.0.0', 
        '10x_CytAssist_FFPE_Human_Colon_Rep1_2.1.0', 
        '10x_CytAssist_FFPE_Human_Lung_Squamous_Cell_Carcinoma_2.0.0',
        '10x_Visium_FFPE_Human_Breast_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Cervical_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Intestinal_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Normal_Prostate_1.3.0',
        '10x_Visium_FFPE_Human_Ovarian_Cancer_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Acinar_Cell_Carcinoma_1.3.0',
        '10x_Visium_FFPE_Human_Prostate_Cancer_1.3.0'
    ]
    val_prefixes = [
        'BT23901', 'BT23903', 'BT23944', 'BT24044', 'BT24223'
    ]

    spot_scale = 1.3
    use_gene_smooth = False

    if os.environ['CLUSTER_NAME'] == 'Biowulf':
        data_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], f'data_genes_{spot_scale}_{use_gene_smooth}_{version}')
    else:
        data_root = os.path.join('/tmp/zhongz2/', f'data_genes_{spot_scale}_{use_gene_smooth}_{version}')
    hist_plot_save_dir = os.path.join(data_root, 'hists')
    os.makedirs(hist_plot_save_dir, exist_ok=True)

    root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
    df = pd.read_csv(f'{root}/all.csv', index_col=0, low_memory=False)
    patient_ids = [v.split('_')[1] for v in df['slide_id'].values]

    # with open('gene_names.pkl', 'rb') as fp:
    #     all_gene_names = pickle.load(fp)['all_gene_names']
    # if len(selected_gene_names) == 0:
    #     selected_gene_names = sorted(all_gene_names)
    gene_map_dict = {v: str(i) for i, v in enumerate(selected_gene_names)}

    use_gene_tar = False
    bins = np.arange(-8, 8, 1)
    
    data = {'train': [], 'val': []}
    for _, row in df.iterrows():
        svs_prefix = row['slide_id']
        patient_id = svs_prefix.split('_')[1]
        svs_filename = row['svs_filename']
        save_filename = os.path.join(data_root, svs_prefix+'.tar.gz')
        vst_filename_db = row['vst_db_filename']
        parquet_file = pq.ParquetFile(vst_filename_db)
        existing_columns = parquet_file.schema.names
        meta_columns = ['__barcode', '__spot_X', '__spot_Y']

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

        for gene_name in query_columns: 
            vst_df[gene_name].hist(bins=bins)
            plt.savefig(f'{hist_plot_save_dir}/{svs_prefix}_{gene_name}_hist.jpg')
            plt.close()

        vst_df = vst_df.rename(columns=gene_map_dict)

        spot_size = row['spot_size']
        patch_size = int(np.ceil(spot_scale * spot_size)) # expand some area (10% here)
        st_patch_size = patch_size
        slide = openslide.open_slide(svs_filename)

        if use_gene_tar:
            fh = io.BytesIO()
            tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        items = []
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
                items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), os.path.join(svs_prefix, f'x{x}_y{y}.txt')))
            else:
                items.append((os.path.join(svs_prefix, f'x{x}_y{y}.jpg'), list(labels_dict.values())))

        if use_gene_tar:
            tar_fp.close()
            with open(save_filename, 'wb') as fp:
                fp.write(fh.getvalue())
        
        print(svs_prefix)
        if patient_id in val_prefixes:
            data['val'].extend(items)
        else:
            data['train'].extend(items)

    if use_gene_tar:
        with open(os.path.join(data_root, 'run.sh'), 'w') as fp:
            fp.write('#!/bin/bash\nfor f in `ls *.tar.gz`;do tar -xf $f; done')


    with open(os.path.join(data_root, 'meta.pkl'), 'wb') as fp:
        pickle.dump({
            'data': data,
            'selected_gene_names': selected_gene_names,
            'use_gene_smooth': use_gene_smooth,
            'val_prefixes': val_prefixes
        }, fp)

