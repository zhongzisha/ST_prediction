

import sys,os,glob,shutil,pickle,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
import scanpy
import io
import time
import tarfile
import torch
import hashlib
# import idr_torch
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True

def step1():
        
    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    cache_data = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'TNBC')
    os.makedirs(cache_data, exist_ok=True)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    for rowind, row in df.iterrows(): 

        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(row['slide_id']), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])

        coord_df = pd.read_csv(row['coord_filename'], header=None, index_col=0, low_memory=False)
        barcode_col_name = 0
        Y_col_name = 4
        X_col_name = 5
        in_tissue_col_name = 1

        coord_df = coord_df[coord_df[in_tissue_col_name]==1]

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
        coord_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

        coord_df.to_csv(os.path.join(cache_data, row['slide_id']+'_coords.csv'))
        counts_df.to_csv(os.path.join(cache_data, row['slide_id']+'_gene_counts.csv'))

        coord_df.to_parquet(os.path.join(cache_data, row['slide_id']+'_coords.parquet', engine='fastparquet'))
        counts_df.to_parquet(os.path.join(cache_data, row['slide_id']+'_gene_counts.parquet', engine='fastparquet'))



def select_or_fill(df, columns):
    """Selects specified columns if they exist, otherwise fills with 0."""

    selected_columns = [col for col in columns if col in df.columns]
    not_selected_columns = [col for col in columns if col not in df.columns]
    result_df = df[selected_columns]
    if len(not_selected_columns) > 0:
        result_df.loc[:, not_selected_columns] = 0

    result_df = result_df[columns]
    return result_df

def step2():

    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    cache_data = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'TNBC')
    os.makedirs(cache_data, exist_ok=True)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    all_coords_df = []
    all_counts_df = []
    bins = np.arange(0, 1000, 10).tolist()+[np.inf]

    valid_genes = []
    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']
    
    rowinds = []
    patient_ids = []
    for rowind, row in df.iterrows(): 

        svs_prefix = row['slide_id']
        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(row['slide_id']), 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])

        coord_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_coords.csv'), index_col=0)
        rowinds.extend([rowind for _ in range(len(coord_df))])
        patient_ids.extend([row['patient_id'] for _ in range(len(coord_df))])

        counts_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_gene_counts.csv'), index_col=0)

        # slide = openslide.open_slide(row['TruePath'])
        # print(svs_prefix, spot_size, fiducial_diameter_fullres)

        all_coords_df.append(coord_df)

        # counts_df1 = counts_df.fillna(0)
        # counts_df1 = counts_df1.astype('int')
        # counts_df2 = counts_df1.apply(lambda col: pd.cut(col, bins=bins, labels=False,include_lowest=True))

        counts_df = select_or_fill(counts_df, valid_genes)

        all_counts_df.append(counts_df)

    counts_df = pd.concat(all_counts_df)
    counts_df = counts_df.fillna(0)
    counts_df = counts_df.astype('int')

    # counts_df1 = counts_df>0  # has expressed
    # res=counts_df1.sum(axis=0)  # number of spots that has expressed
    # res=res.sort_values(ascending=False)
    # res1=res/len(counts_df) # 
    # valid_genes = res1[res1>0.1].index.values

    coords_df = pd.concat(all_coords_df)
    coords_df['slide_index'] = rowinds
    coords_df['patient_id'] = patient_ids
    coords_df.to_parquet(os.path.join(cache_data,'all_coords_11546.parquet'), engine='fastparquet')

    counts_df.to_csv(os.path.join(cache_data,'all_counts_11546.csv'))
    coords_df.to_csv(os.path.join(cache_data,'all_coords_11546.csv'))

    counts_df1 = np.log10(1.0+counts_df)
    counts_df1.to_csv(os.path.join(cache_data,'all_counts_11546_log10.csv'),float_format='%g')

def stop3():


    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    cache_data = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'TNBC')
    os.makedirs(cache_data, exist_ok=True)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    # all_coords_df = []
    # all_counts_df = []
    # bins = np.arange(0, 1000, 10).tolist()+[np.inf]

    valid_genes = []
    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']

    for rowind, row in df.iterrows(): 
        svs_prefix = row['slide_id'] 
        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(row['slide_id']), 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])

        coord_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_coords.csv'), index_col=0)
        counts_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_gene_counts.csv'), index_col=0)

        slide = openslide.open_slide(row['TruePath'])
        # # print(svs_prefix, spot_size, fiducial_diameter_fullres)

        # all_coords_df.append(coord_df)

        # # counts_df1 = counts_df.fillna(0)
        # # counts_df1 = counts_df1.astype('int')
        # # counts_df2 = counts_df1.apply(lambda col: pd.cut(col, bins=bins, labels=False,include_lowest=True))

        counts_df = select_or_fill(counts_df, valid_genes)
        counts_df = np.log10(1.0 + counts_df)

        # all_counts_df.append(counts_df)

        X_col_name = 'pxl_col_in_fullres'
        Y_col_name = 'pxl_row_in_fullres'
        vis_filename = os.path.join(final_save_root, svs_prefix+'.jpg')
        if not os.path.exists(vis_filename):
            spot_size = int(spot_size)
            patch_size = spot_size
            # plot spot figure
            W, H = slide.level_dimensions[0]
            img = slide.read_region((0, 0), 0, (W, H)).convert('RGB')
            draw = ImageDraw.Draw(img)
            img2 = Image.fromarray(255*np.ones((H, W, 3), dtype=np.uint8))
            draw2 = ImageDraw.Draw(img2)
            circle_radius = int(spot_size * 0.5)
            # colors = np.concatenate([colors, 128*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
            for ind, row1 in coord_df.iterrows():
                text = '{}x{}'.format(row1['array_col'], row1['array_row'])
                x, y = row1[X_col_name], row1[Y_col_name]
                xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
                draw.ellipse(xy, outline=(255, 128, 0), width=8)
                x -= patch_size // 2
                y -= patch_size // 2
                xy = [x, y, x+patch_size, y+patch_size]
                draw2.rectangle(xy, fill=(144, 238, 144))
                draw.text((x, y),text,(255,255,255),font=font)
            img3 = Image.blend(img, img2, alpha=0.4)
            img3.save(vis_filename)

        save_filename = os.path.join(final_save_root, svs_prefix+'.tar.gz')
        if not os.path.exists(save_filename): 

            fh = io.BytesIO()
            tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

            for _, row1 in coord_df.iterrows():
                xc, yc = row1[X_col_name], row1[Y_col_name]
                patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')

                patch_filename = os.path.join(svs_prefix, f'x{xc}_y{yc}.jpg')
                im_buffer = io.BytesIO()
                patch.save(im_buffer, format='JPEG')
                info = tarfile.TarInfo(name=patch_filename)
                info.size = im_buffer.getbuffer().nbytes
                info.mtime = time.time()
                im_buffer.seek(0)
                tar_fp.addfile(info, im_buffer)
            tar_fp.close()
            with open(save_filename, 'wb') as fp:
                fp.write(fh.getvalue())

        print(svs_prefix)


    # counts_df = pd.concat(all_counts_df)
    # counts_df = counts_df.fillna(0)
    # counts_df = counts_df.astype('int')

    # counts_df1 = counts_df>0  # has expressed
    # res=counts_df1.sum(axis=0)  # number of spots that has expressed
    # res=res.sort_values(ascending=False)
    # res1=res/len(counts_df) # 
    # valid_genes = res1[res1>0.1].index.values


def get_subdir_path(file_name):
    # 计算文件名的MD5哈希值，将其转换为十六进制字符串
    hash_value = hashlib.md5(file_name.encode()).hexdigest()
    
    # 使用哈希值的前6位构造3级子目录
    subdir1 = hash_value[:2]
    subdir2 = hash_value[2:4]
    subdir3 = hash_value[4:6]
    
    # 构建完整的子目录路径
    subdir_path = os.path.join(subdir1, subdir2, subdir3)
    return subdir_path

def stop4():


    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    cache_data = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'TNBC')
    os.makedirs(cache_data, exist_ok=True)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']

    base_dir = os.path.join(cache_data, 'torch_data')
    os.makedirs(base_dir, exist_ok=True)

    all_items = []
    for rowind, row in df.iterrows(): 
        svs_prefix = row['slide_id'] 
        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        patient_id = row['patient_id']
        section_id = row['section_id']
        with open('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(row['slide_id']), 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])
            spot_size = int(spot_size)
            patch_size = spot_size
        coord_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_coords.csv'), index_col=0)
        counts_df = pd.read_csv(os.path.join(final_save_root, row['slide_id']+'_gene_counts.csv'), index_col=0)

        slide = openslide.open_slide(row['TruePath'])

        counts_df = select_or_fill(counts_df, valid_genes)
        counts_df = np.log10(1.0 + counts_df)

        X_col_name = 'pxl_col_in_fullres'
        Y_col_name = 'pxl_row_in_fullres'
        
        for (_, row1), (_, row2) in zip(coord_df.iterrows(), counts_df.iterrows()):
            xc, yc = row1[X_col_name], row1[Y_col_name]
            patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')
            filename = f'{svs_prefix}_x{xc}_y{yc}.pkl'

            subdir_path = get_subdir_path(filename)
            os.makedirs(os.path.join(base_dir, subdir_path), exist_ok=True)
            torch.save({'patch': patch, 'label': row2.values}, os.path.join(base_dir, subdir_path, filename))

            all_items.append((patient_id, section_id, os.path.join(subdir_path, filename)))

    print(svs_prefix)


def prepare_TNBC():
         
    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    for rowind, row in df.iterrows(): 

        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open('/data/Jiang_Lab/Data/Zisha_Zhong/hk_TNBC_ST/TNBC_data/{}/spatial/scalefactors_json.json'.format(row['slide_id']), 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])

        coord_df = pd.read_csv(row['coord_filename'], header=None, index_col=0, low_memory=False)
        barcode_col_name = 0
        Y_col_name = 4
        X_col_name = 5
        in_tissue_col_name = 1

        coord_df = coord_df[coord_df[in_tissue_col_name]==1]

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
        coord_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

        coord_df.to_csv(os.path.join(final_save_root, row['slide_id']+'_coords.csv'))
        counts_df.to_csv(os.path.join(final_save_root, row['slide_id']+'_gene_counts.csv'))


def create_train_val(patient_id=0):


    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    cache_data = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'TNBC')
    os.makedirs(cache_data, exist_ok=True)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']

    all_coords_df = pd.read_csv(os.path.join(final_save_root, 'all_coords_11546.csv'), index_col=0)
    all_counts_df = pd.read_csv(os.path.join(final_save_root, 'all_counts_11546.csv'), index_col=0)







