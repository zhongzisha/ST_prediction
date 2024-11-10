
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
from sklearn.preprocessing import MinMaxScaler


def select_or_fill(df, columns):
    """Selects specified columns if they exist, otherwise fills with 0."""

    selected_columns = [col for col in columns if col in df.columns]
    not_selected_columns = [col for col in columns if col not in df.columns]
    result_df = df[selected_columns]
    if len(not_selected_columns) > 0:
        result_df.loc[:, not_selected_columns] = 0

    result_df = result_df[columns]
    return result_df

def get_subdir_path(filename):
    # 计算文件名的MD5哈希值，将其转换为十六进制字符串
    hash_value = hashlib.md5(filename.encode()).hexdigest()
    
    # 使用哈希值的前6位构造3级子目录
    subdir1 = hash_value[:2]
    subdir2 = hash_value[2:4]
    subdir3 = hash_value[4:6]
    
    # 构建完整的子目录路径
    subdir_path = os.path.join(subdir1, subdir2, subdir3)
    return subdir_path

def step1_examples():
    valid_genes = []
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC_generated'
    step1(excel_filename, final_save_root, valid_genes)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics_genenerated'
    step1(excel_filename, final_save_root, valid_genes)

def get_10xGenomics():
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics.xlsx'
    slide_ids = [
        'V1_Breast_Cancer_Block_A_Section_1',
        'V1_Breast_Cancer_Block_A_Section_2',
        'Visium_FFPE_Human_Breast_Cancer'
    ]
    items = []
    for slide_id in slide_ids:
        items.append(
            (
                slide_id, 
                f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics/{slide_id}/{slide_id}_filtered_feature_bc_matrix.h5',
                f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics/{slide_id}/{slide_id}_image.tif',
                f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics/{slide_id}/spatial/scalefactors_json.json',
                f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics/{slide_id}/spatial/tissue_positions_list.csv',
                slide_id
            )
        )
    df = pd.DataFrame(items, columns=['patient_id', 'counts_filename', 'TruePath', 'scalefactors_json', 'coord_filename', 'slide_id'])
    df.to_excel(excel_filename)

def step1(excel_filename, final_save_root, valid_genes=None):

    df = pd.read_excel(excel_filename, index_col=0)
    for col in ['patient_id', 'counts_filename', 'TruePath', 'scalefactors_json', 'coord_filename', 'slide_id']:
        assert col in df.columns

    patient_ids = np.unique(df['patient_id'].values)

    os.makedirs(final_save_root, exist_ok=True)

    all_coords_df = []
    all_counts_df = []

    slide_index_ids = []
    patient_ids = []

    for rowind, row in df.iterrows(): 

        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open(row['scalefactors_json'], 'r') as fp:
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

        # invalid_row_index = np.where((counts_df != 0).sum(axis=1) < 100)[0]
        # if len(invalid_row_index):# invalid spots 
        #     counts_df = counts_df.drop(index=counts_df.iloc[invalid_row_index].index)

        coord_df = coord_df.loc[[v for v in counts_df.index.values if v in coord_df.index.values]] # only keep those spots with gene counts
        coord_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

        slide_index_ids.extend([rowind for _ in range(len(coord_df))])
        patient_ids.extend([row['patient_id'] for _ in range(len(coord_df))])

        all_coords_df.append(coord_df)
        all_counts_df.append(counts_df)

        print(svs_filename)

    X_col_name = 'pxl_col_in_fullres'
    Y_col_name = 'pxl_row_in_fullres'
    
    coords_df = pd.concat(all_coords_df)
    coords_df['slide_index'] = slide_index_ids
    coords_df['patient_id'] = patient_ids
    coords_df = coords_df.drop(columns=['in_tissue'])
    del all_coords_df, slide_index_ids, patient_ids

    counts_df = pd.concat(all_counts_df)
    counts_df = counts_df.fillna(0)
    counts_df = counts_df.astype('int')
    del all_counts_df

    if valid_genes is None or len(valid_genes) == 0:
        counts_df1 = counts_df>0  # has expressed
        res=counts_df1.sum(axis=0)  # number of spots that has expressed
        res=res.sort_values(ascending=False)
        res1=res/len(counts_df) # 
        valid_genes = res1[res1>0.1].index.values
        with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'wb') as fp:
            pickle.dump({'valid_genes': valid_genes}, fp)
        del counts_df1, res

    # counts_df = counts_df[valid_genes]
    counts_df = select_or_fill(counts_df, valid_genes)
    counts_df.to_csv(os.path.join(final_save_root, 'all_counts.csv'), index=False)

    coords_df_new = coords_df.copy()
    for rowind, row in df.iterrows():

        coord_df = coords_df[coords_df['slide_index']==rowind]

        svs_prefix = row['slide_id'] 
        count_filename = row['counts_filename']
        svs_filename = row['TruePath']
        with open(row['scalefactors_json'], 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])
            spot_size = int(spot_size)
            patch_size = spot_size

        slide = openslide.open_slide(svs_filename)
        save_filename = os.path.join(final_save_root, row['slide_id']+'_patches.tar.gz')
        
        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
        filenames = []
        for _, row1 in coord_df.iterrows():
            xc, yc = row1[X_col_name], row1[Y_col_name]
            patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')

            filename = f'{svs_prefix}_x{xc}_y{yc}.jpg'
            filename = os.path.join(get_subdir_path(filename), filename)
            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)
            filenames.append(filename)
        
        coords_df_new.loc[coords_df['slide_index']==rowind, 'patch_filename'] = filenames
        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())
        
        print(svs_prefix)

    coords_df_new.to_csv(os.path.join(final_save_root, 'all_coords.csv'), index=False)



def create_train_val(val_patient_ids=[]):
    df = pd.read_excel('/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx', index_col=0)
    patient_ids = np.unique(df['patient_id'].values)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC'
    os.makedirs(final_save_root, exist_ok=True)

    all_coords = pd.read_csv(os.path.join(final_save_root, 'all_coords.csv'))
    all_counts = pd.read_csv(os.path.join(final_save_root, 'all_counts.csv'))
    all_counts = np.log10(1.0 + all_counts)

    patient_ids = all_coords['patient_id'].unique()
    val_patient_ids = [val_patient_id for val_patient_id in val_patient_ids if val_patient_id in patient_ids]
    print('final val patient ids', val_patient_ids)

    # column storage for quick access
    train_coords_df = all_coords[~all_coords['patient_id'].isin(val_patient_ids)]
    train_counts_df = all_counts[~all_coords['patient_id'].isin(val_patient_ids)]
    val_coords_df = all_coords[all_coords['patient_id'].isin(val_patient_ids)]
    val_counts_df = all_counts[all_coords['patient_id'].isin(val_patient_ids)]

    scaler = MinMaxScaler()
    train_counts = scaler.fit_transform(train_counts_df)
    val_counts = scaler.transform(val_counts_df)

    # train_coords_df.T.to_parquet(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'train_coords_df.parquet'), engine='fastparquet')

    save_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'])
    with open(os.path.join(save_root, 'train_val.pkl'), 'wb') as fp:
        pickle.dump({
            'train_counts': train_counts,
            'val_counts': val_counts,
            'train_coords_df': train_coords_df,
            'val_coords_df': val_coords_df 
        }, fp)

def create_train_test():
    # train on TNBC
    # test on 10x

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC_generated'
    train_coords_df = pd.read_csv(os.path.join(final_save_root, 'all_coords.csv'))
    train_counts_df = pd.read_csv(os.path.join(final_save_root, 'all_counts.csv'))
    train_counts_df = np.log10(1.0 + train_counts_df)

    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated'
    val_coords_df = pd.read_csv(os.path.join(final_save_root, 'all_coords.csv'))
    val_counts_df = pd.read_csv(os.path.join(final_save_root, 'all_counts.csv'))
    val_counts_df = np.log10(1.0 + val_counts_df)

    scaler = MinMaxScaler()
    train_counts = scaler.fit_transform(train_counts_df)
    val_counts = scaler.transform(val_counts_df)

    # train_coords_df.T.to_parquet(os.path.join('/lscratch', os.environ['SLURM_JOB_ID'], 'train_coords_df.parquet'), engine='fastparquet')

    save_root = os.path.join('/lscratch', os.environ['SLURM_JOB_ID'])
    with open(os.path.join(save_root, 'train_val.pkl'), 'wb') as fp:
        pickle.dump({
            'train_counts': train_counts,
            'val_counts': val_counts,
            'train_coords_df': train_coords_df,
            'val_coords_df': val_coords_df 
        }, fp)


if __name__ == '__main__':

    if sys.argv[1] != 'None':
        val_patient_ids = [int(v) for v in sys.argv[1].split(',')]
        print(val_patient_ids)
        create_train_val(val_patient_ids)
    else:
        create_train_test()



