
import sys,os,glob,shutil,pickle,json,argparse
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
    use_gene_smooth = False

    valid_genes = []
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC_generated'
    step1(excel_filename, final_save_root, valid_genes)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated'
    step1(excel_filename, final_save_root, valid_genes)


def step1_examples_use_gene_smooth():
    use_gene_smooth = True

    valid_genes = []
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_smooth'
    step1(excel_filename, final_save_root, valid_genes, use_gene_smooth)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']

    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics.xlsx'
    final_save_root = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated_smooth'
    step1(excel_filename, final_save_root, valid_genes, use_gene_smooth)


def step1_examples_use_gene_smooth_use_stain_norm():
    use_gene_smooth = True
    use_stain_normalization = True
    ratio = 0.05

    valid_genes = []
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/TNBC.xlsx'
    final_save_root = f'/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_{ratio}_smooth_stain'
    step1(excel_filename, final_save_root, valid_genes, use_gene_smooth, use_stain_normalization)

    with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'rb') as fp:
        valid_genes = pickle.load(fp)['valid_genes']

    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xGenomics.xlsx'
    final_save_root = f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated_{ratio}_smooth_stain'
    step1(excel_filename, final_save_root, valid_genes, use_gene_smooth, use_stain_normalization)


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

def gene_neighbor_smoothing(counts_df, use_10x=False):
    spot_names = counts_df.index.values  # should replace the index
    new_counts_df = counts_df.copy()
 
    for s in spot_names:
        r, c = [int(v) for v in s.split('x')]
        if use_10x:
            ns = ['{}x{}'.format(rr, cc) for rr, cc in [
                (r+1, c+1), (r+1, c-1), (r, c-2), (r, c), (r, c+2), (r-1, c-1), (r-1, c+1)
            ]]
        else:
            ns = ['{}x{}'.format(rr, cc) for rr in [r-1, r, r+1] for cc in [c-1, c, c+1]]
        ns = [v for v in ns if v in spot_names]
        new_counts_df.loc[s] = counts_df.loc[ns].mean()
    return new_counts_df



def normalizeStaining(img, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images
    
    Example use:
        see test.py
        
    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    
    Reference: 
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1,3))

    # calculate optical density
    OD = -np.log((img.astype(np.float32)+1)/Io)
    
    # remove transparent pixels
    ODhat = OD[~np.any(OD<beta, axis=1)]
        
    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    
    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100-alpha)
    
    vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:,0], vMax[:,0])).T
    else:
        HE = np.array((vMax[:,0], vMin[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE,Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC,maxCRef)
    C2 = np.divide(C,tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm>255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H>255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E>255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    return Image.fromarray(Inorm)
    

def step1(excel_filename, final_save_root, valid_genes=None, use_gene_smooth=False, use_stain_normalization=False, ratio=0.1):

    df = pd.read_excel(excel_filename, index_col=0)
    for col in ['patient_id', 'counts_filename', 'TruePath', 'scalefactors_json', 'coord_filename', 'slide_id']:
        assert col in df.columns

    patient_ids = np.unique(df['patient_id'].values)

    os.makedirs(final_save_root, exist_ok=True)

    all_coords_df = []
    all_counts_df = []

    slide_index_ids = []
    patient_ids = []

    X_col_name = 'pxl_col_in_fullres'
    Y_col_name = 'pxl_row_in_fullres'
    font = ImageFont.load_default(32)

    for rowind, row in df.iterrows(): 

        with open(row['scalefactors_json'], 'r') as fp:
            spot_size = float(json.load(fp)['spot_diameter_fullres'])

        coord_df = pd.read_csv(row['coord_filename'], header=None, index_col=0, low_memory=False)
        coord_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

        vis_filename = os.path.join(final_save_root, row['slide_id']+'.jpg')
        if False:# not os.path.exists(vis_filename):
            slide = openslide.open_slide(row['TruePath'])
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
                text = '{}x{}'.format(row1['array_row'], row1['array_col'])
                x, y = row1[X_col_name], row1[Y_col_name]
                xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
                draw.ellipse(xy, outline=(255, 128, 0), width=8)
                x -= patch_size // 2
                y -= patch_size // 2
                xy = [x, y, x+patch_size, y+patch_size]
                if row1['in_tissue'] == 1:
                    draw2.rectangle(xy, fill=(144, 238, 144))
                else:
                    draw2.rectangle(xy, fill=(50, 50, 50))
                draw.text((x, y),text,(255,255,255),font=font)
            img3 = Image.blend(img, img2, alpha=0.4)
            img3.save(vis_filename)
            del img, draw, img2, draw2, img3
            slide.close()

        coord_df = coord_df[coord_df['in_tissue']==1]

        counts_df = scanpy.read_10x_h5(row['counts_filename']).to_df().T
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

        coord_df['barcode'] = coord_df.index.values
        xy_names = ['{}x{}'.format(row['array_row'], row['array_col']) for rowid, row in coord_df.iterrows()]
        coord_df.index = xy_names
        counts_df.index = xy_names

        if use_gene_smooth:
            counts_df = gene_neighbor_smoothing(counts_df, use_10x=True)

        slide_index_ids.extend([rowind for _ in range(len(coord_df))])
        patient_ids.extend([row['patient_id'] for _ in range(len(coord_df))])

        all_coords_df.append(coord_df)
        all_counts_df.append(counts_df)

        print(row)
    
    coords_df = pd.concat(all_coords_df)
    coords_df['slide_index'] = slide_index_ids
    coords_df['patient_id'] = patient_ids
    coords_df = coords_df.drop(columns=['in_tissue'])
    del all_coords_df, slide_index_ids, patient_ids

    counts_df = pd.concat(all_counts_df)
    counts_df = counts_df.fillna(0)
    # counts_df = counts_df.astype('int')
    del all_counts_df

    if valid_genes is None or len(valid_genes) == 0:
        counts_df1 = counts_df>0  # has expressed
        res=counts_df1.sum(axis=0)  # number of spots that has expressed
        res=res.sort_values(ascending=False)
        res1=res/len(counts_df) # 
        valid_genes = res1[res1>ratio].index.values
        with open(os.path.join(final_save_root, 'valid_genes.pkl'), 'wb') as fp:
            pickle.dump({'valid_genes': valid_genes}, fp)
        del counts_df1, res

    counts_df = select_or_fill(counts_df, valid_genes)

    coords_df.index = np.arange(len(coords_df))
    counts_df.index = np.arange(len(counts_df))

    coords_df_new = coords_df.copy()
    invalid_inds = []
    for rowind, row in df.iterrows():

        coord_df = coords_df[coords_df['slide_index']==rowind]

        svs_prefix = row['slide_id'] 
        with open(row['scalefactors_json'], 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])
            spot_size = int(spot_size)
            patch_size = spot_size

        slide = openslide.open_slide(row['TruePath'])
        save_filename = os.path.join(final_save_root, row['slide_id']+'_patches.tar.gz')
        
        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
        filenames = []
        for rowind1, row1 in coord_df.iterrows():
            xc, yc = row1[X_col_name], row1[Y_col_name]
            patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')

            if use_stain_normalization:
                try:
                    patch = normalizeStaining(np.array(patch))
                except:
                    invalid_inds.append(rowind1)
                    pass

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
        slide.close()
        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())
        
        print(svs_prefix)

    if len(invalid_inds) > 0:
        coords_df_new.drop(invalid_inds, inplace=True)
        counts_df.drop(invalid_inds, inplace=True)

    coords_df_new.to_csv(os.path.join(final_save_root, 'all_coords.csv'), index=False)
    counts_df.to_csv(os.path.join(final_save_root, 'all_counts.csv'), index=False)



def create_train_val(val_patient_ids=[], use_gene_smooth=False, use_stain_normalization=False, ratio=0.1):
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

def create_train_test(use_gene_smooth=False, use_stain_normalization=False, ratio=0.1):
    # train on TNBC
    # test on 10x

    postfix_str = ''
    if use_gene_smooth:
        postfix_str = '_smooth'
    if use_stain_normalization:
        postfix_str = f'{postfix_str}_stain'

    final_save_root = f'/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_{ratio}{postfix_str}'
    train_coords_df = pd.read_csv(os.path.join(final_save_root, 'all_coords.csv'))
    train_counts_df = pd.read_csv(os.path.join(final_save_root, 'all_counts.csv'))
    train_counts_df = np.log10(1.0 + train_counts_df)

    final_save_root = f'/data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated_{ratio}{postfix_str}'
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gene_smooth', type=str, default='False')
    parser.add_argument('--use_stain_normalization', type=str, default='False')
    parser.add_argument('--val_inds', type=str, default='None')
    parser.add_argument('--ratio', type=float, default=0.1)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    val_patient_ids = args.val_inds
    use_gene_smooth = args.use_gene_smooth == 'True'
    use_stain_normalization = args.use_stain_normalization == 'True'
    if val_patient_ids != 'None':
        val_patient_ids = [int(v) for v in val_patient_ids.split(',')]
        print(val_patient_ids)
        create_train_val(val_patient_ids, use_gene_smooth, use_stain_normalization, ratio=args.ratio)
    else:
        create_train_test(use_gene_smooth, use_stain_normalization, ratio=args.ratio)



