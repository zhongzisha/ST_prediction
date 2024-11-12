


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
import idr_torch
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_subdir_path(filename):
    hash_value = hashlib.md5(filename.encode()).hexdigest()
    
    subdir1 = hash_value[:2]
    subdir2 = hash_value[2:4]
    subdir3 = hash_value[4:6]
    subdir4 = hash_value[6:8]
    
    subdir_path = os.path.join(subdir1, subdir2, subdir3, subdir4)
    return subdir_path


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



def get_10xBreast():
    excel_filename = '/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx'
    dirs = glob.glob('/data/Jiang_Lab/Data/Zisha_Zhong/10xBreast/*')
    items = []
    cohort_name = 'TenXBreast'
    for ind, d in enumerate(dirs):
        try:
            splits = os.path.basename(d).split('_')
            slide_id = os.path.basename(d)
            data_version = splits[-1]
            patient_id = ind
            image_filename = glob.glob(d+'/*.tif')[0]
            coord_filename = os.path.join(d, 'spatial/tissue_positions_list.csv')
            if not os.path.exists(coord_filename):
                coord_filename = os.path.join(d, 'spatial/tissue_positions.csv')
            scalefactors_json = os.path.join(d, 'spatial/scalefactors_json.json')
            gene_count_filename = glob.glob(d+'/*_filtered_feature_bc_matrix.h5')[0]
        except:
            continue
        items.append(
            (
                cohort_name, data_version, patient_id, slide_id, image_filename, scalefactors_json, coord_filename, gene_count_filename
            )
        )
    df = pd.DataFrame(items, columns=['cohort_name', 'data_version', 'patient_id', 'slide_id', 'image_filename', 'scalefactors_json', 'coord_filename', 'gene_count_filename'])
    
    df.to_excel(excel_filename)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filename', type=str, default='/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx')
    parser.add_argument('--save_root', type=str, default='/data/zhongz2/temp29/ST_prediction_data')

    return parser.parse_args()


def main(args):

    os.makedirs(args.save_root, exist_ok=True)

    if 'xlsx' == args.csv_filename[-4:]:
        df = pd.read_excel(args.csv_filename)
    else:
        df = pd.read_csv(args.csv_filename)

    for col in ['cohort_name', 'data_version', 'patient_id', 'slide_id', 'image_filename', 'scalefactors_json', 'coord_filename', 'gene_count_filename']:
        assert col in df.columns
    assert len(df['slide_id'].unique()) == len(df)  # slide_id within one cohort should be unique

    indices = np.arange(len(df))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size) 
    sub_df = df.iloc[index_splits[idr_torch.rank]]
    sub_df = sub_df.reset_index(drop=True)

    X_col_name = 'pxl_col_in_fullres'
    Y_col_name = 'pxl_row_in_fullres'
    font = ImageFont.load_default(32)

    if len(sub_df) == 0:
        return

    for rowid, row in sub_df.iterrows():

        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        final_save_filename = os.path.join(args.save_root, save_prefix+'_gene_count_smooth.parquet')
        if os.path.exists(final_save_filename):
            continue
        
        with open(row['scalefactors_json'], 'r') as fp:
            tmp = json.load(fp)
            spot_size = float(tmp['spot_diameter_fullres'])
            fiducial_diameter_fullres = float(tmp['fiducial_diameter_fullres'])
            spot_size = int(spot_size)
            patch_size = spot_size

        if 'tissue_positions_list' in row['coord_filename']:
            coord_df = pd.read_csv(row['coord_filename'], header=None, index_col=0, low_memory=False)
        else:
            coord_df = pd.read_csv(row['coord_filename'], index_col=0, low_memory=False)
        coord_df.index.name = 'barcode'
        coord_df.columns = ['in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

        slide = openslide.open_slide(row['image_filename'])
        save_filename = os.path.join(args.save_root, save_prefix+'.jpg')
        if not os.path.exists(save_filename):
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
            img3.save(save_filename)
            del img, draw, img2, draw2, img3

        coord_df = coord_df[coord_df['in_tissue']==1]

        counts_df = scanpy.read_10x_h5(row['gene_count_filename']).to_df().T
        counts_df = counts_df.astype(np.float32)
        counts_df = counts_df.fillna(0)
        counts_df = counts_df.groupby(counts_df.index).sum().T
        counts_df = counts_df.loc[[v for v in coord_df.index.values if v in counts_df.index.values]]
        counts_df.columns = [n.upper() for n in counts_df.columns]  # gene name to UPPER
        counts_df.index.name = 'barcode'

        invalid_col_index = np.where(counts_df.sum(axis=0) == 0)[0]
        if len(invalid_col_index):# invalid genes 
            counts_df = counts_df.drop(columns=counts_df.columns[invalid_col_index])  

        # invalid_row_index = np.where((counts_df != 0).sum(axis=1) < 100)[0]
        # if len(invalid_row_index):# invalid spots 
        #     counts_df = counts_df.drop(index=counts_df.iloc[invalid_row_index].index)

        coord_df = coord_df.loc[[v for v in counts_df.index.values if v in coord_df.index.values]] # only keep those spots with gene counts

        save_filename = os.path.join(args.save_root, save_prefix+'_patches.tar.gz')
        save_filename2 = os.path.join(args.save_root, save_prefix+'_patches_stain.tar.gz')

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
        fh2 = io.BytesIO()
        tar_fp2 = tarfile.open(fileobj=fh2, mode='w:gz')

        filenames = []
        invalid_inds = []
        for rowind1, row1 in coord_df.iterrows():
            xc, yc = row1[X_col_name], row1[Y_col_name]
            patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')

            filename = f'{save_prefix}_x{xc}_y{yc}.jpg'
            filename = os.path.join(get_subdir_path(filename), filename)
            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)
            filenames.append(filename)

            try:
                patch_stain = normalizeStaining(np.array(patch))
            except:
                invalid_inds.append(rowind1)
                continue
        
            filename = f'{save_prefix}_x{xc}_y{yc}.jpg'
            filename = os.path.join(get_subdir_path(filename), filename)
            im_buffer = io.BytesIO()
            patch_stain.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp2.addfile(info, im_buffer)

        slide.close()
        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())
        tar_fp2.close()
        with open(save_filename2, 'wb') as fp:
            fp.write(fh2.getvalue())
        del fh, tar_fp, fh2, tar_fp2

        coord_df.loc[coord_df.index, 'patch_filename'] = filenames

        if len(invalid_inds) > 0:
            coord_df = coord_df.drop(invalid_inds)
            counts_df.drop(invalid_inds, inplace=True)
        
        coord_df.to_csv(os.path.join(args.save_root, save_prefix+'_coord.csv'))
        counts_df.to_parquet(os.path.join(args.save_root, save_prefix+'_gene_count.parquet'), engine='fastparquet')

        counts_df.index = ['{}x{}'.format(row['array_row'], row['array_col']) for rowid, row in coord_df.iterrows()]
        counts_df = gene_neighbor_smoothing(counts_df, use_10x=True)
        counts_df.index = coord_df.index.values
        counts_df.to_parquet(final_save_filename, engine='fastparquet')


if __name__ == '__main__':
    args = get_args()
    main(args)







