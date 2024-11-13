




import sys,os,glob,shutil,pickle,json,argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
import scanpy
import io
import time
import tarfile
import hashlib
import idr_torch

from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
Image.MAX_IMAGE_PIXELS = 12660162500
ImageFile.LOAD_TRUNCATED_IMAGES = True

import cv2


def get_subdir_path(filename):
    hash_value = hashlib.md5(filename.encode()).hexdigest()
    
    subdir1 = hash_value[:2]
    subdir2 = hash_value[2:4]
    subdir3 = hash_value[4:6]
    subdir4 = hash_value[6:8]
    
    subdir_path = os.path.join(subdir1, subdir2, subdir3, subdir4)
    return subdir_path


def RGB2HSD(X): # Hue Saturation Density
    '''
    Function to convert RGB to HSD
    from https://github.com/FarhadZanjani/Histopathology-Stain-Color-Normalization/blob/master/ops.py
    Args:
        X: RGB image
    Returns:
        X_HSD: HSD image
    '''
    eps = np.finfo(float).eps # Epsilon
    X[np.where(X==0.0)] = eps # Changing zeros with epsilon
    OD = -np.log(X / 1.0) # It seems to be calculating the Optical Density
    D  = np.mean(OD,3) # Getting density?
    D[np.where(D==0.0)] = eps # Changing zero densitites with epsilon
    cx = OD[:,:,:,0] / (D) - 1.0 
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    D = np.expand_dims(D,3) # Hue?
    cx = np.expand_dims(cx,3) # Saturation
    cy = np.expand_dims(cy,3) # Density?
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD

def clean_thumbnail(thumbnail):
    '''
    Function to clean thumbnail
    Args:
        thumbnail: thumbnail image
    Returns:
        wthumbnail: cleaned thumbnail image
    '''
    # thumbnail array
    thumbnail_arr = np.asarray(thumbnail)
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    # Remove pen marking here
    # We are skipping this
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD(np.array([wthumbnail.astype('float32')/255.]))[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    # return writable thumbnail
    return wthumbnail



def main():

    save_root='/data/zhongz2/tcga_thumbnails'

    with open('all_brca_svs.txt', 'r') as fp:
        lines = fp.readlines()
    
    indices = np.arange(len(lines))
    index_splits = np.array_split(indices, indices_or_sections=idr_torch.world_size)
    lines = [lines[i].strip() for i in index_splits[idr_torch.rank]]

    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

    filenames = []
    for ind, line in enumerate(lines):
        svs_prefix = os.path.basename(line).replace('.svs', '')

        slide = openslide.open_slide(line)
        W, H = slide.level_dimensions[0]
        scale = 2000. / max(W, H)
        thumbnail = slide.get_thumbnail((int(scale*W), int(scale*H)))

        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = cthumbnail.mean(axis=2) != 255
        ys, xs = np.where(tissue_mask)
        # x, y = xs.max()//2, ys.max()//2
        # x, y = int(x/scale), int(y/scale)
        indexes = np.random.choice(np.arange(len(xs)), size=100)
        size = int(2048*scale)
        valid = 0
        for index in indexes:
            if valid == 10:
                break

            x, y = xs[index], ys[index]
            if len(np.where(tissue_mask[x:x+size, y:y+size])[0])/(size*size) > 0.7:
                x, y = int(x/scale), int(y/scale)
                size = np.random.randint(768, 1280)
                patch = slide.read_region((x, y), level=0, size=(size, size)).convert('RGB')

                # patch = slide.read_region((x, y), level=0, size=(1000, 1000)).convert('RGB')
                # patch.save('/data/zhongz2/temp29/a.png')
                if np.random.rand() < 0.3:
                    scale1 = np.random.randint(60, 120) / 100.
                    w,h = patch.size
                    patch = patch.resize((int(w*scale1), int(h*scale1)), Image.Resampling.LANCZOS)

                filename = f'{svs_prefix}_x{x}_y{y}.jpg'
                filename = os.path.join(get_subdir_path(filename), filename)

                im_buffer = io.BytesIO()
                patch.save(im_buffer, format='JPEG')
                info = tarfile.TarInfo(name=filename)
                info.size = im_buffer.getbuffer().nbytes
                info.mtime = time.time()
                im_buffer.seek(0)
                tar_fp.addfile(info, im_buffer)

                valid += 1
                filenames.append(filename)


    tar_fp.close()
    with open(os.path.join(save_root, 'part{}.tar.gz'.format(idr_torch.rank)), 'wb') as fp:
        fp.write(fh.getvalue())

    with open(os.path.join(save_root, 'part{}.txt'.format(idr_torch.rank)), 'w') as fp:
        fp.writelines([line+'\n' for line in filenames])


if __name__ == '__main__':
    main()











