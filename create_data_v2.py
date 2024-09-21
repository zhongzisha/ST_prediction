

import sys,os,glob,shutil,pickle,json,io,tarfile,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
# import idr_torch
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

ensembl_df = pd.read_csv('ensembl.tsv', sep='\t', index_col=0, low_memory=False)
ensembl_df = ensembl_df[~ensembl_df['Ensembl ID(supplied by Ensembl)'].isna()]
ensembl_df = ensembl_df.drop_duplicates('Ensembl ID(supplied by Ensembl)')
gene_symbol_dict = {row['Ensembl ID(supplied by Ensembl)']: row['Approved symbol'] for _, row in ensembl_df.iterrows()}


root = '/Users/zhongz2/down/He2020/'
root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
df = pd.read_csv(f'{root}/metadata.csv')
low_thres, high_thres = -4, 4
patch_size = 224
patients = df['patient'].unique()
version = '20240920_v1'
selected_gene_names = sorted([
    'GNAS', 'ACTG1', 'FASN', 'DDX5', 'XBP1'
])

version = '20240920_all'
selected_gene_names = []
if len(selected_gene_names) == 0:
    selected_gene_names = sorted(list(gene_symbol_dict.values()))

use_gene_tar = True

alldata = {patient: {} for patient in patients}
save_root = f'{root}/cache_data/data_{patch_size}_{version}'
os.makedirs(save_root, exist_ok=True)
for _, row in df.iterrows():

    svs_prefix = row['histology_image'].replace('.jpg', '')
    save_filename = os.path.join(save_root, svs_prefix+'.tar.gz')
    patient = row['patient']
    alldata[patient][svs_prefix] = {}
    vst_filename = f'{root}/{svs_prefix}_gene_vst.tsv'
    svs_filename = f'{root}/{svs_prefix}.jpg'
    coord_filename = '{}/{}'.format(root, row['spot_coordinates'])

    coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False).astype(np.int32)

    vst_df = pd.read_csv(vst_filename, sep='\t', low_memory=False).T
    common_spots = sorted(list(set(coord_df.index.values).intersection(set(vst_df.index.values))))
    coord_df = coord_df.loc[common_spots]
    vst_df = vst_df.loc[common_spots]
    vst_df = vst_df.rename(columns=gene_symbol_dict)
    vst_df = vst_df.clip(lower=low_thres, upper=high_thres, axis=1)
    existed_gene_names = list(set(selected_gene_names).intersection(set(vst_df.columns.values)))
    vst_df = vst_df[existed_gene_names]
    slide = openslide.open_slide(svs_filename)

    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')
    items = []
    mean = np.zeros((3, ), dtype=np.float32)
    std = np.zeros((3, ), dtype=np.float32)
    for (_, row1), (_, row2) in zip(coord_df.iterrows(), vst_df.iterrows()):
        xc, yc = row1['X'], row1['Y']
        patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')
        label = row2.values.tolist()

        patch_filename = None
        if use_gene_tar:
            patch_filename = os.path.join(svs_prefix, f'x{xc}_y{yc}.jpg')
            im_buffer = io.BytesIO()
            patch.save(im_buffer, format='JPEG')
            info = tarfile.TarInfo(name=patch_filename)
            info.size = im_buffer.getbuffer().nbytes
            info.mtime = time.time()
            im_buffer.seek(0)
            tar_fp.addfile(info, im_buffer)


        labels_dict = {k:np.nan for k in selected_gene_names}
        labels_dict.update(row2.to_dict())

        if use_gene_tar:
            label = ','.join(['{:.3f}'.format(v) if v is not np.nan else 'nan' for k,v in labels_dict.items()])
            txt_buffer = io.StringIO(label)
            btxt_buffer = io.BytesIO(txt_buffer.read().encode())
            txt_filename = os.path.join(svs_prefix, f'x{xc}_y{yc}.txt')
            info = tarfile.TarInfo(name=txt_filename)
            info.size = btxt_buffer.getbuffer().nbytes
            info.mtime = time.time()
            btxt_buffer.seek(0)
            tar_fp.addfile(info, btxt_buffer)

        if use_gene_tar:
            items.append((patch_filename, txt_filename))
        else:
            if patch_filename is None:
                items.append((patch, label))
            else:
                items.append((patch_filename, label))

        patch = np.array(patch, dtype=np.float32) / 255
        mean += patch.mean((0, 1))
        std += patch.std((0, 1))
    mean /= len(vst_df)
    std /= len(vst_df)

    tar_fp.close()
    with open(save_filename, 'wb') as fp:
        fp.write(fh.getvalue())

    print(svs_prefix)

    alldata[patient][svs_prefix]['data'] = items
    alldata[patient][svs_prefix]['mean'] = mean
    alldata[patient][svs_prefix]['std'] = std

with open(os.path.join(save_root, 'run.sh'), 'w') as fp:
    fp.write('#!/bin/bash\nfor f in `ls *.tar.gz`;do tar -xf $f; done')
with open(os.path.join(save_root, 'meta.pkl'), 'wb') as fp:
    pickle.dump({
        'alldata': alldata,
        'gene_names': selected_gene_names,
        'gene_thres': (low_thres, high_thres)
    }, fp)



# check the data
with open(os.path.join(root, 'alldata_FASN.pkl'), 'rb') as fp:
    alldata = pickle.load(fp)

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

trnY = [item[1] for item in train_data]
valY = [item[1] for item in val_data]
trn_df = pd.DataFrame(trnY, columns=['FASN']).clip(lower=low_thres, upper=high_thres, axis=1)
val_df = pd.DataFrame(valY, columns=['FASN']).clip(lower=low_thres, upper=high_thres, axis=1)
plt.close()
bins = np.arange(low_thres, high_thres, 1)
trn_df['FASN'].hist(bins=bins)
val_df['FASN'].hist(bins=bins)
plt.savefig('FASN.jpg')
plt.close()












