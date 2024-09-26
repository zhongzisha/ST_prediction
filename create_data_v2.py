

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

def create_data_v2():

    ensembl_df = pd.read_csv('ensembl.tsv', sep='\t', index_col=0, low_memory=False)
    ensembl_df = ensembl_df[~ensembl_df['Ensembl ID(supplied by Ensembl)'].isna()]
    ensembl_df = ensembl_df.drop_duplicates('Ensembl ID(supplied by Ensembl)')
    gene_symbol_dict = {row['Ensembl ID(supplied by Ensembl)']: row['Approved symbol'] for _, row in ensembl_df.iterrows()}
    all_gene_names = [v.upper() for v in list(gene_symbol_dict.values())]

    root = '/Users/zhongz2/down/He2020/'
    root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
    root = '/data/zhongz2/temp29/ST_prediction/data/He2020'
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
        selected_gene_names = sorted(all_gene_names)

    version = '20240925_secreted'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])

    version = '20240926_secreted'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    # selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])
    with open(os.path.join(root, 'metainfo.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        frequencies_df_sum = tmp['frequencies_df_sum']
        del tmp
    selected_gene_names = sorted(frequencies_df_sum.columns)

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
        existed_gene_names = sorted(list(set(selected_gene_names).intersection(set(vst_df.columns.values))))
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
        fp.write('#!/bin/bash\nfor f in `ls {}/*.tar.gz`;do tar -xf $f; done'.format(save_root))

    with open(os.path.join(save_root, 'meta.pkl'), 'wb') as fp:
        pickle.dump({
            'alldata': alldata,
            'gene_names': selected_gene_names,
            'gene_thres': (low_thres, high_thres)
        }, fp)


def get_selected_gene_names():

    ensembl_df = pd.read_csv('ensembl.tsv', sep='\t', index_col=0, low_memory=False)
    ensembl_df = ensembl_df[~ensembl_df['Ensembl ID(supplied by Ensembl)'].isna()]
    ensembl_df = ensembl_df.drop_duplicates('Ensembl ID(supplied by Ensembl)')
    gene_symbol_dict = {row['Ensembl ID(supplied by Ensembl)']: row['Approved symbol'] for _, row in ensembl_df.iterrows()}
    all_gene_names = sorted([v.upper() for v in list(gene_symbol_dict.values())])

    root = '/Users/zhongz2/down/He2020/'
    root = '/scratch/cluster_scratch/zhongz2/debug/data/He2020'
    root = '/data/zhongz2/temp29/ST_prediction/data/He2020'
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
        selected_gene_names = sorted(all_gene_names)

    version = '20240925_secreted'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])

    version = '20240926_secreted'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])


    alldata = {patient: {} for patient in patients}
    save_root = f'{root}/cache_data/data_{patch_size}_{version}'
    os.makedirs(save_root, exist_ok=True)

    bins = [-np.inf, 0, np.inf]
    sum_df = pd.DataFrame(np.zeros((1, len(all_gene_names)), dtype=np.float32), columns=all_gene_names)
    count_df = pd.DataFrame(np.zeros((1, len(all_gene_names)), dtype=np.int32), columns=all_gene_names)
    # frequencies_df = sum_df.apply(lambda col: pd.cut(col, bins=bins, labels=False, include_lowest=True).value_counts().sort_index())

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
        existed_gene_names = list(set(all_gene_names).intersection(set(vst_df.columns.values)))
        vst_df = vst_df[existed_gene_names]
        # slide = openslide.open_slide(svs_filename)

        sum_df = sum_df + vst_df.sum(axis=0)
        count_df = count_df + vst_df.notna().sum(axis=0)

        # frequencies_df = vst_df.apply(lambda col: pd.cut(col, bins=bins, include_lowest=True).value_counts().sort_index())
        print(svs_prefix)

    sum_df1 = sum_df.copy()
    count_df1 = count_df.copy()

    count_df = count_df.dropna(axis=1)
    count_df = count_df[count_df>0]
    sum_df = sum_df[count_df.columns]
    mean_df = sum_df / count_df
    mean_df.sum(axis=0).sort_values(ascending=False)

    selected_gene_names = sorted(mean_df.columns)

    bins = [-np.inf, 0, np.inf]
    sum_df = pd.DataFrame(np.zeros((1, len(selected_gene_names)), dtype=np.float32), columns=selected_gene_names)
    count_df = pd.DataFrame(np.zeros((1, len(selected_gene_names)), dtype=np.int32), columns=selected_gene_names)
    frequencies_dfs = []

    for rowind, row in df.iterrows():

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
        existed_gene_names = sorted(list(set(selected_gene_names).intersection(set(vst_df.columns.values))))
        vst_df = vst_df[existed_gene_names]

        sum_df = sum_df + vst_df.sum(axis=0)
        count_df = count_df + vst_df.notna().sum(axis=0)

        frequencies_dfs.append(vst_df.apply(lambda col: pd.cut(col, bins=bins, include_lowest=True).value_counts()))
        print(rowind, svs_prefix)

    frequencies_df_sum = frequencies_dfs[0].copy()
    for j in range(1, len(frequencies_dfs)):
        frequencies_df_sum += frequencies_dfs[j]

    sum_df = sum_df[selected_gene_names]
    count_df = count_df[selected_gene_names]
    frequencies_df_sum = frequencies_df_sum[selected_gene_names]


    with open(os.path.join(root, 'metainfo.pkl'), 'wb') as fp:
        pickle.dump({
            'sum_df': sum_df,
            'count_df': count_df,
            'frequencies_df_sum': frequencies_df_sum,
            'bins': bins
        }, fp)




