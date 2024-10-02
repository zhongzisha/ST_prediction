

import sys,os,glob,shutil,pickle,json,io,tarfile,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openslide
# import idr_torch
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 12660162500
from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
ImageFile.LOAD_TRUNCATED_IMAGES = True


def neighbor_smoothing_vst(vst_df):
    spot_names = vst_df.index.values
    new_vst_df = vst_df.copy()
    for s in spot_names:
        r, c = [int(v) for v in s.split('x')]
        ns = ['{}x{}'.format(rr, cc) for rr in [r-1, r, r+1] for cc in [c-1, c, c+1]]
        ns = [v for v in ns if v in spot_names]
        new_vst_df.loc[s] = vst_df.loc[ns].mean()
    return new_vst_df


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
    low_thres, high_thres = -2, 6
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

    version = '20240927'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    # selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])
    with open(os.path.join(root, 'metainfo.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        frequencies_df_sum = tmp['frequencies_df_sum']
        bins = tmp['bins'] if 'bins' in tmp else [-np.inf, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf]
        bin_labels = tmp['bin_labels'] if 'bin_labels' in tmp else np.array([-2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        del tmp
    selected_gene_names = sorted(frequencies_df_sum.columns)
    selected_gene_names = sorted([
        'GNAS', 'ACTG1', 'FASN', 'DDX5', 'XBP1', 'C3'
    ])

    version = '20240927_v3'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    # selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])
    with open(os.path.join(root, 'metainfo.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        sum_df = tmp['sum_df']
        count_df = tmp['count_df']
        frequencies_df_sum = tmp['frequencies_df_sum']
        bins = tmp['bins'] if 'bins' in tmp else [-np.inf, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf]
        bin_labels = tmp['bin_labels'] if 'bin_labels' in tmp else np.array([-2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        del tmp
    selected_gene_names = sorted(frequencies_df_sum.columns)
    count_df = count_df.dropna(axis=1)
    count_df = count_df[count_df>0]
    sum_df = sum_df[count_df.columns]
    mean_df = sum_df / count_df
    selected_gene_names = sorted(list(set(protein_df.index.values).intersection(mean_df.columns.values)))

    version = '20241002'
    selected_gene_names = []
    protein_df = pd.read_csv('protein_class_Predicted.tsv', sep='\t', index_col=0)
    # selected_gene_names = sorted([v.upper() for v in protein_df.index.values.tolist() if v.upper() in all_gene_names])
    with open(os.path.join(root, 'metainfo.pkl'), 'rb') as fp:
        tmp = pickle.load(fp)
        sum_df = tmp['sum_df']
        count_df = tmp['count_df']
        frequencies_df_sum = tmp['frequencies_df_sum']
        bins = tmp['bins'] if 'bins' in tmp else [-np.inf, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf]
        bin_labels = tmp['bin_labels'] if 'bin_labels' in tmp else np.array([-2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        del tmp
    selected_gene_names = sorted(frequencies_df_sum.columns)
    count_df = count_df.dropna(axis=1)
    count_df = count_df[count_df>0]
    sum_df = sum_df[count_df.columns]
    mean_df = sum_df / count_df
    selected_gene_names = sorted(list(set(protein_df.index.values).intersection(mean_df.columns.values)))


    use_gene_tar = True
    font = ImageFont.load_default(32)

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
        vst_df = vst_df[existed_gene_names].astype(np.float32)
        # vst_df_cut = vst_df.apply(lambda col: pd.cut(col, bins=bins, labels=False,include_lowest=True))
        slide = openslide.open_slide(svs_filename)

        if False:
            spot_size = 224
            patch_size = 224
            X_col_name = 'X'
            Y_col_name = 'Y'
            # plot spot figure
            W, H = slide.level_dimensions[0]
            img = slide.read_region((0, 0), 0, (W, H)).convert('RGB')
            draw = ImageDraw.Draw(img)
            img2 = Image.fromarray(255*np.ones((H, W, 3), dtype=np.uint8))
            draw2 = ImageDraw.Draw(img2)
            circle_radius = int(spot_size * 0.5)
            # colors = np.concatenate([colors, 128*np.ones((colors.shape[0], 1), dtype=np.uint8)], axis=1)
            for ind, row1 in coord_df.iterrows():
                x, y = row1[X_col_name], row1[Y_col_name]
                xy = [x-circle_radius, y-circle_radius, x+circle_radius, y+circle_radius]
                draw.ellipse(xy, outline=(255, 128, 0), width=8)
                x -= patch_size // 2
                y -= patch_size // 2
                xy = [x, y, x+patch_size, y+patch_size]
                draw2.rectangle(xy, fill=(144, 238, 144))
                draw.text((x, y),str(ind),(255,255,255),font=font)
            img3 = Image.blend(img, img2, alpha=0.4)
            # img3.save(spot_vis_filename)

        fh = io.BytesIO()
        tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

        mean = np.zeros((3, ), dtype=np.float32)
        std = np.zeros((3, ), dtype=np.float32)
        for _, row1 in coord_df.iterrows():
            xc, yc = row1['X'], row1['Y']
            patch = slide.read_region((int(xc - patch_size//2), int(yc - patch_size//2)), 0, (patch_size, patch_size)).convert('RGB')

            patch_filename = os.path.join(svs_prefix, f'x{xc}_y{yc}.jpg')
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
        mean /= len(vst_df)
        std /= len(vst_df)

        tar_fp.close()
        with open(save_filename, 'wb') as fp:
            fp.write(fh.getvalue())

        print(svs_prefix)

        alldata[patient][svs_prefix]['coord_df'] = coord_df
        alldata[patient][svs_prefix]['vst_df'] = vst_df
        alldata[patient][svs_prefix]['mean'] = mean
        alldata[patient][svs_prefix]['std'] = std

    with open(os.path.join(save_root, 'run.sh'), 'w') as fp:
        fp.write('#!/bin/bash\nfor f in `ls {}/*.tar.gz`;do tar -xf $f; done'.format(save_root))

    with open(os.path.join(save_root, 'meta.pkl'), 'wb') as fp:
        pickle.dump({
            'alldata': alldata,
            'gene_names': selected_gene_names,
            'gene_thres': (low_thres, high_thres),
            'bins': bins,
            'bin_labels': bin_labels
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
    patch_size = 224
    patients = df['patient'].unique()

    sum_df = pd.DataFrame(np.zeros((1, len(all_gene_names)), dtype=np.float32), columns=all_gene_names)
    count_df = pd.DataFrame(np.zeros((1, len(all_gene_names)), dtype=np.int32), columns=all_gene_names)

    for _, row in df.iterrows():

        svs_prefix = row['histology_image'].replace('.jpg', '')
        patient = row['patient']
        vst_filename = f'{root}/{svs_prefix}_gene_vst.tsv'
        svs_filename = f'{root}/{svs_prefix}.jpg'
        coord_filename = '{}/{}'.format(root, row['spot_coordinates'])

        coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False).astype(np.int32)

        vst_df = pd.read_csv(vst_filename, sep='\t', low_memory=False).T
        common_spots = sorted(list(set(coord_df.index.values).intersection(set(vst_df.index.values))))
        coord_df = coord_df.loc[common_spots]
        vst_df = vst_df.loc[common_spots]
        vst_df = vst_df.rename(columns=gene_symbol_dict)
        existed_gene_names = list(set(all_gene_names).intersection(set(vst_df.columns.values)))
        vst_df = vst_df[existed_gene_names]

        sum_df = sum_df + vst_df.sum(axis=0)
        count_df = count_df + vst_df.notna().sum(axis=0)

        print(svs_prefix)

    sum_df1 = sum_df.copy()
    count_df1 = count_df.copy()

    count_df = count_df.dropna(axis=1)
    count_df = count_df[count_df>0]
    sum_df = sum_df[count_df.columns]
    mean_df = sum_df / count_df
    mean_df.sum(axis=0).sort_values(ascending=False)

    selected_gene_names = sorted(mean_df.columns)

    bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf]
    bin_labels = np.array([-2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    sum_df = pd.DataFrame(np.zeros((1, len(selected_gene_names)), dtype=np.float32), columns=selected_gene_names)
    count_df = pd.DataFrame(np.zeros((1, len(selected_gene_names)), dtype=np.int32), columns=selected_gene_names)
    frequencies_dfs = []

    for rowind, row in df.iterrows():

        svs_prefix = row['histology_image'].replace('.jpg', '')
        patient = row['patient']
        vst_filename = f'{root}/{svs_prefix}_gene_vst.tsv'
        svs_filename = f'{root}/{svs_prefix}.jpg'
        coord_filename = '{}/{}'.format(root, row['spot_coordinates'])

        coord_df = pd.read_csv(coord_filename, index_col=0, low_memory=False).astype(np.int32)

        vst_df = pd.read_csv(vst_filename, sep='\t', low_memory=False).T
        common_spots = sorted(list(set(coord_df.index.values).intersection(set(vst_df.index.values))))
        coord_df = coord_df.loc[common_spots]
        vst_df = vst_df.loc[common_spots]
        vst_df = vst_df.rename(columns=gene_symbol_dict)
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
            'bins': bins,
            'bin_labels': bin_labels
        }, fp)




