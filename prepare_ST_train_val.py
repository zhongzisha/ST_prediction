


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

from PIL import Image, ImageFile, ImageDraw, ImageFilter, ImageFont
Image.MAX_IMAGE_PIXELS = 12660162500
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pyarrow.parquet as pq
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx')
    parser.add_argument('--val_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx')
    parser.add_argument('--data_root', type=str, default='/data/zhongz2/temp29/ST_prediction_data')
    parser.add_argument('--use_smooth', type=str, default="True")
    parser.add_argument('--gene_names', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_0.05_smooth_stain/valid_genes.pkl')
    return parser.parse_args()


def get_gene_infos(args, df, gene_names):

    all_count_df = []

    for rowid, row in df.iterrows():

        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        print(save_prefix)

        gene_count_filename = os.path.join(args.data_root, save_prefix+'_gene_count_smooth.parquet' if args.use_smooth == "True" else save_prefix+'_gene_count.parquet')

        parquet_file = pq.ParquetFile(gene_count_filename)
        existing_columns = parquet_file.schema.names

        valid_gene_names = [v for v in gene_names if v in existing_columns]
        count_df = pd.read_parquet(gene_count_filename, columns=valid_gene_names)
        count_df = select_or_fill(count_df, gene_names)
        count_df = np.log10(1.0 + count_df)

        all_count_df.append(count_df)
    
    count_df = pd.concat(all_count_df)
    del all_count_df

    # remove the genes with all identity values, e.g. all zeros
    nunique = count_df.nunique()
    cols_to_drop = nunique[nunique == 1].index
    count_df.drop(cols_to_drop, axis=1, inplace=True)
    valid_gene_names = count_df.columns.values

    # normalize to [0, 1]
    gene_scaler = MinMaxScaler()
    gene_scaler.fit(count_df)

    return gene_scaler, valid_gene_names


def generate_normalized_data(args, df, gene_names, gene_scaler, cache_root, prefix='train'):

    all_counts = []
    for rowid, row in df.iterrows():

        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        print(save_prefix)

        gene_count_filename = os.path.join(args.data_root, save_prefix+'_gene_count_smooth.parquet' if args.use_smooth == "True" else save_prefix+'_gene_count.parquet')

        parquet_file = pq.ParquetFile(gene_count_filename)
        existing_columns = parquet_file.schema.names

        valid_gene_names = [v for v in gene_names if v in existing_columns]
        count_df = pd.read_parquet(gene_count_filename, columns=valid_gene_names)
        count_df = select_or_fill(count_df, gene_names)
        count_df = np.log10(1.0 + count_df)

        count_np = gene_scaler.transform(count_df).astype(np.float32)
        all_counts.append(count_np)

        count_pt = torch.from_numpy(count_np)

        torch.save(count_pt, os.path.join(cache_root, save_prefix+'_gene_count.pth'))

    all_counts = np.concatenate(all_counts)
    np.save(os.path.join(cache_root, prefix + '_all_gene_count.npy'), all_counts)


def draw_histogram(args, cache_root):

    with open(os.path.join(cache_root, 'gene_infos.pkl'), 'rb') as fp:
        gene_names = pickle.load(fp)['gene_names']
    train_counts = np.load(os.path.join(cache_root, 'train_all_gene_count.npy'))
    val_counts = np.load(os.path.join(cache_root, 'val_all_gene_count.npy'))
    train_df = pd.DataFrame(train_counts, columns=gene_names)
    val_df = pd.DataFrame(val_counts, columns=gene_names)
    bins = np.arange(0, 1, 0.1)

    fh = io.BytesIO()
    tar_fp = tarfile.open(fileobj=fh, mode='w:gz')

    for j,gene_name in enumerate(gene_names):
        print(gene_name)
        fig, ax = plt.subplots(nrows=1, ncols=1)

        train_df[gene_name].hist(bins=bins, ax=ax)
        val_df[gene_name].hist(bins=bins, ax=ax)

        im_buffer = io.BytesIO()
        plt.savefig(im_buffer, format='JPEG')
        info = tarfile.TarInfo(name=os.path.join('gene_{:02d}_{}.jpg'.format(j+1, gene_name)))
        info.size = im_buffer.getbuffer().nbytes
        info.mtime = time.time()
        im_buffer.seek(0)
        tar_fp.addfile(info, im_buffer)

        plt.close('all')

    tar_fp.close()

    save_filename = os.path.join(cache_root, 'histogram_figures.tar.gz')
    with open(save_filename, 'wb') as fp:
        fp.write(fh.getvalue())



def main(args):
    cache_root = os.path.join(args.data_root, "exp_smooth{}".format(args.use_smooth))
    os.makedirs(cache_root, exist_ok=True)

    if os.path.isfile(args.gene_names):
        with open(args.gene_names, 'rb') as fp:
            gene_names = pickle.load(fp)['valid_genes']
    else:
        gene_names = [v.upper() for v in args.gene_names.split(',')]
    
    train_df = pd.read_excel(args.train_csv) if 'xlsx' in args.train_csv else pd.read_csv(args.train_csv)
    val_df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    gene_scaler, valid_gene_names = get_gene_infos(args, train_df, gene_names)
    with open(os.path.join(cache_root, 'gene_infos.pkl'), 'wb') as fp:
        pickle.dump({'gene_scaler': gene_scaler, 'gene_names': valid_gene_names}, fp)

    generate_normalized_data(args, train_df, valid_gene_names, gene_scaler, cache_root, prefix='train')
    generate_normalized_data(args, val_df, valid_gene_names, gene_scaler, cache_root, prefix='val')

    draw_histogram(args, cache_root)


if __name__ == '__main__':
    args = get_args()
    print('args', args)
    main(args)
























