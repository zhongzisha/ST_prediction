


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
    parser.add_argument('--use_smooth', action='store_true')
    parser.add_argument('--gene_names', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_0.05_smooth_stain/valid_genes.pkl')
    parser.add_argument('--exp_name', type=str, default='exp0')
    return parser.parse_args()


def get_gene_scaler(args, df, gene_names):

    all_count_df = []

    for rowid, row in df.iterrows():

        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        print(save_prefix)

        gene_count_filename = os.path.join(args.data_root, save_prefix+'_gene_count_smooth.parquet' if args.use_smooth else save_prefix+'_gene_count.parquet')

        parquet_file = pq.ParquetFile(gene_count_filename)
        existing_columns = parquet_file.schema.names

        valid_gene_names = [v for v in gene_names if v in existing_columns]
        count_df = pd.read_parquet(gene_count_filename, columns=valid_gene_names)
        count_df = select_or_fill(count_df, gene_names)
        count_df = np.log10(1.0 + count_df)

        all_count_df.append(count_df)
    
    count_df = pd.concat(all_count_df)
    del all_count_df

    gene_scaler = MinMaxScaler()
    gene_scaler.fit(count_df)

    return gene_scaler


def generate_normalized_data(args, df, gene_names, gene_scaler, cache_root):

    for rowid, row in df.iterrows():

        save_prefix = '{}_{}_{}'.format(row['cohort_name'], row['data_version'], row['slide_id'])
        print(save_prefix)

        # coord_filename = os.path.join(args.data_root, save_prefix+'_coord.csv')
        gene_count_filename = os.path.join(args.data_root, save_prefix+'_gene_count_smooth.parquet' if args.use_smooth else save_prefix+'_gene_count.parquet')

        parquet_file = pq.ParquetFile(gene_count_filename)
        existing_columns = parquet_file.schema.names

        valid_gene_names = [v for v in gene_names if v in existing_columns]
        count_df = pd.read_parquet(gene_count_filename, columns=valid_gene_names)
        count_df = select_or_fill(count_df, gene_names)
        count_df = np.log10(1.0 + count_df)

        count_np = gene_scaler.transform(count_df)
        
        count_np = torch.from_numpy(count_np.astype(np.float32))

        torch.save(count_np, os.path.join(cache_root, save_prefix+'_gene_count.pth'))


def main(args):
    cache_root = os.path.join(args.data_root, args.exp_name)
    os.makedirs(cache_root, exist_ok=True)

    if os.path.isfile(args.gene_names):
        with open(args.gene_names, 'rb') as fp:
            gene_names = pickle.load(fp)['valid_genes']
    else:
        gene_names = [v.upper() for v in args.gene_names.split(',')]
    
    train_df = pd.read_excel(args.train_csv) if 'xlsx' in args.train_csv else pd.read_csv(args.train_csv)
    val_df = pd.read_excel(args.val_csv) if 'xlsx' in args.val_csv else pd.read_csv(args.val_csv)

    gene_scaler = get_gene_scaler(args, train_df, gene_names)

    with open(os.path.join(cache_root, 'gene_infos.pkl'), 'wb') as fp:
        pickle.dump({'gene_scaler': gene_scaler, 'gene_names': gene_names}, fp)

    generate_normalized_data(args, train_df, gene_names, gene_scaler, cache_root)
    generate_normalized_data(args, val_df, gene_names, gene_scaler, cache_root)


if __name__ == '__main__':
    args = get_args()
    print('args', args)
    main(args)
























