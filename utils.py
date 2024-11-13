
import argparse





def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--action', type=str, default='train')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--trigger_eval_every', type=int, default=10)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--fixed_backbone', type=str, default="True")
    parser.add_argument('--use_smooth', type=str, default="True")
    parser.add_argument('--use_stain', type=str, default="True")
    parser.add_argument('--train_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx')
    parser.add_argument('--val_csv', type=str, default='/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx')
    parser.add_argument('--data_root', type=str, default='/data/zhongz2/temp29/ST_prediction_data')
    parser.add_argument('--ckpt_dir', type=str, default='/data/zhongz2/temp29/ST_prediction_data/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-05_b32_e100_accum1_v0_smoothTrue_stainTrue')
    parser.add_argument('--gene_names', type=str, default='') # seperated by comma, a list of genes of interest

    return parser.parse_args()