#!/bin/bash

#SBATCH --mail-type=FAIL



if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    CACHE_ROOT=/tmp/zhongz2/$SLURM_JOB_ID
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CACHE_ROOT=/lscratch/$SLURM_JOB_ID
fi
export PYTHONPATH=`pwd`:$PYTHONPATH


srun python prepare_ST.py \
--csv_filename ${1} \
--save_root ${2}

exit;

CSV_FILENAME="/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx"
CSV_FILENAME="/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx"
SAVE_ROOT="/data/zhongz2/temp29/ST_prediction_data"
SAVE_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial"
SAVE_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial_meanstd"
mkdir -p $SAVE_ROOT
sbatch --time=24:00:00 \
    --ntasks=16 \
    --ntasks-per-node=1 \
    --partition=multinode \
    --mem=100G \
    --cpus-per-task=2 \
    prepare_ST.sh \
    ${CSV_FILENAME} \
    ${SAVE_ROOT}


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data" \
--use_smooth "False"


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data" \
--use_smooth "True"


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data_fiducial" \
--use_smooth "False"


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data_fiducial" \
--use_smooth "True"


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data_fiducial_meanstd" \
--use_smooth "False"


python prepare_ST_train_val.py \
--train_csv "/data/zhongz2/temp29/ST_prediction/data/TNBC_new.xlsx" \
--val_csv "/data/zhongz2/temp29/ST_prediction/data/10xBreast.xlsx" \
--data_root "/data/zhongz2/temp29/ST_prediction_data_fiducial_meanstd" \
--use_smooth "True"
