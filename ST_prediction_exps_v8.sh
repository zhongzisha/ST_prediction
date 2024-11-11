#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`
NUM_GPUS=${1}
BACKBONE=${2}
LR=${3}
BS=${4}
USE_SMOOTH=${5}
FIX_BACKBONE=${6}
VAL_INDEX=${7}
MAX_EPOCHS=${8}


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    if [ -d /tmp/zhongz2/data ]; then rm -rf /tmp/zhongz2/data; fi
    CACHE_ROOT=/tmp/zhongz2/$SLURM_JOB_ID
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CACHE_ROOT=/lscratch/$SLURM_JOB_ID
fi
# export OMP_NUM_THREADS=4

cd $CACHE_ROOT
mkdir images
cd images
for f in `ls /data/zhongz2/temp29/ST_prediction/data/TNBC/*_patches.tar.gz`; do tar -xf $f; done
cd $current_dir

python prepare_TNBC_v2.py \
--val_inds ${VAL_INDEX} \
--use_gene_smooth ${USE_SMOOTH}
wait;

while
  PORT=$(shuf -n 1 -i 20080-60080)
  netstat -atun | grep -q "${PORT}"
do
  continue
done

echo "begin training"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    ST_prediction_exps_v8.py \
    --num_gpus ${NUM_GPUS} \
    --backbone ${BACKBONE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --fixed_backbone ${FIX_BACKBONE} \
    --use_gene_smooth ${USE_SMOOTH} \
    --val_inds ${VAL_INDEX} \
    --max_epochs ${MAX_EPOCHS} \
    --data_root "/data/zhongz2/temp29/ST_prediction/data/TNBC"


wait;

rsync -avh $CACHE_ROOT/results /data/zhongz2/temp29/ST_prediction/data/TNBC/



exit;



VAL_INDEX=13 14 6 5 18 17 20 19 22 21 15 24 23 9 2 1 11 12 4 3 8 7


NUM_GPUS=2
BACKBONE=resnet50
MAX_EPOCHS=300
for VAL_INDEX in 14 6 5 18 17 20 19 22 21 15 24 23 9 2 1 11 12 4 3 8 7; do
for LR in 1e-5 1e-6; do
for BS in 64; do
for FIX_BACKBONE in "True"; do
for USE_SMOOTH in "False"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:v100x:${NUM_GPUS},lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=64G \
ST_prediction_exps_v8.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${VAL_INDEX} ${MAX_EPOCHS}
done
done
done
done
done


NUM_GPUS=2
BACKBONE=resnet50
MAX_EPOCHS=100
for VAL_INDEX in 13; do
for LR in 1e-4 1e-5 5e-5 1e-6 5e-6; do
for BS in 32 64; do
for FIX_BACKBONE in "False"; do
for USE_SMOOTH in "False"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:v100x:${NUM_GPUS},lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=64G \
ST_prediction_exps_v8.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${VAL_INDEX} ${MAX_EPOCHS}
done
done
done
done
done






