#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`
NUM_GPUS=${1}
BACKBONE=${2}
LR=${3}
BS=${4}
USE_SMOOTH=${5}
FIX_BACKBONE=${6}
DATA_ROOT=${7}


if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    if [ -d /tmp/zhongz2/data ]; then rm -rf /tmp/zhongz2/data; fi
    CACHE_ROOT=/tmp/zhongz2/$SLURM_JOB_ID/ST_prediction_data
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0
    CACHE_ROOT=/lscratch/$SLURM_JOB_ID/ST_prediction_data
fi
export OMP_NUM_THREADS=8


while
  PORT=$(shuf -n 1 -i 20080-60080)
  netstat -atun | grep -q "${PORT}"
do
  continue
done

mkdir $CACHE_ROOT
cd $CACHE_ROOT
bash $DATA_ROOT/run.sh

cd $current_dir;


echo "begin training"
for VAL_INDEX in {0..22}; do
torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    ST_prediction_exps_v5.py ${NUM_GPUS} ${DATA_ROOT} ${BACKBONE} ${LR} ${BS} ${FIX_BACKBONE} ${USE_SMOOTH} ${VAL_INDEX}

sleep 1
done

exit;

NUM_GPUS=2
DATA_ROOT=/home/zhongz2/ST_prediction/data/He2020/cache_data/data_224_20241002
BACKBONE=resnet50
for LR in 1e-4 1e-5 5e-5 1e-6 5e-6; do
for BS in 32 64; do
for FIX_BACKBONE in "True" "False"; do
for USE_SMOOTH in "True" "False"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:v100x:${NUM_GPUS},lscratch:64 --cpus-per-task=16 --time=108:00:00 --mem=64G \
ST_prediction_exps_v5.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${DATA_ROOT}
done
done
done
done







