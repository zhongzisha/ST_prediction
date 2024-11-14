#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`

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
export OMP_NUM_THREADS=8

NUM_GPUS=${1}
BACKBONE=${2}
LR=${3}
BS=${4}
USE_SMOOTH=${5}
FIX_BACKBONE=${6}
MAX_EPOCHS=${7}
USE_STAIN=${8}
DATA_ROOT=${9}

cd $CACHE_ROOT
mkdir images
cd images
if [ "${USE_STAIN}" == "True" ]; then
for f in `ls ${DATA_ROOT}/TNBC*_patches_stain.tar.gz`; do tar -xf $f; done
else
for f in `ls ${DATA_ROOT}/TNBC*_patches.tar.gz`; do tar -xf $f; done
fi

cd $current_dir

while
  PORT=$(shuf -n 1 -i 60000-65000)
  netstat -atun | grep -q "${PORT}"
do
  continue
done
echo $PORT

echo "begin training"

torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost \
    train.py \
    --num_gpus ${NUM_GPUS} \
    --backbone ${BACKBONE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --fixed_backbone ${FIX_BACKBONE} \
    --use_smooth ${USE_SMOOTH} \
    --use_stain ${USE_STAIN} \
    --max_epochs ${MAX_EPOCHS} \
    --data_root ${DATA_ROOT}

exit;



# debug
NUM_GPUS=2
MAX_EPOCHS=200
DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data"
DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial"
for BACKBONE in "resnet50"; do
for VAL_INDEX in "None"; do
for LR in 1e-6; do
for BS in 128; do
for FIX_BACKBONE in "True"; do
for USE_SMOOTH in "True"; do
for USE_STAIN in "True"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:${NUM_GPUS},lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=100G \
train.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${MAX_EPOCHS} ${USE_STAIN} ${DATA_ROOT}
done
done
done
done
done
done
done



NUM_GPUS=2
BACKBONE=resnet50
LR=1e-6
BS=64
USE_SMOOTH="True"
FIX_BACKBONE="True"
MAX_EPOCHS=10
USE_STAIN="True"


















