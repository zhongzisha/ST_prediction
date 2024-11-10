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

# NUM_GPUS=2
# BACKBONE=resnet50
# LR=1e-5
# BS=32
# USE_SMOOTH=False
# FIX_BACKBONE=True
# VAL_INDEX="None"
# MAX_EPOCHS=10

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

cd $CACHE_ROOT
mkdir images
cd images
for f in `ls /data/zhongz2/temp29/ST_prediction/data/TNBC_generated/*_patches.tar.gz`; do tar -xf $f; done
for f in `ls /data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated/*_patches.tar.gz`; do tar -xf $f; done
cd $current_dir

python prepare_TNBC_v2.py ${VAL_INDEX}
wait;

while
  PORT=$(shuf -n 1 -i 20080-60080)
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
    ST_prediction_exps_v8.py \
    --num_gpus ${NUM_GPUS} \
    --backbone ${BACKBONE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --fixed_backbone ${FIX_BACKBONE} \
    --use_vst_smooth ${USE_SMOOTH} \
    --val_inds ${VAL_INDEX} \
    --max_epochs ${MAX_EPOCHS} \
    --data_root "/data/zhongz2/temp29/ST_prediction/data/TNBC_generated"


wait;

rsync -avh $CACHE_ROOT/results /data/zhongz2/temp29/ST_prediction/data/TNBC_generated/



exit;



VAL_INDEX=13 14 6 5 18 17 20 19 22 21 15 24 23 9 2 1 11 12 4 3 8 7


NUM_GPUS=2
BACKBONE=resnet50
MAX_EPOCHS=300
for VAL_INDEX in "None"; do
for LR in 1e-5 5e-6 1e-6; do
for BS in 32 64; do
for FIX_BACKBONE in "True"; do
for USE_SMOOTH in "False"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:${NUM_GPUS},lscratch:64 --cpus-per-task=16 --time=108:00:00 --mem=64G \
ST_prediction_exps_v8_external.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${VAL_INDEX} ${MAX_EPOCHS}
done
done
done
done
done

NUM_GPUS=2
BACKBONE=resnet50
MAX_EPOCHS=100
for VAL_INDEX in "None"; do
for LR in 1e-5 5e-5 1e-6; do
for BS in 32 64; do
for FIX_BACKBONE in "False"; do
for USE_SMOOTH in "False"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:${NUM_GPUS},lscratch:64 --cpus-per-task=16 --time=108:00:00 --mem=64G \
ST_prediction_exps_v8_external.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${VAL_INDEX} ${MAX_EPOCHS}
done
done
done
done
done






