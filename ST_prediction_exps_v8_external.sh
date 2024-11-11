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
USE_STAIN=${9}
RATIO=${10}

# NUM_GPUS=2
# BACKBONE=resnet50
# LR=1e-5
# BS=32
# USE_SMOOTH=False
# FIX_BACKBONE=True
# VAL_INDEX="None"
# MAX_EPOCHS=10
# USE_STAIN="True"
# RATIO=0.05

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

POSTFIX_STR=""
if [ "${USE_SMOOTH}" == "True" ]; then 
POSTFIX_STR="_smooth"
echo "use smooth"
fi
if [ "${USE_STAIN}" == "True" ]; then 
POSTFIX_STR="${POSTFIX_STR}_stain"
echo "use stain"
fi

cd $CACHE_ROOT
mkdir images
cd images
for f in `ls /data/zhongz2/temp29/ST_prediction/data/TNBC_generated_${RATIO}${POSTFIX_STR}/*_patches.tar.gz`; do tar -xf $f; done
for f in `ls /data/zhongz2/temp29/ST_prediction/data/10xGenomics_generated_${RATIO}${POSTFIX_STR}/*_patches.tar.gz`; do tar -xf $f; done
cd $current_dir

python prepare_TNBC_v2.py \
--val_inds ${VAL_INDEX} \
--use_gene_smooth ${USE_SMOOTH} \
--use_stain_normalization ${USE_STAIN} \
--ratio ${RATIO}
wait;

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
    ST_prediction_exps_v8.py \
    --num_gpus ${NUM_GPUS} \
    --backbone ${BACKBONE} \
    --lr ${LR} \
    --batch_size ${BS} \
    --fixed_backbone ${FIX_BACKBONE} \
    --use_gene_smooth ${USE_SMOOTH} \
    --val_inds ${VAL_INDEX} \
    --max_epochs ${MAX_EPOCHS} \
    --data_root "/data/zhongz2/temp29/ST_prediction/data/TNBC_generated_${RATIO}${POSTFIX_STR}" \
    --use_stain_normalization ${USE_STAIN}

wait;

rsync -avh $CACHE_ROOT/results /data/zhongz2/temp29/ST_prediction/data/TNBC_generated_${RATIO}${POSTFIX_STR}/



exit;



VAL_INDEX=13 14 6 5 18 17 20 19 22 21 15 24 23 9 2 1 11 12 4 3 8 7


# debug
NUM_GPUS=2
MAX_EPOCHS=100
RATIO=0.05
for BACKBONE in "UNI"; do
for VAL_INDEX in "None"; do
for LR in 1e-6; do
for BS in 128; do
for FIX_BACKBONE in "True"; do
for USE_SMOOTH in "True"; do
for USE_STAIN in "True"; do
sbatch --ntasks=1 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:${NUM_GPUS},lscratch:64 --cpus-per-task=16 --time=108:00:00 --mem=48G \
ST_prediction_exps_v8_external.sh ${NUM_GPUS} ${BACKBONE} ${LR} ${BS} ${USE_SMOOTH} ${FIX_BACKBONE} ${VAL_INDEX} ${MAX_EPOCHS} ${USE_STAIN} ${RATIO}
done
done
done
done
done
done
done




