#!/bin/bash

#SBATCH --mail-type=FAIL

current_dir=`pwd`

if [ "$CLUSTER_NAME" == "FRCE" ]; then
    source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
    module load cuda/11.8
    module load cudnn/8.8.3-cuda11
    if [ -d /tmp/zhongz2/data ]; then rm -rf /tmp/zhongz2/data; fi
    DATA_ROOT=/tmp/zhongz2/$SLURM_JOB_ID/data
    if [ ! -d ${DATA_ROOT} ]; then
        echo "prepare data"
        mkdir -p $DATA_ROOT
        cd $DATA_ROOT
        for f in `ls /scratch/cluster_scratch/zhongz2/debug/data_v4_1.3/*.tar.gz`; do
            tar -xf $f;
        done
    fi
else
    source /data/zhongz2/anaconda3/bin/activate th24
    module load CUDA/12.1
    module load cuDNN/8.9.2/CUDA-12
    module load gcc/11.3.0   
    DATA_ROOT=/lscratch/$SLURM_JOB_ID/data
    if [ ! -d ${DATA_ROOT} ]; then
        echo "prepare data"
        mkdir -p $DATA_ROOT
        cd $DATA_ROOT
        for f in `ls /data/zhongz2/temp_ST_prediction/data_v4_1.3/*.tar.gz`; do
            tar -xf $f;
        done
    fi
fi
export OMP_NUM_THREADS=8

NUM_GPUS=${1}
BACKBONE=${2}
LR=${3}
BS=${4}
VAL_INDEX=${5}

while
  PORT=$(shuf -n 1 -i 20080-60080)
  netstat -atun | grep -q "${PORT}"
do
  continue
done

cd $current_dir;


echo "begin training"
torchrun \
    --nnodes=1 \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${PORT} \
    ST_prediction_exps.py ${NUM_GPUS} ${DATA_ROOT} ${BACKBONE} ${LR} ${BS} ${VAL_INDEX} False


exit;

sbatch --ntasks=1 --ntask-per-node=1 --partition=gpu --gres=gpu:v100x:1,lscratch:10 --cpus-per-task=32 --time=108:00:00 --mem=100G \
ST_prediction_exps.sh 4 resnet50 5e-4 64 0

for VAL_INDEX in {0..23}; do 
echo $VAL_INDEX;
sbatch --ntasks=1 --partition=gpu --gres=gpu:8 --cpus-per-task=32 --time=108:00:00 --mem=100G \
ST_prediction_exps.sh 8 resnet50 5e-4 32 ${VAL_INDEX}
done











