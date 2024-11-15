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

CKPT_DIR=${1}
DATA_ROOT=${2}

srun --export ALL --jobid $SLURM_JOB_ID bash data.sh ${CKPT_DIR} ${DATA_ROOT}

wait

cd $current_dir

srun --export ALL --jobid $SLURM_JOB_ID python test.py \
--action "test" \
--ckpt_dir ${CKPT_DIR} \
--data_root ${DATA_ROOT}


exit;


CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-06_b128_e100_accum1_v0_smoothTrue_stainTrue"

CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-06_b128_e200_accum1_v0_smoothTrue_stainTrue"

CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-06_b128_e300_accum1_v0_smoothTrue_stainTrue"
# CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-05_b128_e500_accum1_v0_smoothTrue_stainTrue"

DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data"
sbatch --ntasks=8 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:1,lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=32G \
test.sh ${CKPT_DIR} ${DATA_ROOT}

CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data_fiducial/exp_smoothTrue/results/gpus2/backboneresnet50_fixedTrue/lr1e-06_b128_e200_accum1_v0_smoothTrue_stainTrue"
DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial"
sbatch --ntasks=8 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:1,lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=32G \
test.sh ${CKPT_DIR} ${DATA_ROOT}




CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data_fiducial/exp_smoothTrue/results/gpus2/backboneCTransPath_fixedTrue/lr1e-06_b128_e200_accum1_v0_smoothTrue_stainTrue"
DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial"
sbatch --ntasks=8 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:1,lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=32G \
test.sh ${CKPT_DIR} ${DATA_ROOT}

CKPT_DIR="/data/zhongz2/temp29/ST_prediction_data_fiducial_meanstd/exp_smoothTrue/results/gpus2/backboneCTransPath_fixedTrue/lr1e-06_b128_e200_accum1_v0_smoothTrue_stainTrue_imagenetTrue"
DATA_ROOT="/data/zhongz2/temp29/ST_prediction_data_fiducial_meanstd"
sbatch --ntasks=8 --tasks-per-node=1 --partition=gpu --gres=gpu:a100:1,lscratch:64 --cpus-per-task=10 --time=108:00:00 --mem=32G \
test.sh ${CKPT_DIR} ${DATA_ROOT}













