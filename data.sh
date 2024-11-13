#!/bin/bash

CKPT_DIR=${1}


cd /lscratch/$SLURM_JOB_ID
mkdir images
cd images
if [ -e "$CKPT_DIR/.USE_STAIN" ]; then
for f in `ls /data/zhongz2/temp29/ST_prediction_data/TenX*_patches_stain.tar.gz`; do tar -xf $f; done
else
for f in `ls /data/zhongz2/temp29/ST_prediction_data/TenX*_patches.tar.gz`; do tar -xf $f; done
fi






