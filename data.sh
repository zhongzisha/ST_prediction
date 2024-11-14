#!/bin/bash

CKPT_DIR=${1}
DATA_ROOT=${2}


cd /lscratch/$SLURM_JOB_ID
mkdir images
cd images
if [ -e "$CKPT_DIR/.USE_STAIN" ]; then
for f in `ls ${DATA_ROOT}/TenX*_patches_stain.tar.gz`; do tar -xf $f; done
else
for f in `ls ${DATA_ROOT}/TenX*_patches.tar.gz`; do tar -xf $f; done
fi






