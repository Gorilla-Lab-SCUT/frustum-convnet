#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python kitti/prepare_data.py --car_only --gen_train --gen_val --gen_val_rgb_detection 

OUTDIR='output/car_train'
python train/train_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR $OUTDIR
python train/test_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_0050.pth 

python kitti/prepare_data_refine.py --car_only --gen_train --gen_val_det --gen_val_rgb_detection

OUTDIR='output/car_train_refine'
python train/train_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR $OUTDIR 
python train/test_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_0050.pth