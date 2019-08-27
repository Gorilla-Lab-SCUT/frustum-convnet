#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
OUTDIR='pretrained_models/car'

python kitti/prepare_data.py --car_only --gen_val_rgb_detection 
python train/test_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_0050.pth 
python kitti/prepare_data_refine.py --car_only --gen_from_folder pretrained_models/car/val_nms/result/data

OUTDIR='pretrained_models/car_refine'
python train/test_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_0050.pth \
                             OVER_WRITE_TEST_FILE pretrained_models/car/val_nms/result/frustum_caronly_val_rgb_detection_refine.pickle
