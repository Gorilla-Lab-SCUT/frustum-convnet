#!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

python kitti/prepare_data.py --people_only --gen_train --gen_val --gen_val_rgb_detection

OUTDIR1='output/people_train'
python train/train_net_det.py --cfg cfgs/det_sample_people.yaml OUTPUT_DIR $OUTDIR1
python train/test_net_det.py --cfg cfgs/det_sample_people.yaml OUTPUT_DIR $OUTDIR1 TEST.WEIGHTS $OUTDIR1/model_0050.pth

# python train/test_net_det.py --cfg cfgs/det_sample_people.yaml OUTPUT_DIR $OUTDIR1 TEST.WEIGHTS $OUTDIR1/model_best.pth SAVE_SUB_DIR val_nms_best

python kitti/prepare_data_refine.py --people_only --gen_train --gen_val_det --gen_val_rgb_detection
# python kitti/prepare_data_refine.py --people_only --gen_from_folder $OUTDIR1/val_nms_best/result/data

OUTDIR2='output/people_train_refine'
python train/train_net_det.py --cfg cfgs/refine_people.yaml OUTPUT_DIR $OUTDIR2
python train/test_net_det.py --cfg cfgs/refine_people.yaml OUTPUT_DIR $OUTDIR2 TEST.WEIGHTS $OUTDIR2/model_0050.pth

# python train/test_net_det.py --cfg cfgs/refine_people.yaml OUTPUT_DIR $OUTDIR2 TEST.WEIGHTS $OUTDIR/model_best.pth \
#                              OVER_WRITE_TEST_FILE $OUTDIR1/val_nms_best/result/frustum_pedcyc_val_rgb_detection_refine.pickle \
#                              SAVE_SUB_DIR val_nms_best

