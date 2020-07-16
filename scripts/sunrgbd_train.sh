 #!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0
OUTDIR='output/sunrgbd'
MODEL_FILE='models/det_base_sunrgbd.py'
CFG_FILE='cfgs/det_sample_sunrgbd.yaml'
python train/train_net_det.py --cfg $CFG_FILE OUTPUT_DIR $OUTDIR MODEL.FILE $MODEL_FILE
python train/test_net_det_sunrgbd.py --cfg $CFG_FILE OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_final.pth SAVE_SUB_DIR test_gt2D FROM_RGB_DET False
python train/test_net_det_sunrgbd.py --cfg $CFG_FILE OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_final.pth SAVE_SUB_DIR test FROM_RGB_DET True

