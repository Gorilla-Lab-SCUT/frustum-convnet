 #!/bin/bash
set -x
set -e

export CUDA_VISIBLE_DEVICES=0

OUTDIR='output/car_train'
python train/train_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR $OUTDIR
python train/test_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR $OUTDIR TEST.WEIGHTS $OUTDIR/model_0050.pth 