### Prepare SUN RGB-D data

1. Download SUNRGBD v1 data from [HERE](http://rgbd.cs.princeton.edu/data/) (SUNRGBD.zip) and the toolbox (SUNRGBDtoolbox.zip). Move all the downloaded files under data/sunrgbd.
Unzip the zip files. Ensure the directory is like this.

```text
data/sunrgbd/
├── SUNRGBD
    ├── kv1
    ├── kv2
    ├── realsense
    └── xtion
└── SUNRGBDtoolbox
```

2. Change the `PROJECT_DIR` of `extract_rgbd_data.m` according to your project position.
Extract point clouds and annotations by running `extract_rgbd_data.m` under the `sunrgbd/matlab` folder.
It will re-organize the original dataset in  `sunrgbd/mysunrgbd` folder.
```text
sunrgbd/mysunrgbd
└── training
    ├── calib
    ├── depth
    ├── image
    ├── label
    └── pc
```

3. Extract frustum data by running `python sunrgbd/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection`

### Training and evaluation
```shell
bash scripts/sunrgbd_train.sh
```