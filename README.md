# Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection

This repository is the code for our IROS 2019 paper [[arXiv]](https://arxiv.org/abs/1903.01864),[[IEEEXplore]](https://ieeexplore.ieee.org/document/8968513).

## Citation

If you find this work useful in your research, please consider citing.

```BibTeX
@inproceedings{wang2019frustum,
    title={Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection},
    author={Wang, Zhixin and Jia, Kui},
    booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    pages={1742--1749},
    year={2019},
    organization={IEEE}
}
```

## Installation

### Requirements

* PyTorch 1.0+
* Python 3.6+

We test our code under Ubuntu-16.04 with CUDA-9.0, CUDNN-7.0, Python-3.7.2, PyTorch-1.0.

### Clone the repository and install dependencies

```shell
git clone https://github.com/zhixinwang/frustum-convnet.git
```

You may need to install extra packages, like pybind11, opencv, yaml, tensorflow(optional).

If you want to use tensorboard to visualize the training status, you should install tensorflow (CPU version is enough).
Otherwise, you should set the config 'USE_TFBOARD: False' in cfgs/\*.yaml.

### Compile extension

```shell
cd ops
bash clean.sh
bash make.sh
```

## Download data

Download the KITTI 3D object detection dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and organize them as follows.

```text
data/kitti
├── testing
│   ├── calib
│   ├── image_2
│   └── velodyne
└── training
    ├── calib
    ├── image_2
    ├── label_2
    └── velodyne
```

## Training and evaluation

### First stage

Run following command to prepare pickle files for car training. We use the 2D detection results from F-PointNets.
The pickle files will be saved in `kitti/data/pickle_data`.

```shell
python kitti/prepare_data.py --car_only --gen_train --gen_val --gen_val_rgb_detection
```

Run following commands to train and evaluate the final model. You can use `export CUDA_VISIBLE_DEVICES=?` to specify which GPU to use.
And you can modify the setting after `OUTPUT_DIR` to set a directory to save the log, model files and evaluation results.  All the config settings are under the configs/config.py.

```shell
python train/train_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR output/car_train
python train/test_net_det.py --cfg cfgs/det_sample.yaml OUTPUT_DIR output/car_train TEST.WEIGHTS output/car_train/model_0050.pth
```

We also provide the shell script, so you can also run `bash scripts/car_train.sh` instead.

### Refinement stage

Run following command to prepare pickle files for car training. We use the first stage predicted results. If you don't use the default directory in the first stage, you should change the corresponding directory in [here](kitti/prepare_data_refine.py#L888) and [here](kitti/prepare_data_refine.py#L904) before running following commands. The pickle files will be saved in `kitti/data/pickle_data_refine`.

```shell
python kitti/prepare_data_refine.py --car_only --gen_train --gen_val_det --gen_val_rgb_detection
```

Run following commands to train and evaluate the final model.

```shell
python train/train_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR output/car_train_refine
python train/test_net_det.py --cfg cfgs/refine_car.yaml OUTPUT_DIR output/car_train_refine TEST.WEIGHTS output/car_train_refine/model_0050.pth
```

We also provide the shell script, so you can also run `bash scripts/car_train_refine.sh` instead.

### All commands in one script file

You can simply run `bash scripts/car_all.sh` to execute all the above commands.

## Pretrained models
We provide the pretrained models for car category, you can download from [here](https://drive.google.com/open?id=1z7bBVOjtJx6qW0oKP1EcQxECqq0HP3_9).
After extracting the files under root directory, you can run `bash scripts/eval_pretrained_models.sh` to evaluate the pretrained models.
The performance on validation set is as follows:

```text
# first stage
Car AP@0.70, 0.70, 0.70:
bbox AP:98.33, 90.40, 88.24
bev  AP:90.32, 88.02, 79.41
3d   AP:87.76, 77.41, 68.79

# refinement stage
Car AP@0.70, 0.70, 0.70:
bbox AP:98.43, 90.39, 88.15
bev  AP:90.42, 88.99, 86.88
3d   AP:89.31, 79.08, 77.17

```

## SUNRGBD dataset

Please follow the instruction [here](sunrgbd/README.md).

## Note

Since we update our code from PyTorch-0.3.1 to PyTorch-1.0 and our code uses many random sampling operations, the results may be not exactly the same as those reported in our paper.
But the difference should be +-0.5\%, if you can not get the similar results, please contact me. I am still working to make results stable.

Our code is supported multiple GPUs for training, but now the training is very fast for small dataset, like KITTI, SUN-RGBD. All the steps will finish in one day on single GPU.


## Acknowledgements

Part of the code was adapted from [F-PointNets](https://github.com/charlesq34/frustum-pointnets).

## License

Our code is released under [MIT license](LICENSE).
