from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import shutil
import time
import argparse

import pprint
import random as pyrandom
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from configs.config import cfg
from configs.config import merge_cfg_from_file
from configs.config import merge_cfg_from_list
from configs.config import assert_and_infer_cfg

from utils.training_states import TrainingStates
from utils.utils import get_accuracy, AverageMeter, import_from_file, get_logger

from datasets.dataset_info import DATASET_INFO

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a network with Detectron'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file for training (and optionally testing)',
        default=None,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See configs/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def set_random_seed(seed=3):
    pyrandom.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_module_bn_momentum(model, momentum=0.1):
    def set_bn_momentum(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.momentum = momentum

    model.apply(set_bn_momentum)


def get_bn_decay(epoch):
    # 0.5 - 0.01
    BN_INIT_DECAY = 0.1
    BN_DECAY_RATE = 0.5
    BN_DECAY_STEP = cfg.TRAIN.LR_STEPS
    BN_DECAY_CLIP = 0.01
    bn_momentum = max(BN_INIT_DECAY * BN_DECAY_RATE ** (epoch // BN_DECAY_STEP), BN_DECAY_CLIP)

    return bn_momentum


def train(data_loader, model, optimizer, lr_scheduler, epoch, logger=None):

    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    model.train()

    MIN_LR = cfg.TRAIN.MIN_LR
    lr_scheduler.step(epoch)
    if MIN_LR > 0:
        if lr_scheduler.get_lr()[0] < MIN_LR:
            for param_group in optimizer.param_groups:
                param_group['lr'] = MIN_LR

    cur_lr = optimizer.param_groups[0]['lr']
    # cur_mom = get_bn_decay(epoch)
    # set_module_bn_momentum(model, cur_mom)

    tic = time.time()
    loader_size = len(data_loader)

    training_states = TrainingStates()

    for i, (data_dicts) in enumerate(data_loader):

        data_time_meter.update(time.time() - tic)

        batch_size = data_dicts['point_cloud'].shape[0]

        data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}
        optimizer.zero_grad()

        losses, metrics = model(data_dicts_var)
        loss = losses['total_loss']

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        # mean for multi-gpu setting
        losses_reduce = {key: value.detach().mean().item() for key, value in losses.items()}
        metrics_reduce = {key: value.detach().mean().item() for key, value in metrics.items()}

        training_states.update_states(dict(**losses_reduce, **metrics_reduce), batch_size)

        batch_time_meter.update(time.time() - tic)
        tic = time.time()

        if (i + 1) % cfg.disp == 0 or (i + 1) == loader_size:

            states = training_states.get_states(avg=False)

            states_str = training_states.format_states(states)
            output_str = 'Train Epoch: {:03d} [{:04d}/{}] lr:{:.6f} Time:{:.3f}/{:.3f} ' \
                .format(epoch + 1, i + 1, len(data_loader), cur_lr, data_time_meter.val, batch_time_meter.val)

            logging.info(output_str + states_str)

            if (i + 1) == loader_size:
                states = training_states.get_states(avg=True)
                states_str = training_states.format_states(states)
                output_str = 'Train Epoch(AVG): {:03d} [{:04d}/{}] lr:{:.6f} Time:{:.3f}/{:.3f} ' \
                    .format(epoch + 1, i + 1, len(data_loader), cur_lr, data_time_meter.val, batch_time_meter.val)
                logging.info(output_str + states_str)

        if logger is not None:
            states = training_states.get_states(avg=True)
            for tag, value in states.items():
                logger.scalar_summary(tag, value, int(epoch))


def validate(data_loader, model, epoch, logger=None):
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    model.eval()

    tic = time.time()
    loader_size = len(data_loader)

    training_states = TrainingStates()

    for i, (data_dicts) in enumerate(data_loader):
        data_time_meter.update(time.time() - tic)

        batch_size = data_dicts['point_cloud'].shape[0]

        with torch.no_grad():
            data_dicts_var = {key: value.cuda() for key, value in data_dicts.items()}

            losses, metrics = model(data_dicts_var)
            # mean for multi-gpu setting
            losses_reduce = {key: value.detach().mean().item() for key, value in losses.items()}
            metrics_reduce = {key: value.detach().mean().item() for key, value in metrics.items()}

        training_states.update_states(dict(**losses_reduce, **metrics_reduce), batch_size)

        batch_time_meter.update(time.time() - tic)
        tic = time.time()

    states = training_states.get_states(avg=True)

    states_str = training_states.format_states(states)
    output_str = 'Validation Epoch: {:03d} Time:{:.3f}/{:.3f} ' \
        .format(epoch + 1, data_time_meter.val, batch_time_meter.val)

    logging.info(output_str + states_str)

    if logger is not None:
        for tag, value in states.items():
            logger.scalar_summary(tag, value, int(epoch))

    return states['IoU_' + str(cfg.IOU_THRESH)]


def main():
    # parse arguments
    args = parse_args()

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)

    if args.opts is not None:
        merge_cfg_from_list(args.opts)

    assert_and_infer_cfg()

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    # set logger
    cfg_name = os.path.basename(args.cfg_file).split('.')[0]
    log_file = '{}_{}_train.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    log_file = os.path.join(cfg.OUTPUT_DIR, log_file)
    logger = get_logger(log_file)

    logger.info(pprint.pformat(args))
    logger.info('config:\n {}'.format(pprint.pformat(cfg)))

    # set visualize logger
    logger_train = None
    logger_val = None
    if cfg.USE_TFBOARD:
        from utils.logger import Logger
        logger_dir = os.path.join(cfg.OUTPUT_DIR, 'tb_logger', 'train')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        logger_train = Logger(logger_dir)

        logger_dir = os.path.join(cfg.OUTPUT_DIR, 'tb_logger', 'val')
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        logger_val = Logger(logger_dir)

    # import dataset

    set_random_seed()

    logging.info(cfg.DATA.FILE)
    dataset_def = import_from_file(cfg.DATA.FILE)
    collate_fn = dataset_def.collate_fn
    dataset_def = dataset_def.ProviderDataset

    train_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        split=cfg.TRAIN.DATASET,
        one_hot=True,
        random_flip=True,
        random_shift=True,
        extend_from_det=cfg.DATA.EXTEND_FROM_DET)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn)

    val_dataset = dataset_def(
        cfg.DATA.NUM_SAMPLES,
        split=cfg.TEST.DATASET,
        one_hot=True,
        random_flip=False,
        random_shift=False,
        extend_from_det=cfg.DATA.EXTEND_FROM_DET)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn)


    logging.info('training: sample {} / batch {} '.format(len(train_dataset), len(train_loader)))
    logging.info('validation: sample {} / batch {} '.format(len(val_dataset), len(val_loader)))

    logging.info(cfg.MODEL.FILE)
    model_def = import_from_file(cfg.MODEL.FILE)
    model_def = model_def.PointNetDet

    input_channels = 3 if not cfg.DATA.WITH_EXTRA_FEAT else cfg.DATA.EXTRA_FEAT_DIM
    # NUM_VEC = 0 if cfg.DATA.CAR_ONLY else 3
    dataset_name = cfg.DATA.DATASET_NAME
    assert dataset_name in DATASET_INFO
    datset_category_info = DATASET_INFO[dataset_name]
    NUM_VEC = len(datset_category_info.CLASSES) # rgb category as extra feature vector
    NUM_CLASSES = cfg.MODEL.NUM_CLASSES

    model = model_def(input_channels, num_vec=NUM_VEC, num_classes=NUM_CLASSES)

    logging.info(pprint.pformat(model))

    if cfg.NUM_GPUS > 1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()

    parameters_size = 0
    for p in model.parameters():
        parameters_size += p.numel()

    logging.info('parameters: %d' % parameters_size)

    logging.info('using optimizer method {}'.format(cfg.TRAIN.OPTIMIZER))

    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR,
                               betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.BASE_LR,
                              momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        assert False, 'Not support now.'

    # miles = [math.ceil(num_epochs*3/8), math.ceil(num_epochs*6/8)]
    # assert isinstance(LR_SETP, list)

    LR_STEPS = cfg.TRAIN.LR_STEPS
    LR_DECAY = cfg.TRAIN.GAMMA

    if len(LR_STEPS) > 1:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_STEPS, gamma=LR_DECAY)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS[0], gamma=LR_DECAY)

    best_prec1 = 0
    best_epoch = 0
    start_epoch = 0
    # optionally resume from a checkpoint
    if cfg.RESUME:
        if os.path.isfile(cfg.TRAIN.WEIGHTS):
            checkpoint = torch.load(cfg.TRAIN.WEIGHTS)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.TRAIN.WEIGHTS, checkpoint['epoch']))
        else:
            logger.error("=> no checkpoint found at '{}'".format(cfg.TRAIN.WEIGHTS))

        # resume from other pretrained model
        if start_epoch == cfg.TRAIN.MAX_EPOCH:
            start_epoch = 0
            best_prec1 = 0
            best_epoch = 0

    if cfg.EVAL_MODE:
        validate(val_loader, model, start_epoch, logger_val)
        return

    MAX_EPOCH = cfg.TRAIN.MAX_EPOCH

    for n in range(start_epoch, MAX_EPOCH):

        train(train_loader, model, optimizer, lr_scheduler, n, logger_train)

        ious_gt = validate(val_loader, model, n, logger_val)

        prec1 = ious_gt

        is_best = False
        if prec1 > best_prec1:
            best_prec1 = prec1
            best_epoch = n + 1
            is_best = True
            logging.info('Best model {:04d}, Validation Accuracy {:.6f}'.format(best_epoch, best_prec1))

        save_data = {
            'epoch': n + 1,
            'state_dict': model.state_dict() if cfg.NUM_GPUS == 1 else model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_prec1': best_prec1,
            'best_epoch': best_epoch
        }
        if (n + 1) % 5 == 0 or (n + 1) == MAX_EPOCH:
            torch.save(save_data, os.path.join(cfg.OUTPUT_DIR, 'model_%04d.pth' % (n + 1)))

        if is_best:
            torch.save(save_data, os.path.join(cfg.OUTPUT_DIR, 'model_best.pth'))

        if (n + 1) == MAX_EPOCH:
            torch.save(save_data, os.path.join(cfg.OUTPUT_DIR, 'model_final.pth'))

    logging.info('Best model {:04d}, Validation Accuracy {:.6f}'.format(best_epoch, best_prec1))


if __name__ == '__main__':
    main()
