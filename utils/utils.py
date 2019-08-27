import os
import sys

import torch
import importlib
import shutil
import logging

from configs.config import cfg


def import_from_file(def_file, cover=True):
    folder = os.path.dirname(def_file)
    file = os.path.basename(def_file)
    # save setting file
    save_file = os.path.join(cfg.OUTPUT_DIR, file)
    if not os.path.exists(save_file) or cover:
        shutil.copy(def_file, os.path.join(cfg.OUTPUT_DIR, file))

    # import setting
    path = os.path.abspath(folder)
    sys.path.append(path)
    model_file = importlib.import_module(file[:-3])
    sys.path.remove(path)
    return model_file


def get_accuracy(output, target, topk=(1,), ignore=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    assert output.shape[0] == target.shape[0]
    if ignore is not None:
        assert isinstance(ignore, int)
        keep = (target != ignore).nonzero().view(-1)
        output = output[keep]
        target = target[keep]

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 1e-14

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_logger(log_file):
    # FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    FORMAT = '%(message)s'
    logging.root.handlers = []
    logging.basicConfig(filename=log_file, format=FORMAT)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    return logger
