# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Detectron config system.

This file specifies default config options for Detectron. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.

Most tools in the tools directory take a --cfg option to specify an override
file and an optional list of override (key, value) pairs:
 - See tools/{train,test}_net.py for example code that uses merge_cfg_from_file
 - See configs/*/*.yaml for example config files

Detectron supports a lot of different model types, each of which has a lot of
different options. The result is a HUGE set of configuration options.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import yaml
import io
import os
import copy
from ast import literal_eval
import numpy as np

# from past.builtins import basestring
from six import string_types

from configs.collections import AttrDict

__C = AttrDict()

cfg = __C

# Training options

__C.TRAIN = AttrDict()

__C.TRAIN.WEIGHTS = ''

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.MAX_EPOCH = 200

__C.TRAIN.OPTIMIZER = 'adam'

__C.TRAIN.BASE_LR = 0.001

__C.TRAIN.MIN_LR = 1e-5

__C.TRAIN.LR_POLICY = 'step'

__C.TRAIN.GAMMA = 0.1

__C.TRAIN.LR_STEPS = [20]

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0000

# train, val, trainval
__C.TRAIN.DATASET = 'train'


# Model options
__C.MODEL = AttrDict()

__C.MODEL.FILE = ''

__C.MODEL.NUM_CLASSES = 2


# Test options
__C.TEST = AttrDict()

__C.TEST.WEIGHTS = ''

__C.TEST.BATCH_SIZE = 32

# nms / top
__C.TEST.METHOD = 'top'

# NMS overlap threshold
__C.TEST.THRESH = 0.1

# val, test
__C.TEST.DATASET = 'val'


# Data options

__C.DATA = AttrDict()

__C.DATA.FILE = ''

__C.DATA.DATA_ROOT = 'kitti'

# intensity for kitti, rgb for sunrgbd
__C.DATA.WITH_EXTRA_FEAT = True

__C.DATA.NUM_SAMPLES = 1024

__C.DATA.NUM_SAMPLES_DET = 512

__C.DATA.CAR_ONLY = True

__C.DATA.PEOPLE_ONLY = False

__C.DATA.RTC = True

__C.DATA.NUM_HEADING_BIN = 12

# stride of sliding frustum
__C.DATA.STRIDE = (0.25, 0.5, 1.0, 2.0)

# half of the height of frustum
# see the ops/query_depth_point/query_depth_point_cuda_kernel.cu for details, we measure the distance from the frustum centroid
__C.DATA.HEIGHT_HALF = (0.25, 0.5, 1.0, 2.0)

__C.DATA.EXTEND_FROM_DET = False


# Loss options
__C.LOSS = AttrDict()

__C.LOSS.BOX_LOSS_WEIGHT = 1.

__C.LOSS.CORNER_LOSS_WEIGHT = 10.

__C.LOSS.HEAD_REG_WEIGHT = 20.

__C.LOSS.SIZE_REG_WEIGHT = 20.


# MISC options
__C.RESUME = False

__C.NUM_GPUS = 1

__C.OUTPUT_DIR = '/tmp'

__C.SAVE_SUB_DIR = 'test'

__C.OVER_WRITE_TEST_FILE = ''

__C.FROM_RGB_DET = False

__C.NUM_WORKERS = 4

__C.USE_TFBOARD = False

__C.EVAL_MODE = False

# evaluation iou threshold, car 0.7, people 0.5
__C.IOU_THRESH = 0.7

__C.disp = 50


def assert_and_infer_cfg(cache_urls=True, make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """

    if make_immutable:
        cfg.immutable(True)


def get_output_dir(datasets, training=True):
    """Get the output directory determined by the current global config."""
    assert isinstance(datasets, (tuple, list, string_types)), \
        'datasets argument must be of type tuple, list or string'
    is_string = isinstance(datasets, string_types)
    dataset_name = datasets if is_string else ':'.join(datasets)
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset-name>/<model-type>/
    outdir = os.path.join(__C.OUTPUT_DIR, tag, dataset_name, __C.MODEL.TYPE)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def load_cfg(cfg_to_load):
    """Wrapper around yaml.load used for maintaining backward compatibility"""
    assert isinstance(cfg_to_load, (io.IOBase, string_types)), \
        'Expected {} or {} got {}'.format(io.IOBase, string_types, type(cfg_to_load))
    if isinstance(cfg_to_load, io.IOBase):
        cfg_to_load = ''.join(cfg_to_load.readlines())
    return yaml.load(cfg_to_load)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):

        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, string_types):

        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
