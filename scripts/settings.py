#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : 'settings'.py 
Description:
Author: 'zhengyang' 
Date: '2017/11/30' '10:30'
"""


import platform

PROJECT_NAME = "cnn-ctc-ocr"


# 训练参数
# LEARNING_RATE, ACCURACY = 1e-3, 0.85
LEARNING_RATE, ACCURACY = 0.000125, 0.97
# LEARNING_RATE, ACCURACY = 1e-5, 0.95   # 未达到


TRAIN_BATCH_SIZE = 64

# 图片参数
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 512


# 字库参数　
CHAR_TOTAL_NUM = None # 10
CHAR_START = 0  # 3145

# 开发测试参数
DEV_BATCH_SIZE = 64
TEST_BATCH_SIZE = 128

SALT_NOISE_NUMS = [0, 1, 2, 3] # sampe image salt noise num

LOG_LEVEL = 'INFO'


if platform.system() == 'Windows':
    NEED_WHITE = False
    LOG_FILE = r'D:\log\{}.log'.format(PROJECT_NAME)
    TRAIN_DATA_PATH = r"D:\log\train"
    DEV_DATA_PATH = r"D:\log\dev"
    TEST_DATA_PATH = r"D:\log\test"
    MODELS_PATH = r"D:\pycharm2\models"

elif platform.system() == 'Linux':
    NEED_WHITE = True
    LOG_FILE = '/var/log/proj-log/{0}.log'.format(PROJECT_NAME)
    TRAIN_DATA_PATH = r"/usr/local/data/cnn-ctc-ocr/train"
    DEV_DATA_PATH = r"/usr/local/data/cnn-ctc-ocr/dev"
    TEST_DATA_PATH = r"/usr/local/data/cnn-ctc-ocr/test"
    MODELS_PATH = r"/usr/local/data/cnn-ctc-ocr/models"

else:
    LOG_FILE = '/var/log/{0}.log'.format(PROJECT_NAME)
    TRAIN_DATA_PATH = r"/usr/local/cnn-single-ocr-data/train"
    DEV_DATA_PATH = r"/usr/local/cnn-single-ocr-data/dev"
    TEST_DATA_PATH = r"/usr/local/cnn-single-ocr-data/test"
    MODELS_PATH = r"/usr/local/ccnn-ctc-ocr/models"



