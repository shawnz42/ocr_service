
import base64
from io import BytesIO

import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from scripts.settings import IMAGE_HEIGHT, NEED_WHITE


def str2gray(base64_str):
    """
    将base64编码转成灰度图片
    :param base64_str:
    :return:
    """
    im = imread(BytesIO(base64.b64decode(base64_str)))
    im_gray = rgb2gray(im)
    # im_gray = more_white(im_gray)
    return im_gray


def more_white(x):
    """
    把灰度图变得更白一些
    :param x:
    :return:
    """
    # return x * (3 - x) / 2
    return x * (1.27 - 0.27 * x)


def get_gray(im):
    im_gray = rgb2gray(im)
    if NEED_WHITE:
        im_gray = more_white(im_gray)  # ubuntu need this
    return im_gray


def resize_image(im_gray, reset_wide):
    """

    :param im_gray:
    :param reset_wide:
    :return:
    """
    wide = im_gray.shape[1]

    print("wide", wide)

    if wide > reset_wide:
        over_wide = wide - reset_wide
        left_over = int(over_wide / 2) + 1
        right_over = over_wide - left_over
        img = im_gray[:, left_over: wide - right_over]

        print("left_over", left_over, wide - right_over)
    else:
        blank_wide = reset_wide - wide
        left_blank_wide = int(blank_wide / 2)
        right_blank_wide = blank_wide - left_blank_wide

        l_blank_img = np.ones(shape=[IMAGE_HEIGHT, max(0, left_blank_wide)])
        r_blank_img = np.ones(shape=[IMAGE_HEIGHT, max(0, right_blank_wide)])

        img = np.hstack((l_blank_img, im_gray, r_blank_img))


    return img


# deprecated
def get_binary(im_gray, method=threshold_otsu):
    """
    图像二值化
    :param im:
    :return: 图像数组
    """
    im_bool = im_gray < method(im_gray)  # threshold_otsu(im_gray)
    # im_bool = threshold_adaptive(im_gray, block_size=3)
    return im_bool.astype(int)  # 此处０为白　１为黑


def get_hist(im_binary):
    """
    得到直方圖
    :param im_binary: 黑白图
    :return:
    """
    hist = np.sum(im_binary, axis=0)
    return hist



def get_first_nonzero(hist, hist_len=None):
    """

    :param hist:
    :param hist_len:
    :return:
    """
    if hist_len is None:
        hist_len = len(hist)

    for i in range(hist_len):
        if hist[i] != 0:
            return i


def get_last_nonzero(hist, hist_len=None):
    """

    :param hist:
    :param hist_len:
    :return:
    """
    if hist_len is None:
        hist_len = len(hist)

    for i in range(hist_len-1, -1, -1):
        if hist[i] != 0:
            return i


def horizontal_middle(im_binary, reset_wide):
    hist = get_hist(im_binary)
    hist_len = len(hist)
    first = get_first_nonzero(hist, hist_len)
    last = get_last_nonzero(hist, hist_len)

    nonzero_wide = last - first + 1

    if nonzero_wide <= reset_wide:
        blank_wide = reset_wide - nonzero_wide
        left_blank_wide = int(blank_wide / 2)
        right_blank_wide = blank_wide - left_blank_wide

        l_blank_img = np.zeros(shape=[IMAGE_HEIGHT, max(0, left_blank_wide)])
        r_blank_img = np.zeros(shape=[IMAGE_HEIGHT, max(0, right_blank_wide)])

        img = np.hstack((l_blank_img, im_binary[:, first:last+1], r_blank_img)).astype(int)
    else:
        over_wide = nonzero_wide - reset_wide
        left_over = int(over_wide / 2) + 1
        right_over = over_wide - left_over
        img = im_binary[:, left_over: nonzero_wide - right_over].astype(int)

        print("left_over", left_over, nonzero_wide - right_over)


    return img, nonzero_wide



if __name__ == "__main__":
    pass




