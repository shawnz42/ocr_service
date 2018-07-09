#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : 'char_handler'.py 
Description:
Date: '2017/12/15' '14:05'
"""
import os
import numpy as np

from scripts.settings import CHAR_TOTAL_NUM


class HanziHandler:

    def __init__(self, char_len=None, start=0):
        self._start = start
        self._char_len = char_len

        current_dir = os.path.split(os.path.abspath(__file__))[0]
        char_file = "chars.txt"  # "small_chars.txt"   #
        with open(os.path.join(current_dir, char_file), "r", encoding='utf-8') as f:
            self._char_list = f.read().strip()

        self._char_set = set(self._char_list)

        self._char_num_mapping = None
        self._num_char_mapping = None

    @property
    def char_list(self):
        if self._char_len is None:
            return self._char_list[self._start:]
        else:
            return self._char_list[self._start: self._start + self._char_len]

    @property
    def char_num_mapping(self):
        if self._char_num_mapping is None:
            self._char_num_mapping = {char: num for num, char in enumerate(self.char_list)}
        return self._char_num_mapping

    @property
    def num_char_mapping(self):
        if self._num_char_mapping is None:
            self._num_char_mapping = {num: char for num, char in enumerate(self.char_list)}
        return self._num_char_mapping

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """
        Create a sparse representation of x.
        if sequences = [['abc'], ['a']]
        the output will be
        indexes = [[0,0],[0,1],[1,0]]
        values = [1,2,1]
        dense_shape = [2,2] (两个数字串，最大长度为2)

        :param sequences: a list of lists of type dtype where each element is a sequence
        :param dtype:
        :return:   A tuple with (indices, values, shape)
            indices:二维int64的矩阵，代表非0的坐标点
            values:二维tensor，代表indexes位置的数据值
            dense_shape:一维，代表稀疏矩阵的大小
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            for each in seq:
                values.append(self.char_num_mapping[each])

        # print("values", values)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def decode_sparse_tensor(self, sparse_tensor):
        """
        将sparse_tuple_from得到的返回值解码成原来的输入值
        输入：
            indexes = [[0,0],[0,1],[1,0]]
            values = [1,2,1]
            dense_shape = [2,2]
        输出
            sequences =[[1,2], [1]]
        :param sparse_tensor: A tuple with (indices, values, shape)
               indices:二维int64的矩阵，代表非0的坐标点
               values:二维tensor，代表indexes位置的数据值
               dense_shape:一维，代表稀疏矩阵的大小
        :return:
        """
        decoded_indexes = []
        current_i = 0
        current_seq = []

        for offset, (i, index) in enumerate(sparse_tensor[0]):
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = []
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

        # print("decoded_indexes", decoded_indexes)

        result = []
        for index in decoded_indexes:
            result.append(self.decode_a_seq(index, sparse_tensor))
        return result

    def decode_a_seq(self, indexes, spars_tensor):
        decoded = []
        for m in indexes:
            s = self.char_list[spars_tensor[1][m]]
            decoded.append(s)
        return ''.join(decoded)


hanzi_handler = HanziHandler(CHAR_TOTAL_NUM)


if __name__ == "__main__":
    chars = HanziHandler()
    print(chars.char_list)
    print(len(chars.char_list))



