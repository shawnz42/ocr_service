#!/usr/bin/env python
# -*- coding: utf-8 -*-  


"""
@desc: 
@author: 
@time: 17-12-29 下午5:15
"""

import tensorflow as tf

from scripts.utils.cnn_tools import conv_layer, max_pool_2x2
from scripts.settings import LEARNING_RATE
from scripts.data.hanzi_handler import hanzi_handler


graph = tf.Graph()

with graph.as_default():
    with tf.variable_scope("cnn"):
        MAX_STEP_NUM = 10000

        num_classes = len(hanzi_handler.char_list) + 1 + 1  # char_list + blank + ctc blank

        TRAIN_KEEP_PROB = 1
        TEST_KEEP_PROB = 1

        MOMENTUM = 0.9

        keep_prob = tf.placeholder(tf.float32)

        # 定义ctc_loss需要的稀疏矩阵
        targets = tf.sparse_placeholder(tf.int32)

        # 1维向量 序列长度 [batch_size,]
        seq_len = tf.placeholder(tf.int32, [None])

        DNN_HIDDEN_NUM = 256

        MAX_WIDE_SHRINK = 4

        x = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        filter_num = [64, 64, 128, 128, 256, 256, 256]
        conv1 = conv_layer(x, shape=[3, 3, 1, filter_num[0]])
        conv2 = conv_layer(conv1, shape=[3, 3, filter_num[0], filter_num[1]])

        conv2_pool = max_pool_2x2(conv2)
        conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)

        conv3 = conv_layer(conv2_drop, shape=[3, 3, filter_num[1], filter_num[2]])
        conv4 = conv_layer(conv3, shape=[3, 3, filter_num[2], filter_num[3]])

        conv4_pool = max_pool_2x2(conv4)
        conv4_drop = tf.nn.dropout(conv4_pool, keep_prob=keep_prob)

        conv5 = conv_layer(conv4_drop, shape=[3, 3, filter_num[3], filter_num[4]])
        conv6 = conv_layer(conv5, shape=[3, 3, filter_num[4], filter_num[5]])
        conv7 = conv_layer(conv6, shape=[3, 3, filter_num[5], filter_num[6]])

        conv7_pool = max_pool_2x2(conv7)

        batch_s = tf.shape(conv7_pool)[0]

    with tf.variable_scope("dnn"):
        conv_reshape = tf.reshape(conv7_pool, [-1, DNN_HIDDEN_NUM])

        W = tf.Variable(tf.truncated_normal([DNN_HIDDEN_NUM, num_classes], stddev=0.1), name="W")

        b = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")

        logits = tf.matmul(conv_reshape, W) + b

        # [batch_size,max_timesteps,num_classes]
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        # 转置矩阵，第0和第1列互换位置=>[max_timesteps,batch_size,num_classes]
        logits = tf.transpose(logits, (1, 0, 2))

    # tragets是一个稀疏矩阵
    loss = tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len,
                          preprocess_collapse_repeated=False, ctc_merge_repeated=False)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE,momentum=MOMENTUM).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)  # 先用adam
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # 前面说的划分块之后找每块的类属概率分布，ctc_beam_search_decoder方法,是每次找最大的K个概率分布
    # 还有一种贪心策略是只找概率最大那个，也就是K=1的情况ctc_ greedy_decoder
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

    saver = tf.train.Saver()



