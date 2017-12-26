# !/usr/bin/env python
# -*- encoding=utf-8 -*-
# author: ianma
# create at: 2017-12-11
# description: a basic CNN framework for sentence similarity

import json
import sys

import numpy as np
import tensorflow as tf


class ASCNN:
    def __init__(self, name, dtype=tf.float32):
        self.name = name
        self.dtype = dtype

    def init_params(self, params):
        self._q_max_len = params['query_max_length']
        self._a_max_len = params['answer_max_length']
        self._word_dim = params['word_dim']
        self._num_output = params['num_output']
        self._filter_sizes = params['filter_sizes']
        self._num_filters = params['num_filters']

    def build_model(self):
        with tf.name_scope(self.name):
            # input
            self.input_x1 = tf.placeholder(self.dtype, [None, self._q_max_len, self._word_dim], name='input_x1')
            self.input_x2 = tf.placeholder(self.dtype, [None, self._a_max_len, self._word_dim], name='input_x2')
            self.input_y = tf.placeholder(self.dtype, [None, self._num_output], name='input_y')
            self.dropout_keep_prob = tf.placeholder(self.dtype, name="dropout_keep_prob")

            # network
            with tf.name_scope('network'):
                self.input_x1_expanded = tf.expand_dims(self.input_x1, -1)
                self.input_x2_expanded = tf.expand_dims(self.input_x2, -1)
                # CNN
                pooled_outputs_x1 = []
                pooled_outputs_x2 = []
                for i, filter_size in enumerate(self._filter_sizes):
                    with tf.variable_scope('conv-maxpool-%s' % filter_size):
                        # conv
                        filter_shape = [filter_size, self._word_dim, 1, self._num_filters]
                        W = tf.get_variable(name='conv_W', shape=filter_shape, dtype=self.dtype,
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
                        b = tf.get_variable(name='conv_b', shape=[self._num_filters], dtype=self.dtype,
                                            initializer=tf.constant_initializer(0.1))
                        strides = [1, 1, 1, 1]
                        conv1 = tf.nn.conv2d(self.input_x1_expanded, W, strides, padding='VALID', name='conv1')
                        conv2 = tf.nn.conv2d(self.input_x2_expanded, W, strides, padding='VALID', name='conv2')
                        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='relu1')
                        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='relu2')
                        # max pooling
                        ksize_1 = [1, self._q_max_len - filter_size + 1, 1, 1]
                        ksize_2 = [1, self._a_max_len - filter_size + 1, 1, 1]
                        pooled1 = tf.nn.max_pool(h1, ksize=ksize_1, strides=strides, padding='VALID', name='pool1')
                        pooled2 = tf.nn.max_pool(h2, ksize=ksize_2, strides=strides, padding='VALID', name='pool2')
                        pooled_outputs_x1.append(pooled1)
                        pooled_outputs_x2.append(pooled2)

                num_filters_total = self._num_filters * len(self._filter_sizes)
                self.h_pool_1 = tf.concat(pooled_outputs_x1, 3)
                self.full_out_1 = tf.reshape(self.h_pool_1, [-1, num_filters_total])
                self.h_pool_2 = tf.concat(pooled_outputs_x2, 3)
                self.full_out_2 = tf.reshape(self.h_pool_2, [-1, num_filters_total])

                # add dropout
                with tf.variable_scope('dropout'):
                    self.out_1 = tf.nn.dropout(self.full_out_1, self.dropout_keep_prob)
                    self.out_2 = tf.nn.dropout(self.full_out_2, self.dropout_keep_prob)

                # distance
                self.mul_res = tf.multiply(self.out_1, self.out_2)
                self.out = tf.contrib.layers.fully_connected(
                        inputs=self.mul_res,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        biases_initializer=tf.constant_initializer(0.1),
                        num_outputs=self._num_output,
                        scope='linear_layer', activation_fn=None
                )
                self.out = tf.nn.softmax(self.out)

            # learning parameters
            self._lr = tf.get_variable(name='lr', shape=[], dtype=self.dtype, 
                                       initializer=tf.constant_initializer(0.0), trainable=False)
            self._new_lr = tf.get_variable(name='new_lr', shape=[], dtype=self.dtype, 
                                           initializer=tf.constant_initializer(0.0), trainable=False)
            self._lr_update = tf.assign(self._lr, self._new_lr)

            # loss
            with tf.name_scope('loss'):
                self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.input_y * tf.log(self.out), reduction_indices=[1]))
                self.train = tf.train.AdamOptimizer(self._lr).minimize(self.cross_entropy)


    def assign_lr(self, session, lr_val):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_val})

