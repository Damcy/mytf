#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: siamese TextCNN

import tensorflow as tf


class SiameseTextCNN(object):
    def __init__(self, FLAGS, init_emb=None):
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        self._init_param(FLAGS)

        with tf.name_scope('siamese'):
            self.sentence1 = tf.placeholder(self._int_dtype, [None, self._seq_length], name='sentence1')
            self.sentence2 = tf.placeholder(self._int_dtype, [None, self._seq_length], name='sentence2')
            self.y = tf.placeholder(self._float_dtype, [None, self._num_class], name="input_y")
            self.dropout_rate = tf.placeholder(self._float_dtype, name="dropout_rate")

            if init_emb is None:
                self.emb = tf.get_variable(name='embedding',
                                           shape=[self._vocab_size, self._word_dim],
                                           dtype=self._float_dtype,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1))
            else:
                self.emb = tf.get_variable(name='embedding', initializer=init_emb)

            self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_lambda)
            self.sentence1_emb = tf.nn.embedding_lookup(self.emb, self.sentence1)
            self.sentence2_emb = tf.nn.embedding_lookup(self.emb, self.sentence2)

            self.cnn1 = self._text_cnn(self.sentence1_emb)
            self.cnn2 = self._text_cnn(self.sentence2_emb, reuse=True)

            self.sim_output = self._similarity_concat(self.cnn1, self.cnn2)
            self.scores = self._linear_layer(self.sim_output)
            self.softmax = tf.nn.softmax(self.scores, name='softmax')

        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                                            logits=self.scores)
            self.loss = tf.reduce_mean(self.cross_entropy) + tf.losses.get_regularization_loss()
            # self.train = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

        with tf.name_scope('performance'):
            self.prediction = tf.argmax(self.scores, 1)
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   self._float_dtype),
                                           name='accuracy')
            self.f1 = tf.contrib.metrics.f1_score(tf.argmax(self.y, 1), self.prediction)[0]

    def _init_param(self, param):
        self._seq_length = param.seq_length
        self._word_dim = param.word_dim
        self._num_class = param.num_class
        self._filter_sizes = param.filter_size
        self._num_filters = param.num_filters
        self._vocab_size = param.vocab_size
        self._l2_reg_lambda = param.l2_reg_lambda

    def _linear_layer(self, linear_input):
        out = tf.contrib.layers.fully_connected(
            inputs=linear_input,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
            biases_initializer=tf.constant_initializer(0.1),
            num_outputs=self._num_class,
            scope='linear_layer',
            weights_regularizer=self.l2_regularizer
        )
        return out

    def _similarity_concat(self, output1, output2):
        with tf.name_scope('similarity_concat'):
            sim_output = tf.concat([output1, output2], 1, name='concat')
        return sim_output

    def _text_cnn(self, sentence, reuse=False):
        with tf.name_scope('TextCNN') and tf.variable_scope('TextNN', reuse=reuse):
            sentence_expanded = tf.expand_dims(sentence, -1)
            pooled_outputs = []
            for i, filter_size in enumerate(self._filter_sizes):
                with tf.name_scope('conv-maxpool-%s' % filter_size)\
                     and tf.variable_scope('conv-maxpool-%s' % filter_size):
                    # conv
                    filter_shape = [filter_size, self._word_dim, 1, self._num_filters]
                    W = tf.get_variable(name='W', shape=filter_shape, dtype=self._float_dtype,
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable(name='b', shape=[self._num_filters], dtype=self._float_dtype,
                                        initializer=tf.constant_initializer(0.1))
                    strides = [1, 1, 1, 1]
                    conv = tf.nn.conv2d(sentence_expanded, W, strides, padding='VALID', name='conv')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                    ksize = [1, self._seq_length - filter_size + 1, 1, 1]
                    pooled = tf.nn.max_pool(h, ksize=ksize, strides=strides,
                                            padding='VALID', name='pool')
                    pooled_outputs.append(pooled)
            num_filters_total = self._num_filters * len(self._filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            full_out = tf.reshape(h_pool, [-1, num_filters_total])
            with tf.name_scope('dropout'):
                out = tf.nn.dropout(full_out, rate=self.dropout_rate)
        return out


if __name__ == '__main__':
    import numpy as np
    import random
    class Param:
        def __init__(self):
            self.seq_length = 50
            self.word_dim = 128
            self.num_class = 2
            self.filter_size = [2, 3, 4, 5]
            self.num_filters = 256
            self.vocab_size = 1000
            self.l2_reg_lambda = 0.001
            self.lr = 0.001
    test1 = np.random.randint(1000, size=[5, 50])
    test2 = np.random.randint(1000, size=[5, 50])
    testy = np.random.randint(2, size=[5, 2])
    with tf.Session() as sess:
        param = Param()
        net = SiameseTextCNN(param)
        sess.run(tf.global_variables_initializer())
        tmp = [v.name for v in tf.trainable_variables()]
        print(tmp)
        res1, res2 = sess.run([net.cnn1, net.cnn2], feed_dict={net.sentence1: test1,
                                                               net.sentence2: test2,
                                                               net.y: testy,
                                                               net.dropout_rate: 0})
        print(res1)
        print(res2)
        # _ = sess.run(net.train, feed_dict={net.sentence1: test1,
        #                                    net.sentence2: test2,
        #                                    net.y: testy,
        #                                    net.dropout_rate: 0.1})
        # res = sess.run(net.cnn2  , feed_dict={net.sentence1: test1,
        #                                       net.sentence2: test2,
        #                                       net.y: testy,
        #                                       net.dropout_rate: 0})
        # print(res)
