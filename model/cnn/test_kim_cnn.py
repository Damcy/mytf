#!/usr/bin/env python
# -*- encoding=utf -*-
# author: ianma
# create at: 2017-12-11
# description: test answer selection model

import numpy as np
import tensorflow as tf

import kim_cnn as model


def main():
    # define a CNN
    # :param num_output: the number of output class
    # :param filter_sizes: n-gram types, slide window
    # :param num_filters: cnn filters
    param = {"query_max_length": 20, "answer_max_length": 8, 
             "word_dim": 32, "num_output": 2, "filter_sizes": [3, 5], 
             "num_filters": 200}

    answer_selection_net = model.ASCNN("answer_select")
    answer_selection_net.init_params(param)
    answer_selection_net.build_model()
    # input and output
    batch_size = 20
    dropout_keep_pro = 0.8
    x1 = np.random.randn(batch_size, param['query_max_length'], param['word_dim'])
    x2 = np.random.randn(batch_size, param['answer_max_length'], param['word_dim'])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    out1, out2, out = sess.run([answer_selection_net.out_1, answer_selection_net.out_2, answer_selection_net.out], 
                               feed_dict={answer_selection_net.input_x1: x1,
                                          answer_selection_net.input_x2: x2,
                                        answer_selection_net.dropout_keep_prob: dropout_keep_pro})

    print(out)
    # true labels
    labels = [np.random.randint(0, 2) for x in range(0, 20)]
    one_hot_index = np.arange(len(labels)) * 2 + labels
    true_y = np.zeros((len(labels), 2))
    true_y.flat[one_hot_index] = 1
    # training
    answer_selection_net.assign_lr(sess, 0.0001)
    train = sess.run([answer_selection_net.train], 
                     feed_dict={answer_selection_net.input_x1: x1,
                                answer_selection_net.input_x2: x2,
                                answer_selection_net.dropout_keep_prob: dropout_keep_pro,
                                answer_selection_net.input_y: true_y})
    # predict
    out1, out2, out = sess.run([answer_selection_net.out_1, answer_selection_net.out_2, answer_selection_net.out], 
                               feed_dict={answer_selection_net.input_x1: x1,
                                          answer_selection_net.input_x2: x2,
                                          answer_selection_net.dropout_keep_prob: dropout_keep_pro})
    print(out)


if __name__ == '__main__':
    main()

