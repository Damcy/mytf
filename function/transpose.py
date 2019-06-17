#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import tensorflow as tf
import numpy as np

x = np.random.random([2, 5, 2])
y = np.random.random([2, 5, 2])

print(x)
print(y)

attention_weights = tf.matmul(x, tf.transpose(y, [0, 2, 1]))
attentionsoft_a = tf.nn.softmax(attention_weights, dim=1)
# attentionsoft_tmp = tf.nn.softmax(tf.transpose(attention_weights))
attentionsoft_b = tf.nn.softmax(attention_weights, dim=2)

with tf.Session() as sess:
    print(sess.run(attention_weights))
    print('------')
    print(sess.run(attentionsoft_a))
    print('------')
    # print(sess.run(attentionsoft_tmp))
    print(sess.run(attentionsoft_b))
    print('------')
