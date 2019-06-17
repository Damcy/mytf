#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import numpy as np
import tensorflow as tf


def main():
    a = np.random.randint(100, size=[3, 4])
    b = tf.cast(tf.math.greater_equal(a, 50), tf.int32)
    c = tf.cast(tf.math.less(a, 50), tf.int32)
    print(a)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(c))


if __name__ == '__main__':
    main()

