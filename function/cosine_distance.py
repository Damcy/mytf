#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import numpy as np
import tensorflow as tf


def main():
    a = np.array([[1, 0, 0], [0, 0, 1], [1, 2, 0]], dtype=np.float32)
    b = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 0]], dtype=np.float32)

    # res = tf.losses.cosine_distance(a, b, axis=-1, reduction=tf.losses.Reduction.NONE)
    # res = tf.losses.cosine_distance(a, b, axis=-1)
    a_nor = tf.norm(a, axis=1) + 1e-6
    b_nor = tf.norm(b, axis=1) + 1e-6
    dot_product = tf.reduce_sum(tf.multiply(a, b), 1)
    res = dot_product / (a_nor * b_nor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a_nor))
        print(sess.run(b_nor))
        print(sess.run(dot_product))
        print(sess.run(res))


if __name__ == '__main__':
    main()

