#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import tensorflow as tf
import numpy as np

def main():
    y = np.random.randint(2, size=[10])
    y_ = np.random.randint(2, size=[10])

    f1_score = tf.contrib.metrics.f1_score(y, y_)
    # f1_score_0 = f1_score[0]
    f1_score_0 = f1_score

    with tf.Session() as sess:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        f1 = sess.run(f1_score_0)
        print(f1)


if __name__ == '__main__':
    main()

