#!/usr/bin/env python
# -*- encoding=utf-8 -*-

import os

import fasttext
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


# load model 
word2vec = []
words = []
total_word_cnt = 0
with open('wiki.zh.sample', 'r') as f:
    line = f.readline()
    for line in f:
        if len(line.strip()) == 0:
            continue
        content = line.strip().split(' ')
        if len(content) != 301:
            continue
        word = content[0]
        vec = list(map(float, content[1:]))
        total_word_cnt += 1
        words.append(word)
        word2vec.append(np.array(vec).astype('float32'))
        if total_word_cnt == 1000:
            break

dim = len(word2vec[0])
word2vec = np.array(word2vec)
print("dim is: ", dim)

# create a list of vectors


# setup a tensorflow session
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='embedding')
place = tf.placeholder(tf.float32, shape=(total_word_cnt, dim))
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: word2vec})

# write labels
with open('./log/metadata.tsv', 'w') as f:
    for word in words:
        f.write(word + '\n')

# create a tensorflow summary writer
summary_writer = tf.summary.FileWriter('log', sess.graph)
config = projector.ProjectorConfig()
embedding_conf = config.embeddings.add()
embedding_conf.tensor_name = 'embedding:0'
embedding_conf.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(summary_writer, config)

# save the model
saver = tf.train.Saver()
saver.save(sess, os.path.join('log', 'model.ckpt'))


