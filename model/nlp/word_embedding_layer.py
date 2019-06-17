#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian Ma
# e-mail: stmayue@gmail.com
# description: 

import os
import sys

import gensim
import numpy as np
import tensorflow as tf

_ENCODE = "utf-8"
_DECODE = "utf-8"


class WordEmbeddingLayer:
    def __init__(self, word_list, embedding_dim, add_unknown=False,
                 word_vec_file=None, vec_type=None, name="embedding", dtype="float32"):
        with tf.name_scope("embedding"):
            w2v = self.load_word_vec(word_list, embedding_dim, word_vec_file, vec_type,
                                     add_unknown, dtype)


    def load_word_vec(self, word_list, embedding_dim, word_vec_file, vec_type, add_unknown, dtype):
        # position 0 is for unk_vec if add_unknown is True
        vocab_size = len(word_list)
        if add_unknown:
            vocab_size += 1
            word_list = ['unk'] + word_list

        w2v = np.random.uniform(-1.0, 1.0, (vocab_size, embedding_dim))
        w2v = np.cast[dtype](w2v)
        if word_vec_file is not None:
            if vec_type == "word2vec_format":
                w2v_model = gensim.models.Word2Vec.load_word2vec_format(word_vec_file, binary=True,
                                                                        encoding=_ENCODE,
                                                                        unicode_errors="ignore")
                for i in range(0, vocab_size):
                    word = word_list[i]
                    if word in w2v_model:
                        w2v[i] = w2v_model[word]
            elif vec_type == "fasttext_format":
                pass

        return w2v