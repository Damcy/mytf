#!/usr/bin/python
# -*- encoding=utf-8 -*-
# author: Ian
# e-mail: stmayue@gmail.com
# description: 

import tensorflow as tf


class ESIM:
    def init_param(self, param):
        self._seq_length = param.seq_length
        self._word_dim = param.word_dim
        self._num_class = param.num_class
        self._vocab_size = param.vocab_size
        self._emb_trainable = param.emb_trainable
        self._hidden_size = param.hidden_size
        self._lstm_unit_encode = param.lstm_unit_encode
        self._lstm_unit_composition = param.lstm_unit_encode * 4
        self._l2_reg_lambda = param.l2_reg_lambd

    def _full_connect_layer(self, linear_input, output_size, layer_scope, reuse, activation_fn=None):
        out = tf.contrib.layers.fully_connected(
            inputs=linear_input,
            weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            biases_initializer=tf.constant_initializer(0.1),
            num_outputs=output_size,
            scope='linear_layer_'+layer_scope, activation_fn=activation_fn,
            weights_regularizer=self.l2_regularizer,
            reuse=reuse
        )
        return out

    def _bi_lstm(self, sentence, sentence_len, name, unit, reuse=False):
        scope = 'BiLSTM_' + name
        with tf.name_scope(scope) and tf.variable_scope(scope, reuse=reuse):
            # forward
            cell_fw = tf.nn.rnn_cell.LSTMCell(unit, forget_bias=1.0, state_is_tuple=True,
                                              dtype=self._float_dtype)
            cell_fw_dropout = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=1-self.dropout_rate)
            # backward
            cell_bw = tf.nn.rnn_cell.LSTMCell(unit, forget_bias=1.0, state_is_tuple=True,
                                              dtype=self._float_dtype)
            cell_bw_dropout = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=1-self.dropout_rate)
            # BiLSTM
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw_dropout,
                                                                     cell_bw=cell_bw_dropout,
                                                                     inputs=sentence,
                                                                     sequence_length=sentence_len,
                                                                     dtype=self._float_dtype)
            fw_c, fw_h = output_states[0]
            bw_c, bw_h = output_states[1]
            hidden_concat = tf.concat([fw_h, bw_h], axis=-1)
            return hidden_concat

    @staticmethod
    def _composition_concat(rep1, rep2):
        rep1_avg = tf.reduce_mean(rep1, axis=1)
        rep1_max = tf.reduce_max(rep1, axis=1)
        rep2_avg = tf.reduce_mean(rep2, axis=1)
        rep2_max = tf.reduce_max(rep2, axis=1)
        return tf.concat([rep1_avg, rep1_max, rep2_avg, rep2_max], axis=-1)

    def __init__(self, FLAGS, init_emb=None):
        self._float_dtype = tf.float32
        self._int_dtype = tf.int32
        self.init_param(FLAGS)

        with tf.name_scope('ESIM') and tf.variable_scope('ESIM'):
            self.sentence1 = tf.placeholder(self._int_dtype, [None, self._seq_length], name='sentence1')
            self.sentence2 = tf.placeholder(self._int_dtype, [None, self._seq_length], name='sentence2')
            self.sentence1_len = tf.placeholder(self._int_dtype, [None], name='sentence1_len')
            self.sentence2_len = tf.placeholder(self._int_dtype, [None], name='sentence2_len')
            self.y = tf.placeholder(self._float_dtype, [None, self._num_class], name="input_y")
            self.dropout_rate = tf.placeholder(self._float_dtype, name="dropout_rate")

            if init_emb:
                self.emb = tf.get_variable(name='embedding', initializer=init_emb,
                                           trainable=self._emb_trainable)
            else:
                self.emb = tf.get_variable(name='embedding', shape=[self._vocab_size, self._word_dim],
                                           dtype=self._float_dtype,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                           trainable=self._emb_trainable)

            self.l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self._l2_reg_lambda)
            self.sentence1_emb = tf.nn.embedding_lookup(self.emb, self.sentence1)
            self.sentence2_emb = tf.nn.embedding_lookup(self.emb, self.sentence2)
            # encode
            bi_rep1 = self._bi_lstm(self.sentence1_emb, self.sentence1_len,
                                    'encode', self._lstm_unit_encode)
            bi_rep2 = self._bi_lstm(self.sentence2_emb, self.sentence2_len,
                                    'encode', self._lstm_unit_encode, reuse=True)
            # local inference modeling
            with tf.name_scope('local_inference_modeling'):
                with tf.name_scope('word_sim'):
                    attention_weight = tf.einsum('abd, acd->abc', bi_rep1, bi_rep2)
                    att_rep1 = tf.nn.softmax(attention_weight, dim=1)
                    att_rep2 = tf.nn.softmax(attention_weight, dim=2)
                    rep1_hat = tf.einsum('abd, acb->acd', bi_rep2, att_rep2)
                    rep2_hat = tf.einsum('abd, abc->acd', bi_rep1, att_rep1)

                compose1 = tf.concat([bi_rep1, rep1_hat, bi_rep1 - rep1_hat, bi_rep1 * rep1_hat],
                                         axis=-1)
                compose2 = tf.concat([bi_rep2, rep2_hat, bi_rep2 - rep2_hat, bi_rep2 * rep2_hat],
                                         axis=-1)

                rep1_compose = self._full_connect_layer(compose1, self._lstm_unit_composition,
                                                        'compose_projection', False, tf.nn.relu)
                rep2_compose = self._full_connect_layer(compose2, self._lstm_unit_composition,
                                                        'compose_projection', True, tf.nn.relu)

            # inference composition
            with tf.name_scope('inference_composition'):
                self.rep1 = self._bi_lstm(rep1_compose, self.sentence1_len,
                                          'composition', self._lstm_unit_composition)
                self.rep2 = self._bi_lstm(rep2_compose, self.sentence2_len,
                                          'composition', self._lstm_unit_composition, reuse=True)
                self.rep = self._composition_concat(self.rep1, self.rep2)


if __name__ == '__main__':
    import numpy as np
    import random

    class Param:
        def __init__(self):
            self.seq_length = 50
            self.word_dim = 128
            self.num_class = 2
            self._lstm_unit_encode = 128
            self.vocab_size = 1000
            self.emb_trainable = True
            self.l2_reg_lambda = 0.001
            self.lr = 0.001


    test1 = np.random.randint(1000, size=[5, 50])
    test2 = np.random.randint(1000, size=[5, 50])
    testy = np.random.randint(2, size=[5, 2])
