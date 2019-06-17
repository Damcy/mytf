#!/usr/bin/env python
# -*- encoding=utf-8 -*-
# author: ianma
# email: stmayue@gmail.com

import abc

import tensorflow as tf


class ModelBase:
    def __init__(self, name, task_id=0, dtype=tf.float32):
        self.name = name
        self.task_id = task_id
        self.dtype = dtype

        # model general parameters
        self.sess = None
        self.learning_rate = None
        self.global_step = None
        self.trainable_params = []
        self.global_params = []
        self.run_options = None
        self.run_metadata = None

    @abc.abstractmethod
    def init_func(self):
        pass

    @abc.abstractmethod
    def build(selfs):
        pass
