#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Li.  yuezhanl@andrew.cmu.edu
import numpy as np
import tensorflow as tf

class DQN():

    def __init__(self):
        pass

    def get_DQN_prediction(self, state):
        """
        :return:
        """
        with tf.variable_scope('CNN1'):

            cnn1 = tf.nn.conv2d(state, W, [1,8,8,1], data_format='NCHW')


    def train_one_step(self, data):
        pass

    def load_model(self):
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    from train_dqn import Exp
    data = []
    state = np.random.random([4,84,84])
    next_state = np.random.random([4,84,84])
    data.append(Exp(state, 1, 1, False, next_state))

    dqn = DQN()
    dqn.train_one_step(data)
