#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Li.  yuezhanl@andrew.cmu.edu
import numpy as np
import tensorflow as tf
NUM_ACTIONS = 9
GAMMA = 0.99
BATCH_SIZE = 32

class DQN():

    def __init__(self, state_size):
        self.state_size = state_size
        self.state  = tf.placeholder(tf.float32, (None,) + state_size, name='state')
        self.action = tf.placeholder(tf.int64, (None,), name='action')
        self.reward = tf.placeholder(tf.float32, (None,), name='reward')
        self.next_state = tf.placeholder(tf.float32, (None,) + state_size, name='next_state')
        self.done   = tf.placeholder(tf.bool, (None,), name='done')

        self.Qvalue = self.get_DQN_prediction(self.state)
        action_onehot = tf.one_hot(self.action, NUM_ACTIONS, 1.0, 0.0)
        pred_action_value = tf.reduce_mean(self.Qvalue * action_onehot, 1)
        with tf.variable_scope('target'):
            targetQvalue = self.get_DQN_prediction(self.next_state)
        best_v = tf.reduce_max(targetQvalue, 1)
        target = self.reward + (1.0 - tf.cast(self.done, tf.float32)) * GAMMA * tf.stop_gradient(best_v)
        self.cost = tf.truediv(tf.reduce_sum(tf.square(target - pred_action_value)),
                               tf.cast(BATCH_SIZE, tf.float32), name='cost')

    def get_DQN_prediction(self, state):
        """
        :return:
        """
        #TODO: Analyze output dimension for each layer
        with tf.variable_scope('CNN1'):
            W = tf.get_variable('W', [8,8,self.state_size[2],32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('b', [32], initializer=tf.constant_initializer())
            conv = tf.nn.conv2d(state, W, strides=[1,4,4,1], padding='SAME', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")

        with tf.variable_scope('CNN2'):
            W = tf.get_variable('W', [4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('b', [64], initializer=tf.constant_initializer())
            conv = tf.nn.conv2d(h, W, strides=[1,2,2,1], padding='SAME', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")

        with tf.variable_scope('CNN3'):
            W = tf.get_variable('W', [3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('b', [64], initializer=tf.constant_initializer())
            conv = tf.nn.conv2d(h, W, strides=[1,1,1,1], padding='SAME', name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")
            shape = h.get_shape().as_list()[1:]
            h_f = tf.reshape(h, [-1, np.prod(shape)], name='h_f')

        with tf.variable_scope('FC1'):
            input_size = 7744
            W = tf.get_variable('W', [input_size, NUM_ACTIONS], initializer=tf.uniform_unit_scaling_initializer(factor=1.43))
            b = tf.get_variable('b', [NUM_ACTIONS], initializer=tf.constant_initializer())
            output = tf.nn.relu(tf.nn.xw_plus_b(h_f, W, b), name='output')

        return tf.identity(output, name='Qvalue')

    def load_model(self):
        pass

    def save_model(self):
        pass


if __name__ == "__main__":
    from train_dqn import Exp
    #data = []
    state = np.random.random([3,4,84,84])
    next_state = np.random.random([3,4,84,84])
    #data.append(Exp(state, 1, 1, False, next_state))
    #print state

    dqn = DQN()
    dqn.train_one_step(state)
