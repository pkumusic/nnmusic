#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Li.  yuezhanl@andrew.cmu.edu
import gym
from tqdm import tqdm
from logger import logger
import tensorflow as tf
import argparse
import numpy as np
import cv2
import random
from utils import show_image
from collections import deque, namedtuple
# Cur_observation, the action gonna take, the reward after taking the action, if it's done after the action
Memory = namedtuple('Memory', ['ob', 'action', 'reward', 'done']) # ob only contain 1 frame for space efficiency
Exp    = namedtuple('Exp', ['state', 'action', 'reward', 'done', 'next_state']) # state contain histories as training input

IMAGE_SIZE = (84,84)
REPLAY_MEMORY_SIZE = 1000  # 1000000
REPLAY_START_SIZE = 10
HISTORY_LENGTH = 4
MINIBATCH_SIZE = 32
UPDATE_FREQUENCY = 4

random.seed(0)

def train_dqn(env_name, gym_dir):
    env = gym.make(env_name)
    env.monitor.start(gym_dir, force=True)
    # Initialization
    ob, cur_obs = initialize_env(env)
    mem = deque(maxlen=REPLAY_MEMORY_SIZE)
    # Initialize memory with prediction function
    logger.info("Initialize replay memory")
    for i in tqdm(xrange(REPLAY_START_SIZE)):
        action = predict_action(cur_obs)
        new_ob, reward, done, info = env.step(action)
        new_ob = preprocess_state(new_ob)
        cur_obs.append(new_ob)
        mem.append(Memory(ob, action, reward, done))
        ob = new_ob
        if done:
            ob, cur_obs = initialize_env(env)

    while True:
        for i in xrange(UPDATE_FREQUENCY):
            action = predict_action(cur_obs)
            new_ob, reward, done, info = env.step(action)
            new_ob = preprocess_state(new_ob)
            cur_obs.append(new_ob)
            mem.append(Memory(ob, action, reward, done))
            ob = new_ob
            if done:
                ob, cur_obs = initialize_env(env)
        batch = sample(mem, MINIBATCH_SIZE, HISTORY_LENGTH)
        train_one_step(batch)

def initialize_env(env):
    """ Initialize a history queue and add first ob to the queue
    :param env: the environment needs to be reset
    :return:
    """
    cur_obs = deque(maxlen=HISTORY_LENGTH)
    ob = env.reset()
    ob = preprocess_state(ob)
    cur_obs.append(ob)
    return ob, cur_obs


def train_one_step(batch):
    pass


def preprocess_state(ob, debug=False):
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = cv2.resize(ob, IMAGE_SIZE)
    if debug:
        show_image(ob)
    return ob

def predict_action(input):
    # Use the last HISTORY_LENGTH frames to predict the action
    return 1

def sample(mem, batch_size, history_length):
    """
    :param mem:
    :param batch_size:
    :param history_length:
    :return: #[[Exp],]
    """
    data = []
    for i in xrange(batch_size):
        idx = random.randint(0, len(mem) - history_length - 1)
        samples = [mem[k] for k in xrange(idx, idx+history_length+1)]
        memory = samples[-2]
        action, reward, done = memory.action, memory.reward, memory.done
        def concat(idx):
            ans = np.array([x.ob for x in samples[idx:idx+history_length]])
            return np.dstack(ans)  # 84 * 84 * 4
        state = concat(0)
        next_state = concat(1)
        # Zero filling
        for j in xrange(history_length-1, -1, -1):
            if samples[j].done:
                state[:j+1,:,:] = 0
                next_state[:j,:,:] = 0
                break
        data.append(Exp(state, action, reward, done, next_state))
    return data

def gym_test(env_name):
    """
    Test on gym functions
    :return:
    """
    logger.info("Music")
    env = gym.make(env_name)
    env.monitor.start('tmp', force=True)
    observation = env.reset() # An image
    for _ in tqdm(range(1000)):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.monitor.close()

def tensorflow_test():
    # Linear Regression
    x_data = np.float32(np.random.rand(2, 100))
    y_data = np.dot([0.100, 0.200], x_data) + 0.300
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    for step in tqdm(xrange(1000)):
        #sess.run(train, feed_dict={x: [1,2,3], y_true:[0.3]})
        sess.run(train)
        if step % 20 == 0:
            print step, sess.run(W), sess.run(b)
            print sess.run(loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env', help='Gym environment name', default='MsPacman-v0')
    args = parser.parse_args()

    #gym_test(args.env)
    train_dqn(args.env, 'tmp')