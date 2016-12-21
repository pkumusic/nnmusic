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
from utils import show_image
from collections import deque

IMAGE_SIZE = (84,84)
REPLAY_MEMORY_SIZE = 1000  # 1000000
HISTORY_LENGTH = 4


def dqn(env_name, gym_dir):
    env = gym.make(env_name)
    env.monitor.start(gym_dir, force=True)
    ob = env.reset()
    mem = deque(maxlen=REPLAY_MEMORY_SIZE)
    while True:
        ob = preprocess_state(ob, debug=False)
        mem.append(ob)
        action = predict_action(mem[-4:])
        ob, reward, done, info = env.step(action)
        if done:
            exit()

def preprocess_state(ob, debug=False):
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = cv2.resize(ob, IMAGE_SIZE)
    if debug:
        show_image(ob)
    return ob

def predict_action(input):

    return 1


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
    dqn(args.env, 'tmp')