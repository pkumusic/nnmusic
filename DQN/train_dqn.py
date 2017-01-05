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
from dqn import DQN
from utils import show_image
from collections import deque, namedtuple
import os
# Cur_observation, the action gonna take, the reward after taking the action, if it's done after the action
Memory = namedtuple('Memory', ['ob', 'action', 'reward', 'done']) # ob only contain 1 frame for space efficiency
Exp    = namedtuple('Exp', ['state', 'action', 'reward', 'done', 'next_state']) # state contain histories as training input

IMAGE_SIZE = (84,84)
REPLAY_MEMORY_SIZE = 1000  # 1000000
REPLAY_START_SIZE = 10
HISTORY_LENGTH = 4
MINIBATCH_SIZE = 32
UPDATE_FREQUENCY = 4
STATE_SIZE = IMAGE_SIZE + (HISTORY_LENGTH,)

random.seed(0)

def train_dqn(env_name, gym_dir, out_dir):

    # Initialize tf session and define training ops
    sess = tf.Session()
    with sess.as_default():
        dqn = DQN(STATE_SIZE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer   = tf.train.AdamOptimizer(1e-4) #learning rate
        grads_and_vars = optimizer.compute_gradients(dqn.cost)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        logger.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", dqn.cost)

        summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
        summary_writer = tf.train.SummaryWriter(out_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        def train_one_step(batch):
            state = np.array([e.state for e in batch]) / 255.0
            action = np.array([e.action for e in batch])
            reward = np.array([e.reward for e in batch])
            done = np.array([e.done for e in batch])
            next_state = np.array([e.next_state for e in batch]) / 255.0
            feed_dict = {
                dqn.state: state,
                dqn.action: action,
                dqn.reward: reward,
                dqn.done: done,
                dqn.next_state: next_state,
            }
            _, step, summaries, cost = sess.run(
                [train_op, global_step, summary_op, dqn.cost],
                feed_dict=feed_dict)

            #logger.info("step {}, cost {:g}".format(step, cost))
            summary_writer.add_summary(summaries, step)


        env = gym.make(env_name)
        #num_action = env.action_space.n
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
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 1000 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))

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
                state[:,:,:j+1] = 0
                next_state[:,:,:j] = 0
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
    train_dqn(args.env, 'tmp', 'tmp_model')