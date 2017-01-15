#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 15:58:50 2017

@author: daniel
"""

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

class LinearMNIST(object):
    
    def __init__(self,train_dir='.',**config):

        
        self.config = {"image_size":28,
                 "num_classes": 10,
                 "batch_size": 100,
                 "eval_batch_size": 1,
                 "hidden1_units": 128,
                 "hidden2_units": 32,
                 "training_steps": 2000,
                 "learning_rate": 1e-4,
                 "type":"cnn",
                 "dropout":0.4
        }
        self.images_placeholder = tf.placeholder(tf.float32)                                       
        self.labels_placeholder = tf.placeholder(tf.int32)
        self.keep_prob = tf.placeholder(tf.float32)
        
        tf.add_to_collection("images", self.images_placeholder)
        tf.add_to_collection("labels", self.labels_placeholder)
        
        self.trained=False
        self.config["num_pixels"]=self.config["image_size"]*self.config["image_size"]
        self.train_dir=train_dir
        self.losses=[]
        
        self.data_sets = read_data_sets(train_dir, False)
       
        
        if self.config["type"]=="cnn":
            self.build_model_cnn()
        else:
            self.build_model_linear()
        
        tf.add_to_collection("logits", self.logits)
        self.saver = tf.train.Saver()
        self.training_setup()
        
        
        
    def build_model_linear(self):
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([self.config["num_pixels"], self.config["hidden1_units"]],
                                    stddev=1.0 / np.sqrt(float(self.config["hidden1_units"]))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.config["hidden1_units"]]),
                                 name='biases')
            hidden1 = tf.nn.relu(tf.matmul(self.images_placeholder, weights) + biases)
        # Hidden 2
        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([self.config["hidden1_units"], self.config["hidden2_units"]],
                                    stddev=1.0 / np.sqrt(float(self.config["hidden1_units"]))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.config["hidden2_units"]]),
                                 name='biases')
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        # Linear
        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([self.config["hidden2_units"], self.config["num_classes"]],
                                    stddev=1.0 / np.sqrt(float(self.config["hidden2_units"]))),
                name='weights')
            biases = tf.Variable(tf.zeros([self.config["num_classes"]]),
                                 name='biases')
        self.logits = tf.matmul(hidden2, weights) + biases
        
        return self.logits
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    def build_model_cnn(self):
        input_layer = tf.reshape(self.images_placeholder, [-1, 28, 28, 1])

        with tf.name_scope('conv1'):
            # 5x5 conv, 1 input, 32 outputs
            W_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = tf.nn.relu(self.conv2d(input_layer, W_conv1) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)
        
        with tf.name_scope('conv2'):
            # 5x5 conv, 32 inputs, 64 outputs
            W_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = self.max_pool_2x2(h_conv2)
        
        with tf.name_scope('dense1'):
            W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])
            
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
        
        with tf.name_scope('output'):
            W_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])
        
            self.logits=tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2)
        
        
        return self.logits
        
    def training_setup(self):
        self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels_placeholder, name='xentropy')
        self.loss = tf.reduce_mean(self.cross_entropy, name='xentropy_mean')
        
        tf.summary.scalar('loss', self.loss)
        
        # Create the gradient descent optimizer with the given learning rate.
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.optimizer = tf.train.AdamOptimizer(self.config["learning_rate"])
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        self.train_op = self.optimizer.minimize(self.loss)
        
        
        
        
    def train(self):
        correct_pred = tf.equal(tf.nn.top_k(self.logits), tf.nn.top_k(self.labels_placeholder))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        self.init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            # Run the Op to initialize the variables.
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.train_dir, sess.graph)
            checkpoint_file = os.path.join(self.train_dir, 'checkpoint')
            
            sess.run(self.init)
            # Start the training loop.
            for step in xrange(self.config["training_steps"]):
                # Read a batch of images and labels.
                images_feed, labels_feed = self.data_sets.train.next_batch(self.config["batch_size"])
        
                _, loss_value,summary,acc = sess.run([self.train_op, self.loss, merged,accuracy], 
                                                 feed_dict={self.images_placeholder: images_feed, self.labels_placeholder: labels_feed,
                                                            self.keep_prob: self.config["dropout"]})
                self.losses.append(loss_value)
                # Print out loss value.
                if step % 100 == 0:
                    print('Step %d: loss = %.2f' % (step, loss_value))
                    self.saver.save(sess, checkpoint_file, global_step=step)
                    train_writer.add_summary(summary, step)
            
            print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={self.images_placeholder: images_feed, self.labels_placeholder: labels_feed, self.keep_prob: 1.}))
            self.trained=True
        
        return loss_value
        
    def predict(self,img,eval_size=1):
        with tf.Session() as sess:
            logits = tf.get_collection("logits")[0]
            images_placeholder = tf.get_collection("images")[0]
            labels_placeholder = tf.get_collection("labels")[0]
        
            # Add an Op that chooses the top k predictions.
            eval_op = tf.nn.top_k(logits)
            
            images_feed, labels_feed = self.data_sets.train.next_batch(eval_size)
            
            prediction = sess.run(eval_op,
                              feed_dict={images_placeholder: images_feed,
                                         labels_placeholder: labels_feed})
        return prediction.indices[0][0]
        

if __name__ == '__main__':   
    tf.reset_default_graph()
    mnist=LinearMNIST(train_dir='./train')
    mnist.train()
        
        