#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 20:02:53 2018

@author: amir
"""

#MNIST Dataset, Machine Learning's Hello World Project
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



#Where the data will actually be stored, in a placeholder array called "x"
x = tf.placeholder(tf.float32, [None, 784])


# =============================================================================
# Weight tensor and bias tensor. Weight tensor has shape of 784x10 so that when 
# multiplied with the image tensor "x", will output a 10-dimensional vector 
# corresponding to the ten possible digits. The bias tensor is also ten- 
# dimensional so that it can be added to the output. 
# =============================================================================
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


#Making sure the variables have the dimensions we want
print('Shape of W: ', W.shape)
print('Shape of b: ', b.shape)


# =============================================================================
# The actual model itself, called "softmax". Softmax is useful when attempting
# to categorize something amongst multipile possibilities. In this algorithm,
# x is multiplied with W, then added to b, and then softmax is applied. The
# softmax itself will convert the "evidence" of the inputs with the weights
# and biases into probablities that they fit into each category. 
# =============================================================================
y = tf.nn.softmax(tf.matmul(x, W) + b)


#Placeholder for the correct answers to the digit catgorizations
ans = tf.placeholder(tf.float32, [None, 10])

# =============================================================================
# Cross Entropy Function. I'll need to explore this further, but it essentially 
# allows us to determine the error or "cost" of our model. It multiplies the 
# log of the predicted probabilities by the actual probabilities, adds the 
# elements in the second dimension of y, then takes the mean.
# =============================================================================
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ans * tf.log(y), reduction_indices=[1]))


# =============================================================================
# Online it says that the function below is more numerically stable. I'm not 
# sure why or how. This is something else that needs to be explored further. 
# This will be unused for now.
# =============================================================================
'ce = tf.nn.softmax_cross_entropy_with_logits(logits = tf.matmul(x, W) + b)'


#The step in which we train the model, minimizing our Cross Entropy function.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Creates a session in which to train the model
sess = tf.InteractiveSession()

#Intializes all of the variables we have in the session
tf.global_variables_initializer().run()

#Loops batches of 100 a thousand times to train the model
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, ans: batch_ys})


# =============================================================================
# Does our model work? This will evaluate whether the top value in both tensors
# are equal. This will come out as a list of booleans.
# =============================================================================
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(ans,1))

#Turns the booleans into floats 0 & 1 and takes the mean 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Prints our accuracy. Does the model work?
print('Accuracy = ', sess.run(accuracy, feed_dict={x: mnist.test.images, ans: mnist.test.labels}))
