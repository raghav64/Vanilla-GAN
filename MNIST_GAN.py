#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:24:23 2017

@author: raghav
"""
# Tips and tricks followed from: https://github.com/soumith/ganhacks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib.gridspec as gridspec

X_train = pd.read_csv('train.csv').iloc[:,1:].values.astype(np.float32)
X_train = X_train / 255.
#plt.imshow(np.reshape(X_train[0],[28,28]), cmap='Greys_r')

#Read about xavier_init: https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape = size, stddev = xavier_stddev)

latent = 100
epsilon = 1e-7
batch_size = 128

                  ####### Creating batches ########
                  
q1=tf.RandomShuffleQueue(capacity=2000,min_after_dequeue=300,dtypes=tf.float32,shapes=[784])

enqueue_op1= q1.enqueue_many(vals=X_train)

numberOfThreads = 3
qr1 = tf.train.QueueRunner(q1, [enqueue_op1] * numberOfThreads)

tf.train.add_queue_runner(qr1)

sess=tf.Session()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
image_batch=q1.dequeue_many(batch_size)


                  ####### Creating Discriminator ########

X = tf.placeholder(dtype = tf.float32, shape = [None,784],name = 'Real_data_ip')

D_W1 = tf.get_variable('Dweight1',shape = [784,128],initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

D_b1 = tf.get_variable('Dbias1',shape=[128],dtype=tf.float32)

D_W2 = tf.get_variable('Dweight2',shape = [128,1],initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

D_b2 = tf.get_variable('Dbias2',shape=[1],dtype=tf.float32)

"""
D_W1 = tf.Variable(xavier_init([784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))
D_W2 = tf.Variable(xavier_init([128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
"""
theta_D = [D_W1, D_W2, D_b1, D_b2]

                 ######## Creating Generator ########

Z = tf.placeholder(dtype = tf.float32, shape = [None,latent],name = 'Latent_data_ip')

G_W1 = tf.get_variable('Gweight1',shape = [latent,256],initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

G_b1 = tf.get_variable('Gbias1',shape=[256],dtype=tf.float32)

G_W2 = tf.get_variable('Gweight2',shape = [256,784],initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

G_b2 = tf.get_variable('Gbias2',shape=[784],dtype=tf.float32)


"""
G_W1 = tf.Variable(xavier_init([latent, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))
"""
theta_G = [G_W1, G_W2, G_b1, G_b2]

# Can use dropout on both gen and disc
# Can use batch normalization: https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
def generator(z):
    G_p1 = tf.add(tf.matmul(z,G_W1), G_b1)
    #G_q1 = tf.nn.leaky_relu(G_p1)
    G_q1 = tf.nn.tanh(G_p1)
    G_p2 = tf.add(tf.matmul(G_q1,G_W2), G_b2)
    G_q2 = tf.nn.sigmoid(G_p2)
    
    return G_q2

def discriminator(x):
    D_p1 = tf.add(tf.matmul(x,D_W1), D_b1)
    #D_p1 = tf.nn.dropout(D_p1,0.8)
    #D_q1 = tf.nn.leaky_relu(D_p1)
    D_q1 = tf.nn.tanh(D_p1)
    D_p2 = tf.add(tf.matmul(D_q1,D_W2), D_b2)
    D_q2 = tf.nn.sigmoid(D_p2)
    
    return D_q2

#The sampling here is done uniformly.
#We can do it via normal(gaussian) distribution too.
def sample_Z(m, n):
    return np.random.normal(loc=0.0, scale=1.0, size=[m, n])


                 ###### Defining losses ######
Gen_image = generator(Z) 
D_real = tf.log(discriminator(X) + epsilon)
D_fake = tf.log(1. - discriminator(Gen_image) + epsilon)
G_fake = tf.log(discriminator(Gen_image) + epsilon)
Discriminator_loss = -tf.reduce_mean(D_real + D_fake)
Generator_loss = -tf.reduce_mean(G_fake) 

                 ###### Defining optmizers ######

#Read about optimisers
# https://towardsdatascience.com/types-of-optimization-algorithms-used-in-neural-networks-and-ways-to-optimize-gradient-95ae5d39529f

D_solver = tf.train.AdamOptimizer().minimize(Discriminator_loss,var_list = theta_D)
G_solver = tf.train.AdamOptimizer().minimize(Generator_loss,var_list = theta_G)


epochs = 40
no_of_batches = int (X_train.shape[0]/batch_size)
sess.run(tf.global_variables_initializer())
#sample_img = sess.run(Gen_image, feed_dict = {Z: sample_Z})
for i in range(epochs):
    TGLOSS = 0.
    TDLOSS = 0.
    for j in range(no_of_batches):
        _,dloss = sess.run([D_solver,Discriminator_loss],feed_dict = {X:sess.run(image_batch),Z:sample_Z(batch_size,latent)})
        _,gloss = sess.run([G_solver,Generator_loss],feed_dict = {Z:sample_Z(batch_size,latent)})
        
        TGLOSS = TGLOSS + gloss
        TDLOSS = TDLOSS + dloss
        
    print ('After Epoch ' + str(i+1) + ': Generator loss= '+ str(TGLOSS/no_of_batches) +' and Discriminator loss= ' + str(TDLOSS/no_of_batches))

print ('Real_Image: ')
plt.imshow(sess.run(image_batch)[0].reshape(28,28), cmap='gray')

print ('Generated_Image: ')
Gen1 = sess.run([Gen_image],feed_dict = {Z:sample_Z(1,latent)})
plt.imshow(np.reshape(Gen1,[28,28]), cmap='gray')

coord.request_stop()
coord.join(threads)

    
    

