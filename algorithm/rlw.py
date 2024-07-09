# Function: Compute the obj decent gradient using iCaRL
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = (1/T)*\sum_i{g_i}

import tensorflow as tf

def ComputeGradient(gradients):
    '''
    gradients:    gradients from each activate tasks
    '''
    d = []
    w = tf.nn.softmax(tf.random.uniform([len(gradients)], dtype=tf.float64))
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += w[i]*gradients[i][k]
        d.append(g)
    return d
