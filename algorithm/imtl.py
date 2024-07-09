# Function: Compute the obj decent gradient using iCaRL
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = (1/T)*\sum_i{g_i}

import tensorflow as tf

def ComputeGradient(gradients):
    d = []
    for k in range(len(gradients[0])): # for each layer
        g = 0
        ng = []
        D = [] # gradient differences
        U = [] # unit-norm gradient differences
        for i in range(len(gradients)):
            ng.append(gradients[i][k]/tf.norm(gradients[i][k], ord=2, keepdims=False))

        for i in range(len(gradients)-1):
            D.append(tf.reshape(gradients[0][k]-gradients[i+1][k], [-1]))
            U.append(tf.reshape(ng[0] - ng[i+1], [-1]))
            
        alpha_2_T = tf.matmul(tf.reshape(gradients[1][k], [1,-1]), tf.transpose(U))
        DU = tf.matmul(D, tf.transpose(U))
        alpha_2_T = tf.squeeze(tf.matmul(alpha_2_T, tf.linalg.inv(DU)))
        alpha = tf.nn.softmax(tf.concat([1. - tf.reduce_sum(alpha_2_T, keepdims=True), alpha_2_T], 0))
        
        for i in range(len(gradients)):
            g += alpha[i]*gradients[i][k]
        d.append(g)
    return d


def ComputeGradient_v2(gradients):

    ng = []
    D = [] # gradient differences
    U = [] # unit-norm gradient differences
    gf = []
    
    for i in range(len(gradients)):
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        gf.append(g_task_flat)
        t = tf.norm(g_task_flat, ord=2, keepdims=True)
        ng.append(g_task_flat/t)

    for i in range(len(gradients)-1):
        D.append(gf[0]-gf[i+1])
        U.append(ng[0] - ng[i+1])

    alpha_2_T = tf.matmul(tf.reshape(gf[1], [1,-1]), tf.transpose(U))
    DU = tf.matmul(D, tf.transpose(U))
    alpha_2_T = tf.squeeze(tf.matmul(alpha_2_T, tf.linalg.inv(DU)))
    alpha = tf.nn.softmax(tf.concat([1. - tf.reduce_sum(alpha_2_T, keepdims=True), alpha_2_T], 0))
        
    d = []
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += alpha[i]*gradients[i][k]
        d.append(g)
    return d
