# Function: Compute the obj decent gradient using Minimal Asymmemic Distance
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = (1/T)*\sum_i{g_i}

import tensorflow as tf
import numpy as np

def ComputeGradient(gradients):
    d = []

    gs = []
    gs_norm = []
    for i in range(len(gradients)): # for each task
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        gs.append(g_task_flat)
        gs_norm.append(tf.norm(g_task_flat, ord=2, keepdims=False))

    gs_mean = tf.reduce_mean(gs, 0)    
    # gs_score = 2.*tf.nn.softmax(tf.squeeze(tf.linalg.matmul(gs, tf.expand_dims(gs_mean,-1))))
    angles = []
    gs_mean_norm = tf.norm(gs_mean, ord=2, keepdims=False)
    for i in range(len(gradients)): # for each task
        cos = tf.linalg.matmul([gs[i]], tf.expand_dims(gs_mean,-1))/(gs_mean_norm*gs_norm[i])
        angles.append(tf.squeeze(cos))
    angles = np.array(angles, dtype=np.float64)
    gs_score = tf.nn.softmax(angles/gs_norm)
    # gs_score /= tf.reduce_mean(gs_score**2)
    # _t = tf.reduce_mean(gs_score**2)
    # print(gs_score,_t)
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += gs_score[i]*gradients[i][k]
        d.append(g)
    return d
