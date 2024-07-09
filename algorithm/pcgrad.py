# Function: Compute the obj decent gradient using PCGrad.
# Reference: 
# Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). 
# Gradient surgery for multi-task learning. 
# arXiv preprint arXiv:2001.06782.
# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d = 

import tensorflow as tf
import numpy as np

def ComputeGradient(gradients):
    gradients_flat = [tf.concat([tf.reshape(grad, [-1]) for grad in gradients[k]], 0) for k in range(len(gradients))]
    g_pc= []
    for i in range(len(gradients_flat)): # for each task
        g_list = list(range(len(gradients)))
        g_list.pop(i)
        sample_index = np.random.choice(g_list)
        sample_g = gradients[sample_index]
        sample_g_flat = gradients_flat[sample_index]
        dotp = tf.reduce_sum(tf.multiply(gradients_flat[i], sample_g_flat))
        ref_mag = tf.reduce_sum(tf.multiply(sample_g_flat, sample_g_flat))
        proj = [(1/len(gradients))*(gradients[i][k] - ((dotp/ ref_mag) * sample_g[k])) for k in range(len(gradients[0]))] # for each layer
        nonproj = [(1/len(gradients))*(gradients[i][k]) for k in range(len(gradients[0]))] # for each layer
        g_pc.append(tf.cond(tf.greater_equal(dotp, 0), lambda: nonproj, lambda: proj))
        # 释放
    return [tf.reduce_sum([g_pc[i][k] for i in range(len(gradients))], 0) for k in range(len(gradients[0]))]


def ComputeGradient_v2(gradients):
    gradients_flat = [tf.concat([tf.reshape(grad, [-1]) for grad in gradients[k]], 0) for k in range(len(gradients))]
    
    g_pc= []
    _t = []
    for i in range(len(gradients_flat)): # for each task
        g_list = list(range(len(gradients)))
        g_list.pop(i)
        sample_index = np.random.choice(g_list)
        sample_g = gradients[sample_index]
        sample_g_flat = gradients_flat[sample_index]
        dotp = tf.reduce_sum(tf.multiply(gradients_flat[i], sample_g_flat))
        ref_mag = tf.reduce_sum(tf.multiply(sample_g_flat, sample_g_flat))
        proj = [gradients[i][k] - ((dotp/ ref_mag) * sample_g[k]) for k in range(len(gradients[0]))] # for each layer
        _t.append(tf.cond(tf.greater_equal(dotp, 0), lambda: 1, lambda: dotp/ ref_mag))
        g_pc.append(tf.cond(tf.greater_equal(dotp, 0), lambda: gradients[i], lambda: proj))
        # 释放
    print(_t)
    # exit()
    return [tf.reduce_sum([g_pc[i][k] for i in range(len(gradients))], 0) for k in range(len(gradients[0]))]


def ComputeGradientFast(gradients):
    gradients_flat = [tf.concat([tf.reshape(grad, [-1]) for grad in gradients[k]], 0) for k in range(len(gradients))]
    g_pc= []
    for i in range(len(gradients_flat)): # for each task
        g_list = list(range(len(gradients)))
        g_list.pop(i)
        g_proj = gradients[i]
        for j in g_list:
            sample_g = gradients[j]
            sample_g_flat = gradients_flat[j]
            dotp = tf.reduce_sum(tf.multiply(gradients_flat[i], sample_g_flat))
            ref_mag = tf.reduce_sum(tf.multiply(sample_g_flat, sample_g_flat))
            proj = [g_proj[k] - ((dotp/ ref_mag) * sample_g[k]) for k in range(len(gradients[0]))] # for each layer
            g_proj = tf.cond(tf.greater_equal(dotp, 0), lambda: g_proj, lambda: proj)

        g_pc.append(g_proj)
        # 释放
    return [tf.reduce_sum([g_pc[i][k] for i in range(len(gradients))], 0) for k in range(len(gradients[0]))]

