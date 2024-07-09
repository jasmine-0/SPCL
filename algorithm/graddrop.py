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
        P = 0.
        ng_sum = 0.
        g_sum = 0.
        for i in range(len(gradients)):
            g_sum += gradients[i][k]
            ng_sum += tf.norm(gradients[i][k], ord=2, keepdims=False)

        P = (1. + g_sum/ng_sum) / 2.
        U = tf.random.uniform(P.shape, dtype=tf.float64)
        
        for i in range(len(gradients)):
            M_pos = tf.math.multiply(
                tf.cast(tf.math.greater(P, U), dtype=tf.float64),
                tf.cast(tf.math.greater(gradients[i][k], 0.), dtype=tf.float64))
            M_neg = tf.math.multiply(
                tf.cast(tf.math.less(P, U), dtype=tf.float64),
                tf.cast(tf.math.less(gradients[i][k], 0.), dtype=tf.float64))
            M = M_pos + M_neg
            g += tf.math.multiply(M, gradients[i][k])
        d.append(g)
    return d


def ComputeGradient_v2(gradients):
    d = []
    
    ng_sum = 0.
    for i in range(len(gradients)):
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        t = tf.norm(g_task_flat, ord=2, keepdims=True)
        ng_sum += tf.norm(t, ord=2, keepdims=False)
        
    for k in range(len(gradients[0])): # for each layer
        g = 0
        P = 0.
        ng_sum = 0.
        g_sum = 0.
        for i in range(len(gradients)):
            g_sum += gradients[i][k]

        P = (1. + g_sum/ng_sum) / 2.
        U = tf.random.uniform(P.shape, dtype=tf.float64)
        
        for i in range(len(gradients)):
            M_pos = tf.math.multiply(
                tf.cast(tf.math.greater(P, U), dtype=tf.float64),
                tf.cast(tf.math.greater(gradients[i][k], 0.), dtype=tf.float64))
            M_neg = tf.math.multiply(
                tf.cast(tf.math.less(P, U), dtype=tf.float64),
                tf.cast(tf.math.less(gradients[i][k], 0.), dtype=tf.float64))
            M = M_pos + M_neg
            g += tf.math.multiply(M, gradients[i][k])
        d.append(g)
    return d
