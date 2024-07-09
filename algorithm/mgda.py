# Function: Compute the obj decent gradient using AGEM
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf
from . import min_norm_solvers


def ComputeGradient(gradients, bid):
   
    d = []

    for k in range(len(gradients[0])): # for each layer
        gs = [gradients[i][k] for i in range(len(gradients))]
        sol, min_norm = min_norm_solvers.find_min_norm_element(gs)
        # print(bid, sol)
        # if bid > 106:
        #     print(bid, sol)
        # if bid > 110:
        #     exit()
        g = 0
        for scale, _g in zip(sol, gs):
            g += scale*_g
        d.append(g)
    return d

def ComputeGradient_v2(gradients, bid):
   
    gs = []
    for i in range(len(gradients)): # for each task
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        gs.append(g_task_flat)
    sol, min_norm = min_norm_solvers.find_min_norm_element(gs)

    d = []
    for k in range(len(gradients[0])):
        g = 0
        for i in range(len(gradients)): # for each task
            g += sol[i] * gradients[i][k]
        d.append(g)
    return d


def ComputeGradient_fast(gradients):
   
    d = []
    if len(gradients) == 2:
        gradient = []
        for g0, g1 in zip(gradients[0], gradients[1]):
            g0_flat = tf.reshape(g0, [-1])
            g1_flat = tf.reshape(g1, [-1])

            g12 = tf.reduce_sum(tf.multiply(g0_flat, g1_flat))
            g11 = tf.reduce_sum(tf.multiply(g0_flat, g0_flat))
            g22 = tf.reduce_sum(tf.multiply(g1_flat, g1_flat))

            if g12>=g11:
                alpha = 1
            elif g12>=g22:
                alpha = 0
            else:
                alpha = tf.reduce_sum(tf.multiply((g1_flat-g0_flat), g1_flat))/((tf.norm(g1_flat-g0_flat,ord=2))**2)
            d.append(alpha*g0 + (1-alpha)*g1)
    else:
        for k in range(len(gradients[0])): # for each layer
            gs = [gradients[i][k] for i in range(len(gradients))]
            sol, min_norm = min_norm_solvers.find_min_norm_element(gs)
            g = 0
            for scale, _g in zip(sol, gs):
                g += scale*_g
            d.append(g)
    return d
