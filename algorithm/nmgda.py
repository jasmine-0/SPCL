# Function: Compute the obj decent gradient using Normalized MGDA
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf
from . import min_norm_solvers

def ComputeGradient(gradients):
    d = []
    for k in range(len(gradients[0])):  # for each layer
        gs = []
        for i in range(len(gradients)):
            t = tf.norm(gradients[i][k], ord=2, keepdims=True)
            gs.append(gradients[i][k]/(t+1e-7))
        sol, min_norm = min_norm_solvers.find_min_norm_element(gs)
        g = 0
        for scale, _g in zip(sol, gs):
            g += scale * _g
        d.append(g)
    return d



def ComputeGradient_v2(gradients):
   
    gs = []
    for i in range(len(gradients)): # for each task
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        t = tf.norm(g_task_flat, ord=2, keepdims=True)
        gs.append(g_task_flat/(t+1e-7))
        
    sol, min_norm = min_norm_solvers.find_min_norm_element(gs)

    d = []
    for k in range(len(gradients[0])):
        g = 0
        for i in range(len(gradients)): # for each task
            g += sol[i] * gradients[i][k]
        d.append(g)
    return d
