# Function: Compute the obj decent gradient using AGEM
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf
import numpy as np
from . import min_norm_solvers

def ComputeGradient(gradients, curr_losses, mem_losses, gradnorm_mom):
        
    gs = []
    for i in range(len(gradients)): # for each task
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        gs.append(g_task_flat)
    tols = ComputeTol(curr_losses, mem_losses, gradnorm_mom)
    sol = min_norm_solvers.find_min_norm_element_with_tol(gs, tols)

    # if len(gs) > 2:
    #     print(3)

    d = []
    for k in range(len(gradients[0])):
        g = 0
        for i in range(len(gradients)): # for each task
            g += sol[i] * gradients[i][k] #/ len(gradients)
        d.append(g)
    return d

def ComputeTol(curr_losses, mem_losses, gradnorm_mom):   
    losses =  [mem_losses] + curr_losses if len(mem_losses) > 0 else curr_losses
    tols = []
    for k in range(len(losses)):
        assert len(losses[k]) > 0
        tols.append(gradnorm_mom[k])
        
    tols = np.array(tols, dtype=np.float64)
    tols = softmax(tols/5, 0) # Softmax Temperature 5
    return tols

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)
