# Function: Compute the obj decent gradient using iCaRL
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = (1/T)*\sum_i{g_i}

import tensorflow as tf
import numpy as np

def ComputeGradient(gradients):
    beta = 1e-2 # as suggested in the paper
    d = []
    for k in range(len(gradients[0])): # for each layer
        g = 0
        A = tf.zeros([len(gradients), len(gradients)], dtype=tf.dtypes.float64)
        Phi = {}
        _Phi = {}
        for i in range(len(gradients)):
            g_i = gradients[i][k]
            for j in range(len(gradients)):
                if i == j:
                    pass
                else:
                    if (i,j) not in list(_Phi.keys()):
                        _Phi[(i,j)] = tf.constant(0., dtype=tf.float64)
                    flat_g_i = tf.reshape(gradients[i][k], [1, -1])
                    flat_g_j = tf.reshape(gradients[j][k], [1, -1])
                    Phi[(i,j)] = tf.squeeze(tf.linalg.matmul(flat_g_i, flat_g_j, transpose_b=True) / (tf.norm(flat_g_i, ord=2, keepdims=False)*tf.norm(flat_g_j, ord=2, keepdims=False)))
                    if Phi[(i,j)] < _Phi[(i,j)]:
                        g_i += gradients[j][k] * (tf.norm(g_i, ord=2, keepdims=False)/tf.norm(flat_g_i, ord=2, keepdims=False)+1e-7) * (_Phi[(i,j)]*tf.math.sqrt(1.-Phi[(i,j)]*Phi[(i,j)]) - _Phi[(i,j)]*tf.math.sqrt(1.-_Phi[(i,j)]*_Phi[(i,j)])) / tf.math.sqrt(1.-_Phi[(i,j)]*_Phi[(i,j)])
                    # update _Phi
                    _Phi[(i,j)] = (1-beta) * _Phi[(i,j)] + beta * Phi[(i,j)]
                    
            g += g_i
        d.append(g)
    return d



def ComputeGradient_v2(gradients):
    beta = 1e-2 # as suggested in the paper
    
    A = tf.zeros([len(gradients), len(gradients)], dtype=tf.dtypes.float64)
    Phi = {}
    _Phi = {}
    _g = []
    d = [tf.zeros(grad.shape, dtype=tf.dtypes.float64) for grad in gradients[0]]
    for i in range(len(gradients)):
        g_i = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        for j in range(len(gradients)):
            if i == j:
                pass
            else:
                g_j = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[j]], 0)
                if (i,j) not in list(_Phi.keys()):
                    _Phi[(i,j)] = tf.constant(0., dtype=tf.float64)
                flat_g_i = tf.reshape(g_i, [1, -1])
                flat_g_j = tf.reshape(g_j, [1, -1])
                Phi[(i,j)] = tf.squeeze(tf.linalg.matmul(flat_g_i, flat_g_j, transpose_b=True) / (tf.norm(flat_g_i, ord=2, keepdims=False)*tf.norm(flat_g_j, ord=2, keepdims=False)))
                if Phi[(i,j)] < _Phi[(i,j)]:
                    
                    for k in range(len(gradients[0])): # for each layer
                        g = 0
                        
                        d[k] +=  gradients[j][k] * (tf.norm(g_i, ord=2, keepdims=False)/tf.norm(flat_g_i, ord=2, keepdims=False)+1e-7) * (_Phi[(i,j)]*tf.math.sqrt(1.-Phi[(i,j)]*Phi[(i,j)]) - _Phi[(i,j)]*tf.math.sqrt(1.-_Phi[(i,j)]*_Phi[(i,j)])) / tf.math.sqrt(1.-_Phi[(i,j)]*_Phi[(i,j)])
                # update _Phi
                _Phi[(i,j)] = (1-beta) * _Phi[(i,j)] + beta * Phi[(i,j)]
        
    return d
