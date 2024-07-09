# Function: Compute the obj decent gradient using Normalized Gradient Sum.
# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d = \sum_i {g_i/||g_i||}

import tensorflow as tf

def ComputeGradient(gradients):
    d = []
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += gradients[i][k]/gradients[i][k]/tf.norm(gradients[i][k], ord=2, keepdims=True)
        d.append(g)
    return d
