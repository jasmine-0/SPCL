# Function: Compute the obj decent gradient using GradNorm.
# Reference: 
# Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 
# GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. 
# In Proceedings ofthe 35th International Conference on Machine Learning, volume 80, pages 794–803, 2018.
# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d =



import tensorflow as tf

def ComputeGradient(gradients, previous_loss_ratios):
    '''
    gradients:    gradients from each activate tasks
    previous_loss_ratios:  l_{k-1}/l_{k-2} the previous loss ratio values for each tasks
    '''
    d = []
    w = tf.nn.softmax(previous_loss_ratios)
    w = tf.cast(w, dtype=tf.float64)
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += (w[i]/len(gradients))*gradients[i][k]
        d.append(g)
    return d

