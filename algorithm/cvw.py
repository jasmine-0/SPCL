# Function: Compute the obj decent gradient using CV-weighting.
# Reference: 

# Rick Groenendijk and Sezer Karaoglu and Theo Gevers and Thomas Mensink. 
# Multi-Loss Weighting with Coefficient of Variations. 
# Arxiv 2009.01717.

# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d = 

import tensorflow as tf
import numpy as np

def ComputeGradient(gradients, losses, losses_continuous_mean):
    '''
    gradients:    gradients from each activate tasks
    losses:       current loss values for each tasks
    fulltime_losses:  all loss values for each tasks, a dict
    '''

    d = []
    _losses_continuous_mean = []
    for l in losses_continuous_mean:
        if l >= 0:
            _losses_continuous_mean.append(l)
    loss_std = np.std(_losses_continuous_mean)
    alpha = [loss_std/losses_continuous_mean[i] for i in range(len(gradients))]
    
    for i in range(len(gradients)):
        if losses_continuous_mean[i] >= 0:
            alpha[i] = 1/len(gradients)
            
    sum_alpha = np.sum(alpha)
    alpha = [a/sum_alpha for a in alpha]
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += alpha[i]*gradients[i][k]
        d.append(g)
    return d
