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
        for i in range(len(gradients)):
            g += (1/len(gradients))*gradients[i][k]
        d.append(g)
    return d
