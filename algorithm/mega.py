# Function: Compute the obj decent gradient using MEGA
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf

def ComputeGradient(gradients, curr_losses, mem_losses, mem_not_begin):
    
    d = []
    if mem_not_begin:
        for k in range(len(gradients[0])):
            g = 0
            for i in range(len(gradients)): # for each task
                g +=  (1/len(gradients))*gradients[i][k]
            d.append(g)
    else:
        for k in range(len(gradients[0])):
            g = 0
            for i in range(len(gradients)-1): # for each task
                g +=  (1/(len(gradients)-1))*gradients[i+1][k]
            d.append(g)

        gradients = [gradients[0], d]

        assert len(gradients)  in [1, 2], 'MEGA is the SLL without timeline'
        d = []
        epsilon = 0.1
        losses = [mem_losses[-1], curr_losses[0][-1]] 
        if curr_losses[0][-1] > 0.1:
            alpha = [1, losses[0]/losses[1]]
        else:
            alpha = [0, 1]
        for k in range(len(gradients[0])): # for each layer
            g = 0
            for i in range(len(gradients)):
                g += alpha[i]*gradients[i][k]
            d.append(g)
    return d
