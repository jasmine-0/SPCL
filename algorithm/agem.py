# Function: Compute the obj decent gradient using AGEM
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf

def ComputeGradient(gradients, mem_not_begin):
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

        assert len(gradients)  in [1, 2], 'A-GEM is the SLL without timeline'
        gradients_flat = [tf.concat([tf.reshape(grad, [-1]) for grad in gradients[k]], 0) for k in range(2)]
        dotp = tf.reduce_sum(tf.multiply(gradients_flat[1], gradients_flat[0]))
        ref_mag = tf.reduce_sum(tf.multiply(gradients_flat[0], gradients_flat[0]))
        proj = [gradients[1][k] - ((dotp/ ref_mag) * gradients[0][k]) for k in range(len(gradients[0]))]
        d = tf.cond(tf.greater_equal(dotp, 0), lambda: gradients[1], lambda: proj)
    return d
