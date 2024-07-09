# Function: Compute the obj decent gradient using GradNorm.
# Reference: 
# Zhao Chen, Vijay Badrinarayanan, Chen-Yu Lee, and Andrew Rabinovich. 
# GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. 
# In Proceedings ofthe 35th International Conference on Machine Learning, volume 80, pages 794–803, 2018.
# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d =
# Note the initial loss for memory is not reliable, we record the loss when each task was first stored.

import tensorflow as tf

def ComputeGradient(gradients, losses, init_losses, bid):
    '''
    gradients:    gradients from each activate tasks
    losses:       current loss values for each tasks
    init_losses:  init loss values for each tasks
    '''

    d = []
    n = len(gradients)
    # optimizer = tf.keras.optimizers.SGD(0.025)
    # print(init_losses)
    for k in range(len(gradients[0])): # for each layer
        g = 0 
        ng = [] # normalized gradient G_W
        loss_ratios = []
        w = tf.Variable(tf.ones([len(gradients)], dtype=tf.float64)/len(gradients), dtype=tf.float64)
        for i in range(len(gradients)):
            ng.append(tf.norm(gradients[i][k], ord=2, keepdims=False))
            # print(losses[i],init_losses[i])
            # exit()
            loss_ratios.append(losses[i]/init_losses[i])
       
        # q = tf.linalg.matmul(w, ng, transpose_a=True)
            
        # avg_ng = tf.reduce_mean(ng, axis=0)
        avg_loss_ratios = tf.reduce_mean(loss_ratios, axis=0)
        gn_loss = 0
        r = loss_ratios/avg_loss_ratios
        iter_count = 1
        l_prev = 0.
        while iter_count < 100:
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
                tape.watch(w)
                # print(w, ng)
                w_ng = tf.math.multiply(w, ng)
                # print(w_ng)
                avg_w_ng = tf.reduce_mean(w_ng)
                # print(avg_w_ng, r)
                l_norm = tf.nn.l2_loss(w_ng - avg_w_ng*tf.pow(r, 1.5))
                # print("Loss {}:".format(iter_count), l_norm)
            dw = tape.gradient(l_norm, w)
            w.assign_sub(0.001*dw) # update w
            # w.assign(w - tf.math.reduce_min(w))
            w.assign(w*(1./tf.reduce_sum(w))) # renormalize w
            if tf.math.abs(l_norm - l_prev) < 1e-5 and iter_count > 1:
                break
            else:
                l_prev = l_norm
            iter_count += 1
        # if bid > 53:
        #     print(w)
        # exit()
        for i in range(len(gradients)):
            g += w[i]*gradients[i][k]

        d.append(g)
    
    # print(losses)
    # print(init_losses)
    # print(loss_ratios)
    # print(r)
    # print('----------------')
    return d



def ComputeGradient_v2(gradients, losses, init_losses, bid):
    '''
    gradients:    gradients from each activate tasks
    losses:       current loss values for each tasks
    init_losses:  init loss values for each tasks
    '''

    
    ng = [] # normalized gradient G_W
    loss_ratios = []    
    with tf.compat.v1.variable_scope("gradnorm", reuse=True):
        w = tf.compat.v1.get_variable("W", [len(gradients)], initializer=tf.keras.initializers.constant(tf.ones([len(gradients)], dtype=tf.float64)/len(gradients)), dtype=tf.float64)
        # print(bid, w)
    for i in range(len(gradients)):
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        t = tf.norm(g_task_flat, ord=2, keepdims=True)
        ng.append(t)
        loss_ratios.append(losses[i]/init_losses[i])

    avg_loss_ratios = tf.reduce_mean(loss_ratios, axis=0)
    gn_loss = 0
    r = loss_ratios/avg_loss_ratios
    iter_count = 1
    l_prev = 0.
    while iter_count < 100:
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(w)
            w_ng = tf.math.multiply(w, ng)
            avg_w_ng = tf.reduce_mean(w_ng)
            l_norm = tf.nn.l2_loss(w_ng - avg_w_ng*tf.pow(r, 1.5))
            dw = tape.gradient(l_norm, w)
            w.assign_sub(0.001*dw) # update w
            w.assign(w*(1./tf.reduce_sum(w))) # renormalize w
        if tf.math.abs(l_norm - l_prev) < 1e-5 and iter_count > 1:
            break
        else:
            l_prev = l_norm
            iter_count += 1
    
    d = []
    n = len(gradients)
    for k in range(len(gradients[0])): # for each layer
        g = 0 
        for i in range(len(gradients)):
            g += w[i]*gradients[i][k]

        d.append(g)
    
    return d


# def ComputeGradient_v2(gradients, losses, loss_start, mem_loss_start):
#     d = []
#     L_i = []
#     if len(mem_loss_start) != 0:
#         L_i.append(losses[0]/mem_loss_start[-1])
#         for task_num in range(len(task_train)):
#             L_i.append(losses[task_num+1] / loss_start[task_train[task_num]])
#     else:
#         for task_num in range(len(task_train)):
#             L_i.append(losses[task_num] / loss_start[task_train[task_num]])
#         sum_L = sum([item for item in L_i])
#         r_i = L_i / sum_L
#         gradient = []
#         for k in range(len(gradients[0])): # for each layer
#             g = 0
#             for i in range(len(gradients)):
#                 g += r_i[i] * gradients[i][k]
#             d.append(g)
#     return d
