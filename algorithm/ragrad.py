# Function: Compute the obj decent gradient using relative asymmetric gradient RAGrad.
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  

import tensorflow as tf
import numpy as np
from . import min_norm_solvers

lr = 50
lr2 = 10
# lr2 = 50
iters = 5
loss_factor = .1

# cifar
# lr = 50
# iters = 5
# loss_factor = 0.1

# emnist
# lr = 50
# iters = 5
# loss_factor = .1

def ComputeGradient(gradients, bath_id):
    # print(bath_id)

    # 1. Flat gradients
    gs = []
    for i in range(len(gradients)): # for each task
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        gs.append(g_task_flat)

    # 2. Compute the weight
    weights = distangle_optimize_weight_with_distance(gs)
    # weights = direct_optimize_weight_with_distance(gs)

    # 3. Obtain the final gradient
    d = []
    for k in range(len(gradients[0])): # for each layer
        g = 0
        for i in range(len(gradients)):
            g += weights[i]*gradients[i][k]
        d.append(g)
    return d


def distangle_optimize_weight_with_distance(grads):
    """optimize the distance for all independent grad
    grads: List, all flat gradients
    """
    # n_grads = [g / tf.norm(g, ord=2) for g in grads]
    A = tf.Variable((1/(len(grads)-1))*tf.ones([len(grads), len(grads)], dtype=tf.float64), trainable=True)
    
    # compute the masks
    A_diag_mask = np.identity(len(grads))
    A_mask = 1 - A_diag_mask
    if len(grads) > 2:
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        optimizer = tf.keras.optimizers.SGD(lr2)
    # optimizer = tf.keras.optimizers.Adam(lr)
    
    # update non-diagonal
    for _ in range(iters):
        with tf.GradientTape() as tape:
            # compute total loss
            # G_combine = tf.matmul(A*A_mask, grads) # (A_m)^TG
            if len(grads) > 2:
                masked_A = masked_softmax(A, A_mask)    
            else:
                masked_A = tf.minimum(tf.maximum(A*A_mask, 0), 1)
            G_combine = tf.matmul(masked_A, grads) # (A_m)^TG
            ##########################################################
            # RA distance
            # rad_mask = loss_factor*relative_asymmetric_distance(G_combine, grads) # rad((A_m)^TG, G)
            # rad_mask = rad_mask*rad_mask
            # Euclidean distance
            # rad_mask = loss_factor*euclidean_distance(G_combine, grads) # rad((A_m)^TG, G)
            # rad_mask = rad_mask*rad_mask
            # Cosine distance
            # rad_mask = loss_factor*cosine_distance(G_combine, grads) # rad((A_m)^TG, G)
            # Normalized Euclidean distance
            rad_mask = loss_factor*norm_euclidean_distance(G_combine, grads) # rad((A_m)^TG, G)
            # KL_div
            # rad_mask = loss_factor*KL_divergence(G_combine, grads) # rad((A_m)^TG, G)
            ##########################################################
            # print(rad_mask)
        _g_A_mask = tape.gradient(rad_mask, A) #*A_mask
        # _g_A_diag = tape.gradient(rad_diag, masked_A)
        _g_A = _g_A_mask  # TODO
        optimizer.apply_gradients([(_g_A, A)])
    # print('-----------------------')
    # update the diagonal
    _g_A_diag = min_norm_solvers.find_min_norm_element_independent(grads)
    if len(grads) > 2:
        A = masked_softmax(1*A, A_mask) + tf.linalg.diag(_g_A_diag)
        return tf.reduce_sum(A, 0) / (len(grads)+1)
    else: 
        A = A*masked_A + tf.linalg.diag(_g_A_diag)
        # return tf.nn.softmax(1 * tf.reduce_sum(A, 0) / (len(grads)+1), -1)
        return tf.nn.softmax(1 * tf.reduce_sum(A, 0), -1)
    # A = tf.maximum( A*A_mask, 0 ) # + tf.linalg.diag(_g_A_diag)
    # A = masked_softmax(A, A_mask) 
    # print(A)
    # return tf.nn.softmax(tf.reduce_sum(A, 0) / (len(grads)+1), -1)
    # return tf.reduce_sum(A, 0) / (len(grads))
    # return tf.nn.softmax(tf.reduce_sum(A, 0), -1)
    # return tf.reduce_sum(A, -1) / len(grads)


def distangle_optimize_weight_with_distance_2(grads):
    """optimize the distance for all independent grad
    grads: List, all flat gradients
    """
    n_grads = [g / tf.norm(g, ord=2) for g in grads]
    A = tf.Variable((1/len(grads))*tf.ones([len(grads), len(grads)-1], dtype=tf.float64), trainable=True)
    # A_diag = tf.Variable((1/len(grads))*tf.ones([1, len(grads)], dtype=tf.float64), trainable=True)
    
    # compute the masks
    A_diag_mask = np.identity(len(grads))
    A_mask = 1 - A_diag_mask
    # optimizer = tf.keras.optimizers.SGD(lr,)
    optimizer = tf.keras.optimizers.Adam(lr)
    
    # update non-diagonal
    for _ in range(10):
        # print(A)
        with tf.GradientTape() as tape:
            # compute total loss
            # masked_A = masked_softmax(A, A_mask)
            # A = tf.nn.softmax(A, -1)
            softmax_A = tf.nn.softmax(A, axis=-1)
            # print(softmax_A)
            G_combine = []
            for i in range(len(grads)):
                G_combine.append(tf.matmul(tf.expand_dims(softmax_A[i], 0), grads[:i]+grads[i+1:]))
            # G_combine = tf.matmul(masked_A, grads) # (A_m)^TG
            G_combine = tf.concat(G_combine, 0)
            ##########################################################
            # RA distance
            # rad_mask = relative_asymmetric_distance(G_combine, grads) # rad((A_m)^TG, G)
            # Euclidean distance
            rad_mask = euclidean_distance(G_combine, grads) # rad((A_m)^TG, G)
            # Cosine distance
            # rad_mask = cosine_distance(G_combine, grads) # rad((A_m)^TG, G)
            ##########################################################
            # print(rad_mask)
        _g_A_mask = tape.gradient(rad_mask, A) #*A_mask
        # _g_A_diag = tape.gradient(rad_diag, masked_A)
        _g_A = _g_A_mask  # TODO
        optimizer.apply_gradients([(_g_A, A)])
    # print('-----------------------')
    # update the diagonal
    _g_A_diag = min_norm_solvers.find_min_norm_element_independent(n_grads)
    # A = masked_softmax(A, A_mask) + tf.linalg.diag(_g_A_diag)
    # A_broadcast = tf.linalg.diag(_g_A_diag)
    softmax_A = tf.nn.softmax(A, axis=-1)
    A_broadcast = np.diag(_g_A_diag)
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i == j:
                continue
            elif i < j:
                A_broadcast[i,j] = softmax_A[i,j-1]
            else:
                A_broadcast[i,j] = softmax_A[i,j]

    # optimizer.apply_gradients([(tf.linalg.diag(_g_A_diag), A)])
    # print(A_broadcast) 
    return tf.reduce_sum(A_broadcast, 0) / (len(grads)+1)


def direct_optimize_weight_with_distance(grads):
    """optimize the distance for all independent grad
    grads: List, all flat gradients
    """
    # grads = grads[::-1]
    n_grads = [g / tf.norm(g, ord=2) for g in grads]
    A = tf.Variable((1/len(grads))*tf.ones([1, len(grads)], dtype=tf.float64), trainable=True)
    optimizer = tf.keras.optimizers.SGD(lr)
    # optimizer = tf.keras.optimizers.Adam(lr)

    for _ in range(iters):
        with tf.GradientTape() as tape:
            # compute total loss
            softmax_A = tf.nn.softmax(A, axis=-1)
            G_combine = tf.matmul(softmax_A, grads) # (A_m)^TG
            ##########################################################
            #*************** RA distance ***************
            loss = loss_factor*relative_asymmetric_distance(G_combine, grads) # rad((A_m)^TG, G)
            #*************** Euclidean distance *******************
            # loss = euclidean_distance(G_combine, grads) # rad((A_m)^TG, G)
            # *************** Cosine distance *************** 
            # loss = cosine_distance(G_combine, grads) # rad((A_m)^TG, G)
            ##########################################################
        _g_A = tape.gradient(loss, A)
        optimizer.apply_gradients([(_g_A, A)])
    # _g_A_diag = min_norm_solvers.find_min_norm_element_independent(grads)
    # return tf.nn.softmax((tf.squeeze(A) + _g_A_diag), -1)
    # return tf.nn.softmax((tf.squeeze(A) + _g_A_diag) / len(grads), -1)
    # return (tf.squeeze(tf.nn.softmax(A, axis=-1)) + _g_A_diag) / len(grads)
    return tf.squeeze(tf.nn.softmax(A, axis=-1))

def relative_asymmetric_distance(x, y):
    """The proposed distance
    rad(x,y) -> Int >=0
    """
    dist = tf.math.reduce_euclidean_norm( x - y, axis=-1 ) # task_num*1
    dist = dist / (dist + tf.math.reduce_euclidean_norm( y, axis=-1 ))
    dist = tf.reduce_mean(dist)
    return dist

def euclidean_distance(x, y):
    """Euclidean distance
    eud(x,y) -> Int >=0
    """
    dist = tf.reduce_mean(tf.math.reduce_euclidean_norm( x - y, axis=-1 ))
    return dist

def cosine_distance(x, y):
    """Cosine distance
    cosd(x,y) -> Int >=0
    """
    lens = len(y)
    dist = 0
    if x.shape[0] == 1 :
        x = tf.tile(x, [len(y), 1])
    for i in range(lens):
        dist += tf.reduce_sum( x[i]* y[i] ) / ( tf.norm(x[i], ord=2) * tf.norm(y[i], ord=2) )
    dist /= lens
    return dist

def norm_euclidean_distance(x, y):
    """Normalized Euclidean distance
    neud(x,y) -> Int >=0
    """
    x_norm = tf.reduce_mean(tf.math.reduce_euclidean_norm( x , axis=-1 ))
    y_norm = tf.reduce_mean(tf.math.reduce_euclidean_norm( y , axis=-1 ))
    dist = tf.reduce_mean(tf.math.reduce_euclidean_norm( x - y, axis=-1 ))/ (x_norm+y_norm+1e-5)
    return dist

def KL_divergence(x, y):
    """Normalized Euclidean distance
    neud(x,y) -> Int >=0
    """
    dist = tf.reduce_mean(y*tf.math.log(x/y))
    return dist


# class EuDist(tf.keras.losses.Loss):
#     def __init__(self):
#         super(EuDist, self).__init__()
 
#     def call(self, y_true, y_pred):        
#         # loss = 1-tf.norm(y_true, ord=2)**2/(tf.norm(y_true, ord=2)**2+tf.norm(y_true-y_pred, ord=2)**2)
#         # loss = 1-1/(1+tf.norm((y_true-y_pred)/y_true, ord=2)**2)
#         # loss = tf.norm(y_true-y_pred, ord=2)**2
#         return tf.math.reduce_euclidean_norm( y_true - y_pred )

def masked_softmax(scores, mask):
    scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
    exp_scores = tf.exp(scores)
    # exp_scores *= mask # TODO Element-wise multiply
    exp_scores = tf.math.multiply(exp_scores, mask)
    exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
    return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])+1e-7) 