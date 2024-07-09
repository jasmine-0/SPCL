import tensorflow as tf
import numpy as np
from scipy.optimize import minimize, minimize_scalar

def ComputeGradient(gradients, curr_losses, mem_losses, gradnorm_mom, flag):

    gs = []
    for i in range(len(gradients)):
        # print(len(gradients[i]))
        g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[i]], 0)
        g_task_2d = tf.expand_dims(g_task_flat, axis=-1)
        gs.append(g_task_2d)
    
    # compute G
    grads = tf.concat(gs, axis = 1)

    grads = compute_G6(grads)

    step = 0
    d = []
    for k in range(len(gradients[0])):
        g = 0
        for i in range(grads.shape[1]):
            temp_size = tf.size(gradients[i][k]).numpy()
            grad = grads[step:temp_size + step, i]
            g += tf.reshape(grad, gradients[i][k].shape) # * w[i]
            
            if i == (len(gradients) - 1):
                step += temp_size
        d.append(g)

    return d


# ||G−Q||2 F
def fit_loss(G, Q):
    return tf.reduce_sum(tf.square(G - Q))

# || d - 1 ||2 F 
def reg_loss(d, sig):
    return tf.reduce_sum(tf.square(d - sig))

# || Q^T Q - diag(d) ||2 F 
def penalty_loss(Q, d):
    return tf.reduce_sum(tf.square(tf.matmul(tf.transpose(Q), Q) - tf.linalg.diag(d)))

def compute_G6(grads):
    G = grads
    cov_grad_matrix = tf.matmul(tf.transpose(grads), grads)
    singulars, basis = tf.linalg.eigh(cov_grad_matrix)
    condition_number = tf.sqrt(tf.abs(singulars[-1])) / tf.sqrt(tf.abs(singulars[0]))
    tol = (tf.reduce_max(singulars) * max(cov_grad_matrix.shape[-2:]) * tf.keras.backend.epsilon())
    rank = tf.reduce_sum(tf.cast(singulars > tol, tf.int32))
    order = tf.argsort(singulars, axis=-1, direction='DESCENDING')
    singulars = tf.gather(singulars, order)[:rank]
    basis = tf.gather(basis, order, axis=1)[:, :rank]
    weights = basis
    sig = tf.sqrt(singulars[-1])

    weights = weights # * sig
    weights = weights / tf.reshape(tf.sqrt(singulars), (1, -1))
    weights = tf.matmul(weights, tf.transpose(basis))
    grads = tf.matmul(grads, weights)

    Q_k = tf.Variable(grads)
    D_k = tf.matmul(tf.transpose(grads), grads)

    d_k_diag = tf.Variable(tf.linalg.diag_part(D_k))
    lambda_val = 10
    sigma_val = 500
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9)
    
    loss_ = fit_loss(G, Q_k).numpy() + lambda_val * reg_loss(d_k_diag, 1).numpy() + sigma_val * penalty_loss(Q_k, d_k_diag).numpy()

    for i in range(100):
        with tf.GradientTape() as tape:
            loss = fit_loss(G, Q_k) + sigma_val * penalty_loss(Q_k, d_k_diag)
        g = tape.gradient(loss, [Q_k])
        optimizer.apply_gradients(zip(g, [Q_k]))

        GtG = tf.matmul(tf.transpose(Q_k), Q_k)
        d_k_diag = 1/(lambda_val + sigma_val) * (lambda_val * tf.linalg.diag_part(tf.eye(num_rows=tf.linalg.diag_part(GtG).shape[0], dtype=tf.float64))  + sigma_val * tf.linalg.diag_part(GtG))

        loss_now = fit_loss(G, Q_k).numpy() + lambda_val * reg_loss(d_k_diag, 1).numpy() + sigma_val * penalty_loss(Q_k, d_k_diag).numpy()
        
        if np.abs(loss_ - loss_now) < 1e-5:
            break
        else:
            loss_ = loss_now

    grads = Q_k  * tf.sqrt(singulars[-1])
    return grads

