# Function: Compute the obj decent gradient using GCM Optimization.
# Code by Fan Lyu : fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation: d = 

import tensorflow as tf

def ComputeGradient(gradients):

    d = []
    # initialize gradient
    # avggrad
    inital_gradient = []
    for k in range(len(gradients[0])):  # for each layer
        g = 0
        for i in range(len(gradients)):
            g += (1 / len(gradients)) * gradients[i][k]  # /losses[i]
            # g += gradients[i][k]  # /losses[i]
        inital_gradient.append(g)

        # Reshape update
    gradient_flat = tf.concat([tf.reshape(grad, [-1]) for grad in inital_gradient], 0)
    delta = tf.Variable(tf.zeros_like(gradient_flat), trainable=True)
    # update gradient
    # optimizer = tf.keras.optimizers.Adam(0.0001)
    optimizer = tf.keras.optimizers.SGD(0.1, decay=1e-6, momentum=0.9, nesterov=True)
    iter_num = 10
    for i in range(iter_num):
        with tf.GradientTape() as tape:
            g_loss = 0
            for i, g_task in enumerate(gradients):
                g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in g_task], 0)
                g_loss += self.gradsim_object(g_task_flat, gradient_flat + delta)
                # g_loss += (losses[i]/(tf.reduce_max(losses)+1e-7))*self.gradsim_object(g_task_flat, gradient_flat+delta)
            g_loss /= len(gradients)
        # print(len(gradients), i, g_loss)
        _g = tape.gradient(g_loss, delta)
        optimizer.apply_gradients([(_g, delta)])
    # loss_div = tf.math.log(tf.reduce_max(losses)/(tf.reduce_min(losses)+1e-7))
    gradient_flat += delta

    # Convert the flat projected gradient vector into a list
    offset = 0
    for v in self.model.trainable_variables:
        shape = v.get_shape()
        v_params = 1
        for dim in shape:
            v_params *= dim
        d.append(tf.reshape(gradient_flat[offset:offset + v_params], shape))
        offset += v_params



    # task_train.sort()
    # loss_values = []
    # side_change = []
    # rate = 0
    #
    #
    # if if_mem == True:
    #     if len(loss_mem[-1]) > 2:
    #         for i in range(len(loss_mem[-1]) - 1):
    #             side_change.append((loss_mem[-1][i + 1] - loss_mem[-1][i]) / loss_mem[-1][i])
    #         for i in range(len(side_change) - 1):
    #             rate += side_change[i]
    #         memvalues = (side_change[-1] - rate / (len(side_change) - 1)) * loss_mem[-1][-1]
    #     else:
    #         memvalues = -3.0
    #     change = tf.cast(tf.sigmoid(10*memvalues), dtype=tf.float64)
    #     loss_values.append(change)
    #
    #
    #
    # for task in task_train:
    #     side_change = []
    #     rate = 0
    #     if len(loss_mem[task]) > 2:
    #         for i in range(len(loss_mem[task]) - 1):
    #             side_change.append((loss_mem[task][i + 1] - loss_mem[task][i]) / loss_mem[task][i])
    #         for i in range(len(side_change) - 1):
    #             rate += side_change[i]
    #         values = (side_change[-1] - rate / (len(side_change) - 1)) * loss_mem[task][-1]
    #     else:
    #         values = -3.0
    #     change = tf.cast(tf.sigmoid(values), dtype=tf.float64)
    #     loss_values.append(change)
    #
    # loss = 0
    # loss_norm = []
    # for k in range(len(loss_values)):  # for each layer
    #     loss += loss_values[k]
    # for k in range(len(loss_values)):  # for each layer
    #     # loss_norm.append(losses[k] / loss)
    #     loss_norm.append(loss_values[k] / loss)
    #
    # gradient = []
    # # initialize gradient
    # # GE
    # inital_gradient = []
    # for k in range(len(gradients[0])): # for each layer
    #     g = 0
    #     for i in range(len(gradients)):
    #         g += gradients[i][k]* loss_norm[i]
    #     inital_gradient.append(g)
    #
    # # Reshape update
    # gradient_flat = tf.concat([tf.reshape(grad, [-1]) for grad in inital_gradient], 0)
    # delta = tf.Variable(tf.zeros_like(gradient_flat), trainable=True)
    # # update gradient
    # # optimizer = tf.keras.optimizers.Adam(0.001)
    # # optimizer = tf.keras.optimizers.SGD(0.2)
    # optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # iter_num = 10
    # for i in range(iter_num):
    #     with tf.GradientTape() as tape:
    #         g_loss = 0
    #         for index in range(len(gradients)):
    #             g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients[index]], 0)
    #             g_loss += self.gradsim_object(g_task_flat, gradient_flat + delta)*loss_values
    #         # for g_task in gradients:
    #         #     g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in g_task], 0)
    #         #     g_loss += self.gradsim_object(g_task_flat, gradient_flat+delta)
    #         g_loss /= len(gradients)
    #     # print(i, g_loss)
    #     _g = tape.gradient(g_loss, delta)
    #     optimizer.apply_gradients([(_g, delta)])
    #     # delta = delta.assign_sub(_g)
    # gradient_flat += delta
    # offset = 0
    # for v in self.model.trainable_variables:
    #     shape = v.get_shape()
    #     v_params = 1
    #     for dim in shape:
    #         v_params *= dim
    #     gradient.append(tf.reshape(gradient_flat[offset:offset+v_params], shape))
    #     offset += v_params

    return d


def ComputeGradientnEUD(gradients):
    d = []
    # initialize gradient
    # GE
    inital_gradient = []
    for k in range(len(gradients[0])):  # for each layer
        g = 0
        for i in range(len(gradients)):
            g += gradients[i][k]
        inital_gradient.append(g)
        # MGDA
    # if len(gradients) == 2:
    #     inital_gradient = []
    #     for g0, g1 in zip(gradients[0], gradients[1]):
    #         g0_flat = tf.reshape(g0, [-1])
    #         g1_flat = tf.reshape(g1, [-1])

    #         g12 = tf.reduce_sum(tf.multiply(g0_flat, g1_flat))
    #         g11 = tf.reduce_sum(tf.multiply(g0_flat, g0_flat))
    #         g22 = tf.reduce_sum(tf.multiply(g1_flat, g1_flat))

    #         if g12>=g11:
    #             alpha = 1
    #         elif g12>=g22:
    #             alpha = 0
    #         else:
    #             alpha = tf.reduce_sum(tf.multiply((g1_flat-g0_flat), g1_flat))/((tf.norm(g1_flat-g0_flat,ord=2))**2)
    #         inital_gradient.append(alpha*g0 + (1-alpha)*g1)
    # else:
    #     inital_gradient = []
    #     for k in range(len(gradients[0])): # for each layer
    #         gs = [gradients[i][k] for i in range(len(gradients))]
    #         sol, min_norm = min_norm_solvers.find_min_norm_element(gs)
    #         g = 0
    #         for scale, _g in zip(sol, gs):
    #             g += scale*_g
    #         inital_gradient.append(g)

    # Reshape update
    gradient_flat = tf.concat([tf.reshape(grad, [-1]) for grad in inital_gradient], 0)
    gradients_flat = [tf.concat([tf.reshape(grad, [-1]) for grad in inital_gradient], 0)] + [tf.concat([tf.reshape(grad, [-1]) for grad in g_task], 0) for g_task in gradients]

    grad_norms = [tf.norm(g, ord=2) for g in gradients_flat]
    # grad_factors = tf.nn.softmax(grad_norms, axis=0)
    grad_factors = grad_norms / tf.math.reduce_sum(grad_norms, axis=0)
    gradients_flat = tf.expand_dims(grad_factors, axis=1) * gradients_flat
    delta = tf.Variable(tf.zeros_like(gradient_flat), trainable=True)
    # update gradient
    optimizer = tf.keras.optimizers.Adam(0.003)
    # optimizer = tf.keras.optimizers.SGD(0.2)
    iter_num = 10
    for i in range(iter_num):
        with tf.GradientTape() as tape:
            g_loss = 0
            for g_task in gradients_flat[1:]:
                # g_task_flat = tf.concat([tf.reshape(grad, [-1]) for grad in g_task], 0)
                g_loss += self.gradsim_object(gradients_flat[0], g_task + delta)
            g_loss /= len(gradients)
        # print(len(gradients), i, g_loss)
        _g = tape.gradient(g_loss, delta)
        # optimizer.minimize(g_loss, [delta])
        # print(_g)
        # print(tf.norm(_g, ord=2))
        # exit()
        optimizer.apply_gradients([(_g, delta)])
        # delta = delta.assign_sub(_g)
    gradient_flat += delta
    # print(gradient_flat)
    # exit()
    # Convert the flat projected gradient vector into a list
    offset = 0
    for v in self.model.trainable_variables:
        shape = v.get_shape()
        v_params = 1
        for dim in shape:
            v_params *= dim
        d.append(tf.reshape(gradient_flat[offset:offset + v_params], shape))
        offset += v_params
    # store_proj_grads = tf.group(*store_proj_grad_ops)
    return d
