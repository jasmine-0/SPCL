import numpy as np
from qpsolvers import solve_qp

import tensorflow as tf



# class MinNormSolver:
MAX_ITER = 250
STOP_CRIT = 1e-5

def _min_norm_element_from2(v1v1, v1v2, v2v2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1 + v2v2 - 2*v1v2) )
    # cost = v2v2 + gamma*(v1v2 - v2v2)
    cost = gamma*gamma*v1v1 + 2*gamma*(1-gamma)*v1v2 + (1.-gamma)*(1.-gamma)*v2v2
    return gamma, cost

def _min_norm_2d(vecs, dps):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = 1e8
    for i in range(len(vecs)):
        for j in range(i+1,len(vecs)):
            if (i,j) not in dps:
                dps[(i,j)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[i], [-1]), tf.reshape(vecs[j], [-1]))).numpy()
            #     dps[(i, j)] = 0.0
            #     print(vecs.shape)
            #     print(vecs[i].shape)
            #     print(len(vecs[i]))
            #     exit()
            #     for k in range(len(vecs[i])):
            #         # dps[(i,j)] += torch.dot(vecs[i][k], vecs[j][k]).data[0]
            #         dps[(i,j)] += tf.reduce_sum(tf.math.multiply(vecs[i][k], vecs[j][k])).numpy()
                dps[(j, i)] = dps[(i, j)]
            if (i,i) not in dps:
                dps[(i,i)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[i], [-1]), tf.reshape(vecs[i], [-1]))).numpy()

            #     dps[(i, i)] = 0.0
            #     for k in range(len(vecs[i])):
            #         # dps[(i,i)] += torch.dot(vecs[i][k], vecs[i][k]).data[0]
            #         dps[(i,j)] += tf.reduce_sum(tf.math.multiply(vecs[i][k], vecs[i][k])).numpy()
            if (j,j) not in dps:
                dps[(j,j)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[j], [-1]), tf.reshape(vecs[j], [-1]))).numpy()

            #     dps[(j, j)] = 0.0   
            #     for k in range(len(vecs[i])):
            #         # dps[(j, j)] += torch.dot(vecs[j][k], vecs[j][k]).data[0]
            #         dps[(i,j)] += tf.reduce_sum(tf.math.multiply(vecs[j][k], vecs[j][k])).numpy()
            c,d = _min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
            if d < dmin:
                dmin = d
                sol = [(i,j),c,d]
    return sol, dps

def _projection2simplex(y):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    m = len(y)
    sorted_y = np.flip(np.sort(y), axis=0)
    tmpsum = 0.0
    tmax_f = (np.sum(y) - 1.0)/m
    for i in range(m-1):
        tmpsum+= sorted_y[i]
        tmax = (tmpsum - 1)/ (i+1.0)
        if tmax > sorted_y[i+1]:
            tmax_f = tmax
            break
    return np.maximum(y - tmax_f, np.zeros(y.shape))

def _next_point(cur_val, grad, n):
    proj_grad = grad - ( np.sum(grad) / n )
    tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
    tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
    
    skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
    t = 1
    if len(tm1[tm1>1e-7]) > 0:
        t = np.min(tm1[tm1>1e-7])
    if len(tm2[tm2>1e-7]) > 0:
        t = min(t, np.min(tm2[tm2>1e-7]))

    next_point = proj_grad*t + cur_val
    next_point = _projection2simplex(next_point)
    return next_point

def find_min_norm_element(vecs):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    # Solution lying at the combination of two points
    dps = {}
    init_sol, dps = _min_norm_2d(vecs, dps)
    
    n = len(vecs)
    sol_vec = np.zeros(n)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec , init_sol[2]

    iter_count = 0

    grad_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grad_mat[i,j] = dps[(i, j)]
            

    while iter_count < MAX_ITER:
        grad_dir = -1.0*np.dot(grad_mat, sol_vec)
        new_point = _next_point(sol_vec, grad_dir, n)
        # Re-compute the inner products for line search
        v1v1 = 0.0
        v1v2 = 0.0
        v2v2 = 0.0
        for i in range(n): # reshape as a vector to compute
            for j in range(n):
                v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc*sol_vec + (1-nc)*new_point
        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < STOP_CRIT:
            return sol_vec, nd
        sol_vec = new_sol_vec

def find_min_norm_element_FW(vecs):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
    as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
    """
    # Solution lying at the combination of two points
    dps = {}
    init_sol, dps = _min_norm_2d(vecs, dps)

    n=len(vecs)
    sol_vec = np.zeros(n)
    sol_vec[init_sol[0][0]] = init_sol[1]
    sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    if n < 3:
        # This is optimal for n=2, so return the solution
        return sol_vec , init_sol[2]

    iter_count = 0

    grad_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grad_mat[i,j] = dps[(i, j)]

    while iter_count < MAX_ITER:
        t_iter = np.argmin(np.dot(grad_mat, sol_vec))

        v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
        v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
        v2v2 = grad_mat[t_iter, t_iter]

        nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        new_sol_vec = nc*sol_vec
        new_sol_vec[t_iter] += 1 - nc

        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < STOP_CRIT:
            return sol_vec, nd
        sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data[0] for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn



def _min_norm_2d_with_tol(vecs, dps, tols):
    """
    Find the minimum norm solution as combination of two points
    This is correct only in 2D
    ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
    """
    dmin = None
    for i in range(len(vecs)):
        for j in range(i+1,len(vecs)):
            if (i,j) not in dps:
                dps[(i,j)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[i], [-1]), tf.reshape(vecs[j], [-1]))).numpy()
                dps[(j, i)] = dps[(i, j)]
            if (i,i) not in dps:
                dps[(i,i)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[i], [-1]), tf.reshape(vecs[i], [-1]))).numpy()
            if (j,j) not in dps:
                dps[(j,j)] = tf.reduce_sum(tf.math.multiply(tf.reshape(vecs[j], [-1]), tf.reshape(vecs[j], [-1]))).numpy()
                
            c,d = _min_norm_element_from2_with_tol_v2(dps[(i,i)], dps[(i,j)], dps[(j,j)], tols[i], tols[j])
            
            if dmin == None:
                dmin = d
                sol = [(i,j),c,d]
            else:
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
    return sol, dps

def _min_norm_element_from2_with_tol(g1, g2, sigma_1, sigma_2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    g1_flat = tf.reshape(g1, [-1])
    g2_flat = tf.reshape(g2, [-1])
    m1 = sigma_1/(sigma_2) * g2_flat - g1_flat
    m2 = g2_flat / (sigma_2)
    d = tf.norm(1-m1, ord=2)
    gamma = -1.0 * tf.math.multiply(m1, m2) / (d**2)
    cost = tf.norm(gamma*g1_flat + (1 - gamma*sigma_1) / sigma_2 * g2_flat, ord=2)
    cost = cost * cost
    return gamma, cost

def _min_norm_element_from2_with_tol_v2(v1v1, v1v2, v2v2, tol1, tol2):
    """
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimzed
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    
    # Case: Fig 1, second column
    gamma =  ((tol1 / (tol2*tol2 + 1e-10)) * v2v2 - (1.0 / (tol2 + 1e-10)) * v1v2) / (v1v1 + ((tol1*tol1) / (tol2*tol2 + 1e-10)) * v2v2 - (tol1/(tol2 + 1e-10)) * 2 * v1v2) 
    # cost =  v1  v2v2 + gamma*(v1v2 - v2v2)
    cost = gamma*gamma*v1v1 + \
        2*gamma*(1-gamma*tol1)*v1v2/(tol2 + 1e-10) + \
        (1.-gamma*tol1)*(1.-gamma*tol1)*v2v2 / (tol2*tol2 + 1e-10)
    return gamma, cost


def _projection2simplex_with_tol(y, tols):
    """
    Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
    """
    sorted_idx = np.flip(np.argsort(y), axis=0)
    tmpsum = 0.0
    tmpsum_tol = 0.0
    tmax_f =  (np.sum(np.inner(y, tols)) - 1.0) / np.sum(np.inner(tols,tols))
    
    for i in sorted_idx[:-1]:        
        tmpsum += y[i] * tols[i] # plus from large to small
        tmpsum_tol += tols[i] * tols[i] # plus from large to small
        tmax =  (tmpsum - 1.) / (tmpsum_tol) #
        if tols[i] * tmax > y[i]: # 基本无法满足条件
            tmax_f = tmax
            break
    
    output = np.maximum(y - tmax_f * tols, np.zeros(y.shape))
    return output
    
def _next_point_with_tol(cur_val, grad, n, tols):
    
    proj_grad = grad - ( np.sum(grad) / n ) # 一定下降的方向
    # tm1 = -1.0 * cur_val[proj_grad<0] / proj_grad[proj_grad<0]
    # tm2 = (1.0 - cur_val[proj_grad>0]) / (proj_grad[proj_grad>0])
    
    # skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
    # t = 1 # 步长
    # if len(tm1[tm1>1e-7]) > 0:
    #     t = np.min(tm1[tm1>1e-7])
    # if len(tm2[tm2>1e-7]) > 0:
    #     t = min(t, np.min(tm2[tm2>1e-7]))

    tm1 = -1.0 * cur_val[proj_grad<0]*tols[proj_grad<0] / proj_grad[proj_grad<0]
    tm2 = (1.0 - cur_val[proj_grad>0]*tols[proj_grad>0]) / (proj_grad[proj_grad>0])
    
    skippers = np.sum(tm1<1e-7) + np.sum(tm2<1e-7)
    t = 1 # 步长
    if len(tm1[tm1>1e-7]) > 0:
        t = np.min(tm1[tm1>1e-7])
    if len(tm2[tm2>1e-7]) > 0:
        t = min(t, np.min(tm2[tm2>1e-7]))
    
    next_point = proj_grad * .01 + cur_val
    # print(cur_val)
    # print(next_point)
    # print(proj_grad)
    # print(t)
    # print(t*proj_grad)
    # exit()
    # _n = next_point
    _n = _projection2simplex_with_tol(next_point, tols)
    return next_point, _n

def _next_point_with_tol_v2(cur_val, grad, n, tols, lr):
    
    # proj_grad = grad - ( np.sum(grad) / n ) # 一定下降的方向

    next_point = grad * lr + cur_val
    # print(cur_val)
    # print(next_point)
    # print(proj_grad)
    # print(t)
    # print(t*proj_grad)
    # exit()
    # _n = next_point
    # _n = _projection2simplex_with_tol(next_point, tols)
    return next_point


def find_min_norm_element_with_tol(vecs, tols):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    
    # Solution lying at the combination of two points
    dps = {}
    init_sol, dps = _min_norm_2d_with_tol(vecs, dps, tols)
    
    n = len(vecs)
    # sol_vec = np.zeros(n)
    # sol_vec[init_sol[0][0]] = init_sol[1]
    # sol_vec[init_sol[0][1]] = 1 - init_sol[1]

    # if n < 3:
    #     # This is optimal for n=2, so return the solution
    #     # return sol_vec , init_sol[2]
    #     return sol_vec

    iter_count = 0

    grad_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grad_mat[i,j] = dps[(i, j)]
    # print(grad_mat)
    # sol_vec = tols / n
    sol_vec = np.ones([n], dtype=np.float64) / n
    P = grad_mat
    A = tols
    q = np.zeros([n], dtype=np.float64)
    # print(grad_mat)
    # print(tols)
    # b = np.ones([n], dtype=np.float64)
    b = np.array([1.], dtype=np.float64)
    lb = (1/(n+1))*np.ones([n], dtype=np.float64)
    # lb = 0.*np.ones([n], dtype=np.float64)
    # ub = .8*np.ones([n], dtype=np.float64)
    sol_method = "quadprog"
    sol_method.encode()
    sol_vec = solve_qp(P=P, q=q, A=A, b=b, lb=lb, initvals=sol_vec, solver=sol_method)
    # print(sol_vec)
    # print(sol_vec, tols, np.sum(np.inner(sol_vec, tols)))
    # print("gamma*norm", np.matmul(np.matmul(sol_vec, grad_mat), sol_vec))
    # print('------')
    # exit()
    return sol_vec
    
    # print(sol_vec)
    lr = 0.001

    while iter_count < MAX_ITER:
        # grad_dir = -1.0*np.dot(grad_mat, sol_vec)
        # grad_dir = - np.inner(sol_vec, tols) * tols + 1
        # print(-1.0*np.dot(grad_mat, sol_vec))
        # print(- np.inner(sol_vec, tols) * tols + 1)
        grad_dir = -1.0*np.dot(grad_mat, sol_vec) - 100*(np.inner(sol_vec, tols) * tols - 1)
        # exit()
        # line search
        print(iter_count, "- 1 gamma*norm", np.matmul(np.matmul(sol_vec, grad_mat), sol_vec))
        new_point = _next_point_with_tol_v2(sol_vec, grad_dir, n, tols, lr)
        print(new_point, tols, np.sum(np.inner(new_point, tols)))
        # print(_n, tols, np.sum(np.inner(_n, tols)))
        print(iter_count, "- 2 gamma*norm", np.matmul(np.matmul(new_point, grad_mat), new_point))
        # print("gamma*norm", np.matmul(np.matmul(_n, grad_mat), new_point))
        # print('--------------------')
        # exit()
        new_sol_vec = new_point
        
        # Re-compute the inner products for line search
        # v1v1 = 0.0
        # v1v2 = 0.0
        # v2v2 = 0.0
        # for i in range(n): # reshape as a vector to compute
        #     for j in range(n):
        #         v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
        #         v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
        #         v2v2 += new_point[i]*new_point[j]*dps[(i,j)]

        # nc, nd = _min_norm_element_from2(v1v1, v1v2, v2v2)
        # new_sol_vec = nc*sol_vec + (1-nc)*new_point
        
        change = new_sol_vec - sol_vec
        if np.sum(np.abs(change)) < STOP_CRIT:
            sol_vec =  _projection2simplex_with_tol(sol_vec, tols)
            print(sol_vec, tols, np.sum(np.inner(sol_vec, tols)))
            print(iter_count, "Mapping: gamma*norm", np.matmul(np.matmul(sol_vec, grad_mat), sol_vec))
            print('-------------')
            return sol_vec
        # return _n
        sol_vec = new_sol_vec
        
        # if iter_count % 50 == 0:
        #     # lr /= 10
        #     sol_vec =  _projection2simplex_with_tol(sol_vec, tols)
        #     # print(sol_vec, tols, np.sum(np.inner(sol_vec, tols)))
        #     # print('--')
            
        iter_count += 1 # delete this line for unlimited optimization
        # if iter_count == 250:
        #     exit()
    sol_vec =  _projection2simplex_with_tol(sol_vec, tols)
    # print(sol_vec, tols, np.sum(np.inner(sol_vec, tols)))
    print(iter_count, "Mapping: gamma*norm", np.matmul(np.matmul(sol_vec, grad_mat), sol_vec))
    print('-------------')
    return sol_vec
    # return _n


def find_min_norm_element_independent(vecs):
    """
    Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
    It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
    Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
    """
    
    # Solution lying at the combination of two points
    dps = {}
    init_sol, dps = _min_norm_2d(vecs, dps)
    
    n = len(vecs)

    iter_count = 0

    grad_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grad_mat[i,j] = dps[(i, j)]
    # print(grad_mat)
    # sol_vec = tols / n
    sol_vec = np.ones([n], dtype=np.float64) / n
    P = grad_mat
    A = np.ones([n], dtype=np.float64)
    q = np.zeros([n], dtype=np.float64)
    # print(grad_mat)
    # print(tols)
    # b = np.ones([n], dtype=np.float64)
    b = np.array([1.], dtype=np.float64)
    # lb = (1/(n+1))*np.ones([n], dtype=np.float64)
    lb = 0.*np.ones([n], dtype=np.float64)
    # ub = .8*np.ones([n], dtype=np.float64)
    sol_method = "quadprog"
    sol_method.encode()
    sol_vec = solve_qp(P=P, q=q, A=A, b=b, lb=lb, initvals=sol_vec, solver=sol_method)
    # print(sol_vec)
    # print(sol_vec, tols, np.sum(np.inner(sol_vec, tols)))
    # print("gamma*norm", np.matmul(np.matmul(sol_vec, grad_mat), sol_vec))
    # print('------')
    # exit()
    return sol_vec
    

