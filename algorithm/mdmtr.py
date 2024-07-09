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

        assert len(gradients)  in [1, 2], 'ER is the SLL without timeline'
        d = []
        for k in range(len(gradients[0])): # for each layer
            g = 0
            for i in range(len(gradients)):
                g += (1/len(gradients))*gradients[i][k]
            d.append(g)
    return d
