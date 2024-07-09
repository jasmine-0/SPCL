# Function: Compute the obj decent gradient using AGEM
# Reference: Rebuffi S A, Kolesnikov A, Sperl G, et al. icarl: Incremental classifier and representation learning[C]//CVPR2017.	
# Code by Fan Lyu: fanlyu@tju.edu.cn
# 2022 Mar 12
# Equation:  d = 

import tensorflow as tf

def ComputeGradient(Trainer, gradients, mem_not_begin):
    if mem_not_begin:
        d = []
        for k in range(len(gradients[0])):
            g = 0
            for i in range(len(gradients)): # for each task
                g +=  (1/len(gradients))*gradients[i][k]
            d.append(g)
    else:
        only_new_d = []
        for k in range(len(gradients[0])):
            g = 0
            for i in range(len(gradients)-1): # for each task
                g +=  (1/(len(gradients)-1))*gradients[i+1][k]
            only_new_d.append(g)
        
        # obtain the previous loss of memory
        loss_before = Trainer.mem_losses_continuous[-1]
        # pseudo update with only new tasks' gradients
        # Trainer.agent_model.apply(only_new_d)
        Trainer.optimizer.apply_gradients(zip(only_new_d, Trainer.model.trainable_variables))

        # Pixel-level update via loss difference

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(Trainer.edited_image)
            predictions = Trainer.agent_model(Trainer.edited_image, Trainer.edited_masks, training=True)
            loss_after, per_example_loss = Trainer.compute_loss_for_onehot(Trainer.edited_labels, predictions)
            gs = tape.gradient(loss_after - loss_before, Trainer.edited_image)
        Trainer.datastream.mem_set[0][Trainer.mem_idx] -= 0.003 * gs

        # Real update with both gradients
        Trainer.train_step_for_onehot(Trainer.datastream.mem_set[0][Trainer.mem_idx], Trainer.edited_labels, Trainer.edited_masks)

        # reassign value of the agent model
        Trainer.agent_model.set_weights(Trainer.model.get_weights())

        gradients = [Trainer.batch_gradients[-1], only_new_d]

        assert len(gradients)  in [1, 2], 'GMED is the SLL without timeline'
        d = []
        for k in range(len(gradients[0])): # for each layer
            g = 0
            for i in range(len(gradients)):
                g += (1/len(gradients))*gradients[i][k]
            d.append(g)
    return d


