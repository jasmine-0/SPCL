import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
# from algorithm.loss import CosLoss,ElcLoss,ElcNormLoss,GradSimLoss
from algorithm import *
from algorithm import aliged

class Trainer():
    def __init__(self, model, datastream, args):
        super(Trainer, self).__init__()
        self.args = args
        self.increment = args.increment
        self.method = args.imp_method
        self.cnn_method = "OCNN"
        self.datastream = datastream
        self.task_num = datastream.__len__()
        self.BuildObjective(model)
        self.BuildOptimizer(args.optimizer, args.lr)
        # batch level 
        self.batch_gradients = []
        self.batch_losses = []
        self.loss_mem = [] # store the losses from the latest 5 batchs 
        self.loss_start = [] # store the initial losses for each task
        self.mem_loss_start = [] # initial memory loss when a task end

        end_time = [e for s, e in self.datastream.TimeLine]
        self.mem_init_time = min(end_time) + 1

        # Gradnorm: init loss for each task
        self.init_losses = {}
        self.init_mem_losses = []

        self.losses_continuous = {k:[] for k in range(self.task_num)}
        self.gradnorm_continuous = {k:[] for k in range(self.task_num)}
        self.losses_continuous_curr = []
        self.gradnorm_continuous_curr = []
        self.losses_continuous_mean_curr = []
        self.mem_losses_continuous = []
        self.mem_gradnorm_continuous = []

        self.previous_loss_ratios = {k: -1. for k in range(self.task_num)}
        self.previous_loss_ratios_curr = []
        self.mem_previous_loss_ratios = 0.

        self._g = {k: None for k in range(self.task_num)}
        self.m_g = None
        self._g_c = []
        self.X = np.zeros(1)

    def BuildObjective(self, model):
        self.model = model
        if self.method == 'gmed':
            from network import ConstructNetwork
            self.agent_model = ConstructNetwork(self.args.network, self.datastream.TotalClass, self.method)
        self.split_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.split_loss_object_for_onehot = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_object_for_onehot = tf.keras.losses.CategoricalCrossentropy()
        self.train_loss = [tf.keras.metrics.Mean(name='train_loss_{}'.format(i)) for i in range(self.task_num)]
        self.test_loss = [tf.keras.metrics.Mean(name='test_loss_{}'.format(i)) for i in range(self.task_num)]
        self.grad_diff_loss = GradDiffLoss()

        self.train_acc = [tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy_{}'.format(i)) for i in range(self.task_num)]
        self.test_acc = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_{}'.format(i)) for i in range(self.task_num)]
        self.previous_losses = []

    def BuildOptimizer(self, optimizer, learning_rate):
        if optimizer == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate)
            self.image_optimizer = tf.keras.optimizers.SGD(learning_rate)
        elif optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.image_optimizer = tf.keras.optimizers.Adam(learning_rate)
        else:
            raise Exception('Invalid optimizer {}'.format(optimizer))
        
    def compute_loss(self, labels, predictions):
        per_example_loss = self.split_loss_object(labels, predictions)
        avg_loss =  tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])
        return avg_loss, per_example_loss

    def compute_loss_for_onehot(self, labels, predictions):
        per_example_loss = self.split_loss_object_for_onehot(labels, predictions)
        avg_loss =  tf.nn.compute_average_loss(per_example_loss, global_batch_size=labels.shape[0])
        return avg_loss, per_example_loss
    
    def orth_dist(self, mat, stride=None):
        mat = tf.reshape(mat, (mat.shape[0], -1))
        if mat.shape[0] < mat.shape[1]:
            mat = tf.transpose(mat)
        identity_matrix = tf.eye(mat.shape[1], dtype=mat.dtype)
        return tf.norm(tf.matmul(tf.transpose(mat), mat) - identity_matrix)

    def deconv_orth_dist(self, kernel, stride):
        filter_height, filter_width, in_channels, out_channels = kernel.shape
        kernel = tf.reshape(kernel, [-1, out_channels])
        kernel = K.transpose(kernel)
        kernel_norm = K.l2_normalize(kernel, axis=-1)
        sim_matrix = K.dot(kernel_norm, K.transpose(kernel_norm))
        diag_mask = (K.ones_like(sim_matrix) - K.eye(tf.shape(sim_matrix)[0]))
        sim_matrix = sim_matrix * diag_mask
        orth_dist = K.sum(K.square(K.abs(sim_matrix)))
        return orth_dist

    def compute_diff_loss(self):
        diff =  self.orth_dist(self.model.res3.layers[0].conv_shortcut.weights[0]) + \
                self.orth_dist(self.model.res4.layers[0].conv_shortcut.weights[0]) + \
                self.orth_dist(self.model.res5.layers[0].conv_shortcut.weights[0])
        diff += self.deconv_orth_dist(self.model.res2.layers[0].conv2a.weights[0], 1) + \
                self.deconv_orth_dist(self.model.res2.layers[1].conv2a.weights[0], 1)
        diff += self.deconv_orth_dist(self.model.res3.layers[0].conv2a.weights[0], 2) + \
                self.deconv_orth_dist(self.model.res3.layers[1].conv2a.weights[0], 1)
        diff += self.deconv_orth_dist(self.model.res4.layers[0].conv2a.weights[0], 2) + \
                self.deconv_orth_dist(self.model.res4.layers[1].conv2a.weights[0], 1)
        diff += self.deconv_orth_dist(self.model.res5.layers[0].conv2a.weights[0], 2) + \
                self.deconv_orth_dist(self.model.res5.layers[1].conv2a.weights[0], 1)
        
        return diff * 1e-6

    def train_step(self, images, labels, task_id, mask):
        with tf.GradientTape() as tape:
            predictions = self.model(images, mask, training=True)
            loss, per_example_loss = self.compute_loss(labels, predictions)

            if self.cnn_method == "OCNN":
                loss += self.compute_diff_loss()          # OCNN
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.batch_gradients.append(gradients)
        self.batch_losses.append(loss)
        if task_id != None:
            self.train_loss[task_id](loss)
            self.train_acc[task_id](labels, predictions)

    def train_step_with_sigmoid(self, images, labels, task_id, masks):
        assert task_id > 0
        old_task_mask = tf.reduce_max([masks[i] for i in range(task_id + 1)], axis=0)
        new_task_mask = masks[task_id]
        with tf.GradientTape() as tape:
            predictions_old = tf.math.sigmoid(self.model(images, old_task_mask, training=True))
            predictions_new = tf.math.sigmoid(self.model(images, new_task_mask, training=True))
            loss_old, per_example_loss_old = self.compute_loss(labels, predictions_old)
            loss_new, per_example_loss_new = self.compute_loss(labels, predictions_new)
            loss = (loss_old + loss_new) / 2
        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.batch_gradients.append(gradients)
        self.batch_losses.append(loss)
        
    def train_step_for_onehot(self, images, labels, mask):
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(images)
            predictions = self.model(images, mask, training=True)
            loss, per_example_loss = self.compute_loss_for_onehot(labels, predictions)

            if self.cnn_method == "OCNN":
                loss += self.compute_diff_loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # self.tape = tape
        self.batch_gradients.append(gradients)
        self.batch_losses.append(loss)

    def train_step_for_ctml_onehot(self, images, labels, mask, batch_id):
        with tf.GradientTape(persistent=True) as tape:
            logits = self.model(images, mask, training=True)
            logits = logits / tf.norm(self.model.trainable_weights[-1], ord=2, keepdims=False)
            theta_logits = tf.math.acos(logits)
            mc, mt, s = 0.01, 0.01, 20
            mask_seen = self.get_class_mask(batch_id=batch_id, increment='class') # get all seen classes
            mask_task = self.get_task_mask_from_label(labels) # get only the memory corr mask
            mask_class = labels # only the memory mask

            theta_logits = theta_logits + mt*mask_task+ mc*mask_class

            rebuilt_logits = tf.math.cos(theta_logits)
            predictions = self.masked_softmax(s*rebuilt_logits, mask_seen)

            loss, per_example_loss = self.compute_loss_for_onehot(labels, predictions)
            if self.cnn_method == "OCNN":
                loss += self.compute_diff_loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # self.tape = tape
        self.batch_gradients.append(gradients)
        self.batch_losses.append(loss)

    def train_step_for_ctml(self, images, labels, mask, task_id, batch_id):
        with tf.GradientTape(persistent=True) as tape:
            # tape.watch(images)
            logits = self.model(images, mask, training=True)
            logits = logits / tf.norm(self.model.trainable_weights[-1], ord=2, keepdims=False)
            theta_logits = tf.math.acos(logits)
            mc, mt, s = 0.01, 0.01, 20
            mask_class = tf.one_hot(tf.squeeze(labels), self.datastream.TotalClass, dtype=tf.float) # only the memory mask
            mask_task = self.get_task_mask_from_label(mask_class) # get only the memory corr mask
            mask_seen = self.get_class_mask(batch_id=batch_id, increment='class') # get all seen classes

            theta_logits = theta_logits + mt*mask_task+ mc*mask_class

            rebuilt_logits = tf.math.cos(theta_logits)
            predictions = self.masked_softmax(s*rebuilt_logits, mask_seen)
            

            loss, per_example_loss = self.compute_loss(labels, predictions)
            if self.cnn_method == "OCNN":
                loss += self.compute_diff_loss()
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # self.tape = tape
        self.batch_gradients.append(gradients)
        self.batch_losses.append(loss)
        if task_id != None:
            self.train_loss[task_id](loss)
            self.train_acc[task_id](labels, predictions)

    def reset_state(self):
        for task in range(self.task_num):
            self.train_loss[task].reset_states()
            self.train_acc[task].reset_states()

    def InitializeBatch(self):
        self.batch_losses = []
        self.batch_gradients = []

    def GetMemoryGradient(self, batch_id):
        '''
        FN: Find the smallest e, and the following step will have memory and train on it
        '''
        start_time = [s for s, e in self.datastream.TimeLine]
        end_time = [e for s, e in self.datastream.TimeLine]
        end_time_plus = [e+1 for s, e in self.datastream.TimeLine]
        end_time_plus_plus = [e+2 for s, e in self.datastream.TimeLine]
        if batch_id > min(end_time)  and self.args.with_mem in ['ur', 'ur_reduce']:
            # when i > first-task-end-time
            if self.method in ['egda++', 'gmed']:
                images, labels, masks, idx = self.datastream.PickABatchFromMem()
                self.mem_idx = idx
                self.edited_image, self.edited_labels, self.edited_masks = tf.constant(images), tf.constant(labels), tf.constant(masks)
            else:
                images, labels, masks = self.datastream.MemStream.get_next()

            if self.method == 'mdmtr':
                self.train_step_for_ctml_onehot(images, labels, masks, batch_id)
            else:
                masks = self.get_class_mask(batch_id=batch_id, increment='class')  if self.increment !='task' else masks # get all seen classes
                self.train_step_for_onehot(images, labels, masks) # watch out, here is onehot


            # save init memory loss for gradnorm
            if batch_id in end_time_plus:
                self.init_mem_losses.append(self.batch_losses[-1])
            
            if len(self.mem_losses_continuous) > 0:
                if batch_id in end_time_plus + end_time_plus_plus:
                    self.mem_previous_loss_ratios = 1.
                else:
                    self.mem_previous_loss_ratios = self.mem_losses_continuous[-1]/self.mem_losses_continuous[-2]

            # Update the continuous mem losses
            self.mem_losses_continuous.append(self.batch_losses[-1])
            if "egda" in self.method:
                g_task = tf.concat([tf.reshape(grad, [-1]) for grad in self.batch_gradients[-1]], 0)
                g_task_norm = tf.norm(g_task, ord=2, keepdims=False)
                self.mem_gradnorm_continuous.append(g_task_norm)
                if self.m_g == None:
                    self.m_g = g_task_norm
                else:
                    self.m_g = 0.9 * self.m_g + 0.1 * g_task_norm

        else:
            pass

    def GetCurrentGradient(self, batch_id):
        # reset the continus curr
        self.losses_continuous_mean_curr = []
        self.losses_continuous_curr = []
        self.gradnorm_continuous_curr = []
        self._g_c = []
        
        start_time = [s for s, e in self.datastream.TimeLine]
        end_time = [e for s, e in self.datastream.TimeLine]

        if batch_id in end_time:
            flag = True
        else:
            flag = False
        
        for task_id in range(self.datastream.__len__()):
            # when [task start <= i <= task end] do
            if batch_id>=self.datastream.TimeLine[task_id][0] and batch_id<=self.datastream.TimeLine[task_id][1]:                        
                images, labels = self.datastream.TrainStream[task_id].get_next()
                

                if self.method == 'mdmtr':
                    masks = self.get_class_mask(batch_id, task_id)
                    self.train_step_for_ctml(tf.Variable(images), labels, masks, task_id, batch_id)
                else:
                    masks = self.get_class_mask(batch_id, task_id)
                    self.train_step(tf.Variable(images), labels, task_id, masks)

                self.previous_loss_ratios_curr.append(self.previous_loss_ratios[task_id])
                if batch_id != self.datastream.TimeLine[task_id][0]:
                    self.losses_continuous_mean_curr.append(tf.reduce_mean(self.losses_continuous[task_id]))
                else:
                    self.losses_continuous_mean_curr.append(-1.0)
   
                if batch_id in [self.datastream.TimeLine[task_id][0], self.datastream.TimeLine[task_id][0]+1]:
                    self.previous_loss_ratios[task_id] = 1.
                else:
                    self.previous_loss_ratios[task_id] = self.losses_continuous[task_id][-1]/self.losses_continuous[task_id][-2]

                
                self.losses_continuous[task_id].append(self.batch_losses[-1])

                if batch_id in start_time:
                    self.init_losses[task_id] = self.batch_losses[-1]
                
                self.losses_continuous_curr.append(self.losses_continuous[task_id])

                flag = False
                if flag:
                    print("task_id", task_id)
                    print(len(self.losses_continuous[task_id]))
                
                if "egda" in self.method or "aliged" in self.method:              
                    g_task = tf.concat([tf.reshape(grad, [-1]) for grad in self.batch_gradients[-1]], 0)
                    g_task_norm = tf.norm(g_task, ord=2, keepdims=False)
                    self.gradnorm_continuous[task_id].append(g_task_norm) 
                    self.gradnorm_continuous_curr.append(self.gradnorm_continuous[task_id])
                    if self._g[task_id] == None:
                        self._g[task_id] = g_task_norm
                    else:
                        self._g[task_id] = 0.9 * self._g[task_id] + 0.1 * g_task_norm
                    self._g_c.append(self._g[task_id])
                
  
    def GetCurrentGradient_EvaluateTime(self, batch_id):
        # reset the continus curr
        self.losses_continuous_mean_curr = []
        self.losses_continuous_curr = []
        self.gradnorm_continuous_curr = []
        self._g_c = []
        
        start_time = [s for s, e in self.datastream.TimeLine]
        for task_id in range(self.datastream.__len__()):
            # when [task start <= i <= task end] do
            if task_id < 3:                        
                images, labels = self.datastream.TrainStream[task_id].get_next()
                mask = self.get_class_mask(batch_id, task_id)
                self.train_step(tf.Variable(images), labels, task_id, mask)

                self.previous_loss_ratios_curr.append(self.previous_loss_ratios[task_id])
                self.losses_continuous_mean_curr.append(-1.0)
                self.previous_loss_ratios[task_id] = 1.
                self.losses_continuous[task_id].append(self.batch_losses[-1])

                # GradNorm: write the init loss for gradnorm
                if batch_id in start_time:
                    self.init_losses[task_id] = self.batch_losses[-1]
                self.losses_continuous_curr.append(self.losses_continuous[task_id])
                
                if "egda" in self.method:              
                    g_task = tf.concat([tf.reshape(grad, [-1]) for grad in self.batch_gradients[-1]], 0)
                    g_task_norm = tf.norm(g_task, ord=2, keepdims=False)
                    self.gradnorm_continuous[task_id].append(g_task_norm) 
                    self.gradnorm_continuous_curr.append(self.gradnorm_continuous[task_id])
                    if self._g[task_id] == None:
                        self._g[task_id] = g_task_norm
                    else:
                        self._g[task_id] = 0.9 * self._g[task_id] + 0.1 * g_task_norm
                    self._g_c.append(self._g[task_id])  


    def Update(self, method, i):
        end_time = [e for s, e in self.datastream.TimeLine]

        if i in end_time:
            flag = True
        else:
            flag = False

        if len(self.batch_gradients) == 1:
            d = self.batch_gradients[0]
            self.optimizer.apply_gradients(zip(d, self.model.trainable_variables))
        else:
            ###################################################
            # SCL Method
            if method == 'icarl':
                assert len(self.batch_gradients) in [1, 2], 'icarl is the SLL without timeline'
                d = agem.ComputeGradient(self.batch_gradients)
            elif method == 'er': # Experience Replay
                d = er.ComputeGradient(self.batch_gradients, i<self.mem_init_time, flag = flag)
            elif method == 'mdmtr': # Experience Replay
                d = mdmtr.ComputeGradient(self.batch_gradients, i<self.mem_init_time)
            elif method == 'agem': # Continual AGEM 
                d = agem.ComputeGradient(self.batch_gradients, i<self.mem_init_time)
            elif method == 'mega': # Continual AGEM 
                d = mega.ComputeGradient(self.batch_gradients, self.losses_continuous_curr, self.mem_losses_continuous, i<self.mem_init_time)
            elif method == 'gmed': # GMED                
                d = gmed.ComputeGradient(self, self.batch_gradients, i<self.mem_init_time)
            #####################################################
            # PCL Method
            elif method == 'sumgrad': # Gradient Summation
                d = sumgrad.ComputeGradient(self.batch_gradients)
            elif method == 'normgrad': # normalized Gradient Summation
                d = avggrad.ComputeGradient(self.batch_gradients)
            elif method == 'avggrad': # Average Gradient
                d = avggrad.ComputeGradient(self.batch_gradients)
            elif method == 'mgda': # MGDA
                d = mgda.ComputeGradient_v2(self.batch_gradients, i)
            elif method == 'nmgda': # Normalized MGDA
                d = nmgda.ComputeGradient_v2(self.batch_gradients)
            elif method == 'pcgrad': # PCGrad
                d = pcgrad.ComputeGradient(self.batch_gradients)
                # d = pcgrad.ComputeGradientFast(self.batch_gradients)
            elif method == 'gradnorm': # GradNorm
                # insert the memory init loss at the begining
                if len(self.init_mem_losses) == 0:
                    init_losses = [v for k, v in self.init_losses.items()]
                else:
                    init_losses = [tf.reduce_mean(self.init_mem_losses)] + [v for k, v in self.init_losses.items()]
                d = gradnorm.ComputeGradient_v2(self.batch_gradients, self.batch_losses, init_losses, i)
                # d = gradnorm.ComputeGradient_v2(self.batch_gradients, self.batch_losses, self.loss_start, self.mem_loss_start)
            elif method == 'uw': # TODO Uncertainty weighing 
                d = uw.ComputeGradient(self.batch_gradients)
            elif method == 'cvw': # CV-Weighting
                if len(self.mem_losses_continuous) > 1:
                    self.losses_continuous_mean_curr.insert(0, tf.reduce_mean(self.mem_losses_continuous[:-1]))
                elif len(self.mem_losses_continuous) == 1:
                    self.losses_continuous_mean_curr.insert(0, -1.0)
                else:
                    pass
                d = cvw.ComputeGradient(self.batch_gradients, self.batch_losses, self.losses_continuous_mean_curr)
            elif method == 'rlw': # Random Loss Weighting
                d = rlw.ComputeGradient(self.batch_gradients)
            elif method == 'dwa': #
                if len(self.mem_losses_continuous) > 0:
                    previous_loss_ratios = [self.mem_previous_loss_ratios] + self.losses_continuous_mean_curr
                else:
                    previous_loss_ratios = self.losses_continuous_mean_curr
                d = dwa.ComputeGradient(self.batch_gradients, previous_loss_ratios)
            elif method == 'imtl':
                d = imtl.ComputeGradient_v2(self.batch_gradients)
            elif method == 'graddrop':
                d = graddrop.ComputeGradient_v2(self.batch_gradients)
            elif method == 'gradvac':
                d = gradvac.ComputeGradient_v2(self.batch_gradients)
            elif method =='egda':
                _g_n = [self.m_g] + self._g_c if self.m_g != None else self._g_c
                d, X = egda.ComputeGradient(self.batch_gradients, self.losses_continuous_curr, self.mem_losses_continuous, _g_n, self.X, flag = flag)
                self.X = X

            elif method =='ragrad':
                d = ragrad.ComputeGradient(self.batch_gradients, i)
            elif method == 'aliged':
                _g_n = [self.m_g] + self._g_c if self.m_g != None else self._g_c
                d = aliged.ComputeGradient(self.batch_gradients, self.losses_continuous_curr, self.mem_losses_continuous, _g_n, flag = flag)
                # d = aliged.ComputeGradient(self.batch_gradients)

            elif method =='maxdo':
                d = maxdo.ComputeGradient(self.batch_gradients, i, flag = flag)
                
            else:
                raise Exception('Invalid method')

            # real update with computed gradient d
            self.optimizer.apply_gradients(zip(d, self.model.trainable_variables))

            

            d.clear() # to release the GPU
        
        for task_id in range(self.datastream.__len__()):
            # when a task finish, save data to memory buffer
            if i==self.datastream.TimeLine[task_id][1] and self.args.with_mem in ['ur', 'ur_reduce']:
                if self.method in ['gmed', 'egda++'] :
                    self.datastream.UpdataeMemory2(task_id, self.args.with_mem, mem_split='ref') # Construct datastream
                else:
                    self.datastream.UpdataeMemory(task_id, self.args.with_mem, mem_split='ref') # Construct datastream

    def get_class_mask(self, batch_id=None, task_id=None, increment=None):
        if increment == None:
            if self.increment == 'task' and self.method != 'mdmtr':
                mask = self.datastream.MaskSet[task_id]
            else:
                start_time = [s for s, e in self.datastream.TimeLine]
                mask =  np.zeros((self.datastream.TotalClass), dtype=np.float64)
                for task_id in range(self.datastream.__len__()):
                    if batch_id >= start_time[task_id]:
                        mask += self.datastream.MaskSet[task_id]
                mask = mask > 0
                mask = mask.astype('float')
        elif increment == 'task':
            mask = self.datastream.MaskSet[task_id]
        elif increment == 'class':
            start_time = [s for s, e in self.datastream.TimeLine]
            mask =  np.zeros((self.datastream.TotalClass), dtype=np.float64)
            for task_id in range(self.datastream.__len__()):
                if batch_id >= start_time[task_id]:
                    mask += self.datastream.MaskSet[task_id]
            mask = mask > 0
            mask = mask.astype('float')
        else:
            raise Exception('Invalid increment type!')
        
        return mask

    def get_task_mask_from_label(self, labels):
        labels = labels.numpy()
        masks = []
        for i in range(labels.shape[0]):
            for j in range(self.datastream.MaskSet.shape[0]):
                if max(labels[i] + self.datastream.MaskSet[j]) == 2:
                    masks.append(self.datastream.MaskSet[j])
                    break
        masks = np.concatenate([masks], -1)
        return masks

    def masked_softmax(self, scores, mask):
        scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
        exp_scores = tf.exp(scores)
        exp_scores *= mask
        exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])+1e-7) 

    
class GradDiffLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(GradDiffLoss, self).__init__()
 
    def call(self, y_true, y_pred):
        loss = tf.norm(y_true-y_pred, ord=2)**2
        return loss