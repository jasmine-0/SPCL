import tensorflow as tf
import numpy as np
import re
from algorithm.loss import CosLoss,ElcLoss,ElcNormLoss,GradSimLoss
# from algorithm import agem,avggrad,icarl,mgda,nmgda,sumgrad,pcgrad,gradnorm,uw,cvw,gcm_optim

class Evaluator():
    def __init__(self, model, datastream, args, recoder, logger):
        super(Evaluator, self).__init__()
        self.args = args
        self.increment = args.increment
        self.model = model
        self.task_num = datastream.__len__()
        self.datastream = datastream
        self.BuildObjective(model)
        self.recorder = recoder
        self.logger = logger
        self.eval_mode = "standard"
        self.cache_value = []

    def BuildObjective(self, model):
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.loss_object_for_onehot = tf.keras.losses.CategoricalCrossentropy()
        self.test_loss = [tf.keras.metrics.Mean(name='test_loss_{}'.format(i)) for i in range(self.task_num)]
        self.test_acc = [tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy_{}'.format(i)) for i in range(self.task_num)]

    def test_step(self, images, labels, task_id, mask):
        if self.args.imp_method == 'mdmtr':
            logits = self.model(images, mask, training=False)
            predictions = self.masked_softmax(logits, mask)
        else:
            predictions = self.model(images, mask, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss[task_id](t_loss)
        self.test_acc[task_id](labels, predictions)

    def test_step_without_save(self, images, labels, task_id, mask):
        predictions = self.model(images, mask, training=False)
        t_loss = self.loss_object(labels, predictions)

    def test_step_cluster(self, images, labels, task_id, mask, class_center):
        predictions = self.model(images, mask, with_softmax=False, training=False)
        preds = []
        for pred in predictions:
            pred = tf.norm(class_center - pred, axis=-1)
            pred = tf.one_hot(tf.argmin(pred), mask.shape[0], 1, 0, dtype=tf.float32)
            preds.append(pred[tf.newaxis, :])
        preds = tf.concat(preds, axis=0)
        self.test_loss[task_id](0)
        self.test_acc[task_id](labels, preds)

    def reset_state(self):
        for task in range(self.task_num):
            self.test_loss[task].reset_states()
            self.test_acc[task].reset_states()

    def EvaluateTask(self,labelset_id, timeline_id, run_id, batch_id):
        if self.eval_mode == "standard":
            end_time = [e for s, e in self.datastream.TimeLine]
            # 1. if a task finished, evaluate for its first shot acc
            if batch_id in end_time:
                task_ids = [i for i, idx in enumerate(end_time) if idx == batch_id] # some tasks end at the same time
                for task_id in task_ids:
                    mask = self.get_class_mask(batch_id, task_id)
                    for images, labels in self.datastream.TestStream[task_id]:
                        self.test_step(images, labels, task_id, mask)
                    # 1.1 record the first shot acc
                    self.recorder.first_acc_record[labelset_id, timeline_id, run_id, task_id] =  self.test_acc[task_id].result().numpy()
                    # 1.2 log info the result
                    self.logger.PrintResultPerTask(labelset_id, timeline_id, run_id, task_id, self.test_loss[task_id].result(), self.test_acc[task_id].result(), is_first=True)
                    # 1.3 reset eval state
                    self.reset_state()
            
            if batch_id == max([e for s, e in self.datastream.TimeLine]): #or batch_id == 399:
                self.reset_state()
                for task_id in range(self.datastream.__len__()):
                    mask = self.get_class_mask(batch_id, task_id)
                    for images, labels in self.datastream.TestStream[task_id]:
                        self.test_step(images, labels, task_id, mask)
                    # 2.2 record the finish shot result
                    self.recorder.finish_acc_record[labelset_id, timeline_id, run_id, task_id] =  self.test_acc[task_id].result().numpy()
                # 2.3 log info the results
                first_acc_avg = np.mean(self.recorder.first_acc_record[labelset_id, timeline_id, run_id])
                first_acc_std = np.std(self.recorder.first_acc_record[labelset_id, timeline_id, run_id])
                finish_acc_avg = np.mean(self.recorder.finish_acc_record[labelset_id, timeline_id, run_id])
                finish_acc_std = np.std(self.recorder.finish_acc_record[labelset_id, timeline_id, run_id])
                forget_record = self.recorder.finish_acc_record[labelset_id, timeline_id, run_id] - self.recorder.first_acc_record[labelset_id, timeline_id, run_id]
                self.recorder.forget_record[labelset_id, timeline_id, run_id] = forget_record
                forget_avg = np.mean(forget_record)
                forget_std = np.std(forget_record)
                self.logger.PrintResultPerRun(labelset_id, timeline_id, run_id, first_acc_avg, first_acc_std, finish_acc_avg, finish_acc_std, forget_avg, forget_std)
                self.reset_state()
        elif self.eval_mode == "eaxct":            
            # 1. if a task finished, evaluate for all seen task acc
            if batch_id in [e for s, e in self.datastream.TimeLine]:                
                for task_id in range(self.datastream.__len__()):
                    for images, labels in self.datastream.TestStream[task_id]:
                        self.test_step(images, labels, task_id, self.datastream.MaskSet[task_id])
                    # 1.1 record the first shot acc
                    self.recorder.first_acc_record[labelset_id, timeline_id, run_id, task_id] =  self.test_acc[task_id].result().numpy()
                    # 1.2 log info the result
                    self.logger.PrintResultPerTask(labelset_id, timeline_id, run_id, task_id, self.test_loss[task_id].result(), self.test_acc[task_id].result())
                    # 1.3 reset eval state
                    self.reset_state()
            raise NotImplementedError
        elif self.eval_mode == "batch":
            raise NotImplementedError
        elif self.eval_mode == "noeval":
            raise NotImplementedError

    def EvaluateBatch(self,labelset_id, timeline_id, run_id, batch_id):
        
        start_time = [s for s, e in self.datastream.TimeLine]
        end_time = [e for s, e in self.datastream.TimeLine]
        seen_task_id = []
        for s in start_time:
            if batch_id >= s:
                seen_task_id.append(start_time.index(s))
        
        self.reset_state()
        
        self.reset_state()
        for images, labels in self.datastream.TestStream[0]:
            self.test_step(images, labels, 0, self.datastream.MaskSet[0])
            # 2.2 record the result
        task_0_acc =  self.test_acc[0].result().numpy()
        self.reset_state()
        
        for images, labels in self.datastream.TestStream[1]:
            self.test_step(images, labels, 1, self.datastream.MaskSet[1])
            # 2.2 record the result
        task_1_acc =  self.test_acc[1].result().numpy()
        self.reset_state()
            
        total_acc = (task_0_acc + task_1_acc)/2
        return task_0_acc, task_1_acc, total_acc

    def get_class_mask(self, batch_id, task_id):
        if self.increment == 'task':
            mask = self.datastream.MaskSet[task_id]
        else:
            start_time = [s for s, e in self.datastream.TimeLine]
            mask =  np.zeros((self.datastream.TotalClass), dtype=np.float)
            for task_id in range(self.datastream.__len__()):
                if batch_id >= start_time[task_id]:
                    mask += self.datastream.MaskSet[task_id]
            mask = mask > 0
            mask = mask.astype('float')
        return mask


    def masked_softmax(self, scores, mask):
        scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
        exp_scores = tf.exp(scores)
        exp_scores *= mask
        exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / (tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])+1e-7) 



        
