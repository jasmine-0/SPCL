import tensorflow as tf
import numpy as np
# from prettytable import PrettyTable
# import min_norm_solvers
import pandas as pd
import argparse

import logging
import time
import os
import yaml



def set_record(task_num):
    test_acc_record = [[] for task in range(task_num)]
    return test_acc_record

def ArgParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str,   default='EMNIST',           help="Dataset.")
    parser.add_argument("--imp_method",     type=str,   default='mdmtr',             help="Implemented Method.")
    parser.add_argument("--network",        type=str,   default='mlp',              help="Implemented Method.")
    parser.add_argument("--optimizer",      type=str,   default='adam',             help="Optimizer.")
    parser.add_argument("--lr",             type=float, default=0.003,             help="Learning rate")
    parser.add_argument("--seed",           type=int,   default=1234,               help="Random Seed.")
    parser.add_argument("--with_mem",       type=str,   default='ur_reduce',        help="Memory type. It can only be [r, ur]")
    parser.add_argument("--mem_per_class",  type=int,   default=5,                 help="Memory size of per class.")
    parser.add_argument("--num_runs",       type=int,   default=3,                  help="Total runs/ experiments.")
    parser.add_argument("--gpu_id",         type=int,   default=6,                  help="GPU to be used.")
    parser.add_argument("--batch_size",     type=int,   default=128,                help="Batch Size.")
    parser.add_argument("--task_num",       type=int,   default=5,                 help="Task num.")
    parser.add_argument("--record_dir",     type=str,   default='record/class_increment/emnist',    help="record path")
    parser.add_argument("--increment",      type=str,   default='task',             help="increment mode; [task, class, free]")
    args = parser.parse_args()
    if args.with_mem not in ['r', 'ur', 'ur_reduce']:
        print('Invalid Memory type. No Memory will be used!')

    return args

def SetFloatType(float_type=64):
    if float_type == 32:
        np_float_x = np.float32
        tf.keras.backend.set_floatx('float32')
    elif float_type == 64: # <=== suggest for stable seed
        np_float_x = np.float
        tf.keras.backend.set_floatx('float64')
    else:
        raise Exception('Invalid Float Type {}'.format(float_type))
    print('Float type setting, Not implemented.')


def SetGPU(gpu_id):
    # set specific gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)

class Recorder():
    '''
    This class record the results to 
    '''
    def __init__(self,args, labelset, timeline):
        super(Recorder, self).__init__()
        self.args = args
        self.labelset_num = len(labelset)
        self.timeline_num = len(timeline[0])
        self.first_acc_record  = np.zeros([len(labelset), len(timeline[0]), args.num_runs, args.task_num])
        self.finish_acc_record = np.zeros([len(labelset), len(timeline[0]), args.num_runs, args.task_num])
        self.forget_record     = np.zeros([len(labelset), len(timeline[0]), args.num_runs, args.task_num])
        # self.first_acc_record  = np.zeros([1, 1, 2, args.task_num])
        # self.finish_acc_record = np.zeros([1, 1, 2, args.task_num])
        # self.forget_record     = np.zeros([1, 1, 2, args.task_num])

        # self.batch_mat = [for t in timeline  np.zeros([timeline[t] args.num_runs, (TimeLines[l][t][-1][1] + 1) // 5])]
        
        self.mean_testacc = []
        self.mean_testacc_finished = []
        self.mean_batchacc = []
        self.time_acc = []

        self.MakeRecordFolder()


    def InitializeRecoder(self):
        pass
        
    def save(self):
        pass

    def BatchRecoder(self):
        raise NotImplementedError

    def MakeRecordFolder(self):
        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        self.save_dir_cache = '{}/cache-{}-{}-{}'.format(self.args.record_dir, time.strftime('%Y%m%d%H%M%s'), self.args.imp_method, self.args.dataset)
        # if there exists a folder with same name, add an extra index to omit error
        if os.path.exists(self.save_dir_cache):
            i = 1
            while os.path.exists(self.save_dir_cache):
                i += 1
                self.save_dir_cache = '{}/cache-{}-{}-{}-{}'.format(self.args.record_dir, time.strftime('%Y%m%d%H%M%s'), self.args.imp_method, self.args.dataset, i)
        try:
            os.makedirs(self.save_dir_cache)
        except Exception as e:
            raise SyntaxError(e)

    def Record2File(self, dataset_num, timeline_num, run_num, task_num):
        # 1. save the args
        with open(self.save_dir_cache + '/args.yml', 'w') as y:
            yaml.dump(vars(self.args), y)
        # 2. save the record
        # 2.1 save total record
        total_run = dataset_num*timeline_num*run_num
        dc = [i for i in range(task_num)]
        di = []
        for i in range(dataset_num):
            for j in range(timeline_num):
                for k in range(run_num):
                    di.append('{}-{}-{}'.format(i+1, j+1, k+1))
        df_first = pd.DataFrame(np.resize(self.first_acc_record, [total_run, task_num]), columns=dc, index=di)
        df_first.to_csv('{}/first.csv'.format(self.save_dir_cache), mode='w', index_label='d-l-r')
        
        df_first = pd.DataFrame(np.resize(self.finish_acc_record, [total_run, task_num]), columns=dc, index=di)
        df_first.to_csv('{}/finish.csv'.format(self.save_dir_cache), mode='w', index_label='d-l-r')

        df_forget = pd.DataFrame(np.resize(self.forget_record, [total_run, task_num]), columns=dc, index=di)
        df_forget.to_csv('{}/forget.csv'.format(self.save_dir_cache), mode='w', index_label='d-l-r')
        # 3. save the batch record
        pass
        
    def save(self, name, labelset, timeline, task_num, run_num, savename):
        columns = []
        for i in range(task_num):
            task_columns = 'task' + str(i)
            columns.append(task_columns)
        index = []
        for i in range(run_num):
            run_index = 'run' + str(i)
            index.append(run_index)
        label_batch = 'labeset' + str(labelset) + 'timeline' + str(timeline)
        if 'acc' in savename:
            df = pd.DataFrame(name, columns=columns, index=index)
            df.to_csv('{}.csv'.format(savename), mode='a', index_label=label_batch)
        elif savename == 'batch':
            df = pd.DataFrame(name, index=index)
            df.to_csv('{}.csv'.format(savename), mode='a', index_label=label_batch)
        elif savename == 'taskbatch':
            index = []
            for i in range(task_num):
                run_index = 'run' + str(i)
                index.append(run_index)
            df = pd.DataFrame(name, index=index)
            df.to_csv('{}.csv'.format(savename), mode='a', index_label=label_batch)
        elif savename == 'test_all':
            df = pd.DataFrame(name)
            df.to_csv('{}.csv'.format(savename), mode='a', index_label=label_batch)
        mean = np.mean(name)
        return mean
   

class Logger():
    def __init__(self, args, record_dir):
        super(Logger, self).__init__()
        
        self.logger = logging.getLogger()
        self.logger.setLevel(level=logging.INFO)
        # self.mode = args.mode
        # formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        # formatter = logging.Formatter('%(filename)s - [line:%(lineno)d]: %(message)s')
        formatter = logging.Formatter('%(message)s - [%(asctime)s]')


        file_handler = logging.FileHandler('{}/run.log'.format(record_dir))
        file_handler.setLevel(level=logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        template = '''Logger start up: 
        ----------------------------------------
        Method: \t{}
        Network: \t{}
        Dataset: \t{}
        Augments:'''.format(args.imp_method, args.network, args.dataset)
        for arg in vars(args):
            if len(arg) > 9:
                template += '\n\t  ├ {}: \t{}'.format(arg, getattr(args, arg))
            else:
                template += '\n\t  ├ {}: \t\t{}'.format(arg, getattr(args, arg))
                
        template += '\n\t----------------------------------------'

        self.logger.info(template)

    def SaveLog2File(self):
        pass
    
    def info(self, text):
        self.logger.info(text)

    def PrintResultPerTask(self, labelset_id, timeline_id, run_id, task_id, loss, acc, is_first=True):
        if is_first:
            template = '├ Run {}-{}-{}, Task {}, First Loss: {:.5f}, First Acc: {:.5f}'
        else:
            template = '├ Run {}-{}-{}, Task {}, Test Loss: {:.5f}, Test Acc: {:.5f}'
        self.info(template.format(labelset_id + 1, timeline_id + 1, run_id + 1, task_id, loss, acc))


    def PrintResultPerRun(self, labelset_id, timeline_id, run_id, first_acc, first_std, finish_acc, finish_std, forget_avg, forget_std):
        '''
        fn:  evaluate all tasks in a run
        '''
        template = '==> Run {}-{}-{}, Avg First: {:.3f} ± {:3f}, Avg Finish:{:.3f} ± {:.3f}, Avg Forget: {:.3f} ± {:3f}'
        self.info(template.format(labelset_id + 1, timeline_id + 1, run_id + 1, first_acc*100, first_std*100, finish_acc*100, finish_std*100, forget_avg*100, forget_std*100))
        print('----------------------------------------------------------------------')

    def PrintFinalResult(self, first_acc_avg, first_acc_std, finish_acc_avg, finish_acc_std, forget_avg, forget_std, record_dir):
        
        '''
        fn:  summarize all tasks first and finish shot acc.
        '''
        template = '''Final result across all labelset + timeline + run: 
        ----------------------------------------
        First Acc: \t{:.3f} ±{:.3f}
        Finish Acc: \t{:.3f} ±{:.3f}
        Forget: \t{:.3f} ±{:.3f}
        Record at : \t{}
        ----------------------------------------
        '''.format(first_acc_avg*100, first_acc_std*100, finish_acc_avg*100, finish_acc_std*100, forget_avg*100, forget_std*100, record_dir)        
        self.info(template)



def RecordAndLog2File(recorder, logger):
    forget_record = recorder.finish_acc_record - recorder.first_acc_record
    first_acc_std = []
    finish_acc_std = []
    forget_std = []
    for k in range(recorder.first_acc_record.shape[2]):
        first_acc_std.append(np.mean(recorder.first_acc_record[:,:,k]))
        finish_acc_std.append(np.mean(recorder.finish_acc_record[:,:,k]))
        forget_std.append(np.mean(recorder.forget_record[:,:,k]))

    first_acc_avg   = np.mean(recorder.first_acc_record)
    finish_acc_avg  = np.mean(recorder.finish_acc_record)
    forget_avg      = np.mean(forget_record)

    first_acc_std   = np.std(first_acc_std)
    finish_acc_std  = np.std(finish_acc_std)
    forget_std      = np.std(forget_std)
    #recorder.Record2File(1, 1, 2, recorder.args.task_num)
    recorder.Record2File(recorder.labelset_num, recorder.timeline_num, recorder.args.num_runs, recorder.args.task_num)
                                                                                                                
        
    record_dir = '{}/{}-{}-{}-{}-{:.0f}-{:.0f}'.format(recorder.args.record_dir, 
                                                       time.strftime('%Y%m%d%H%M'),
                                                       recorder.args.imp_method,
                                                       recorder.args.dataset,
                                                       recorder.args.with_mem,
                                                       first_acc_avg*100,
                                                       finish_acc_avg*100)

    logger.PrintFinalResult(first_acc_avg, first_acc_std, finish_acc_avg, finish_acc_std, forget_avg, forget_std, record_dir)
    
    # record the final results
    with open('{}/results.txt'.format(recorder.save_dir_cache), 'w') as f:
        f.write('First Acc: \t{:.5f} ±{:.5f}\nFinish Acc: \t{:.5f} ± {:.5f}\nForget: \t{:.5f} ±{:.5f}\nFor copy:\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}'.format(first_acc_avg*100, first_acc_std*100,     finish_acc_avg*100,finish_acc_std*100,forget_avg*100,forget_std*100, first_acc_avg*100, first_acc_std*100,     finish_acc_avg*100,finish_acc_std*100,forget_avg*100,forget_std*100))
    # remove the cache mark
    os.rename(recorder.save_dir_cache, record_dir)    

