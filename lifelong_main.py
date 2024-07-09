from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import gc
import time
import numpy as np
import tensorflow as tf

from datastream import ConstructDataStream, LoadLabelsetTimelineFromYAML, preload_dataset
from network import ConstructNetwork
from lifelong_trainer import Trainer
from lifelong_evaluator import Evaluator
from utils import SetGPU, SetFloatType, ArgParser, Logger, Recorder, RecordAndLog2File



# 1. Preparation
# 1.1 parse augments
args = ArgParser()
# 1.3 set GPU
SetGPU(args.gpu_id)
# 1.4 set float 64 for tf and np, reaulting in stable seeds
SetFloatType(64)

# --------------------------------------------------------------------------------
# 2. Load data split
# 2.1 Read Label set and Time line from .yml file
LabelSets, TimeLines, labelset_num, timeline_num = LoadLabelsetTimelineFromYAML(args.dataset, args.increment)
# 2.2 initialize recoder
recorder = Recorder(args, LabelSets, TimeLines)
# 1.2 initialize logger
logger = Logger(args, recorder.save_dir_cache)

# --------------------------------------------------------------------------------
# 3. Train for all labelsets, timelines and runs
#    Total run times: labelset_num * timeline_nums * num_runs

preload_data = preload_dataset(args.dataset)

loss_save = {}
batch_acc = {}
times = {}
for l, ls in enumerate(LabelSets): # label set num
   batch_acc[l] = {}
   for t, tl in enumerate(TimeLines[l]): # time line num
      batch_acc[l][t] = {}
      # for run_id in range(1): # run time
      for run_id in range(args.num_runs): # run time
         batch_acc[l][t][run_id] = {}
         logger.info(f"Start a new run on [ labelset {l+1}, timline {t+1} ], seed: {args.seed+run_id}")
         # 3.1 set a seed and begin a run
         np.random.seed(args.seed+run_id)
         tf.random.set_seed(args.seed+run_id)
         # 3.2 build datastream
         # logger.info("Construct dataset")
         DataStream = ConstructDataStream(dataset=args.dataset,
                                          task_num=args.task_num,
                                          batch_size=args.batch_size,
                                          mem_per_class=args.mem_per_class,
                                          with_mem=args.with_mem,
                                          preload_data = preload_data,
                                          labelset=LabelSets[l],
                                          timeline=TimeLines[l][t])
         # 3.3 build backbone network res18
         Model = ConstructNetwork(args.network, DataStream.TotalClass, args.imp_method)
         # 3.4 build Trainer and Evaluator
         LifelongTrainer = Trainer(Model, DataStream, args)
         LifelongEvaluator = Evaluator(Model, DataStream, args, recorder, logger)
         # 3.5 build progbar, if you want a progbar, set verbose to 1
         progbar = tf.keras.utils.Progbar(DataStream.TotalBatch, unit_name=f'Run {run_id+1}', verbose=0)
         # 3.6 train 
         time_cache = []
         # for i in range(400): # for each batch
         for i in range(DataStream.TotalBatch): # for each batch
            progbar.update(i+1)
            LifelongTrainer.InitializeBatch() # initialize for some specific methods
            start = time.perf_counter()
            LifelongTrainer.GetMemoryGradient(i) # compute gradient on memory
            LifelongTrainer.GetCurrentGradient(i) # compute gradients on current tasks
            LifelongTrainer.Update(args.imp_method, i) # real update
            end = time.perf_counter()
            # get gradients on each ongoing tasks
            LifelongEvaluator.EvaluateTask(labelset_id=l,
                                          timeline_id=t,
                                          run_id=run_id,
                                          batch_id=i)
            time_cache.append(end-start)
         times[DataStream.TotalBatch] = np.sum(time_cache)
         print(f'labelset {l+1}, timline {t+1} : {np.sum(time_cache)}')
         LifelongTrainer.reset_state()
         LifelongEvaluator.reset_state()
         del Model
         del DataStream
         del LifelongTrainer
         del LifelongEvaluator
         gc.collect()

print(times)   
RecordAndLog2File(recorder, logger)