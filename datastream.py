import math
import os
import glob
import re
from scipy import io as spio

import numpy as np
import tensorflow as tf
import yaml


def LoadLabelsetTimelineFromYAML(dataset, increment):

    task_icmt = {
        'CIFAR'     : 'labelset/CIFAR_task10_mean_128.yml',
        'EMNIST'    : 'labelset/EMNIST_task5_mean_128.yml',
        'TINY'      : 'labelset/TINY_task10_mean_32.yml'
    }

    class_icmt = {
        'CIFAR'     : 'labelset/CIFAR_task10_mean_128.yml',
        'EMNIST'    : 'labelset/EMNIST_task5_mean_128.yml',
        'TINY'      : 'labelset/TINY_task10_mean_32.yml'
    }

    free_icmt = {
        'CIFAR'     : 'labelset/CIFAR_free.yml',
        'EMNIST'    : 'labelset/EMNIST_free.yml',
        'TINY'      : 'labelset/TINY_free.yml'
    }

    if dataset in ['CIFARTOY', 'CIFAR', 'EMNIST', 'TINY']:
        if increment == 'task':
            yml = task_icmt[dataset]
        elif increment == 'class':
            yml = class_icmt[dataset]
        elif increment == 'free':
            yml = free_icmt[dataset]
    else:
        raise Exception("Only support EMNIST, CIFAR and TINY!")

    
    if os.path.exists(yml):
        yml = open(yml) 
        yml = yaml.load(yml, Loader=yaml.FullLoader)        
        print('\n==> .yml file loaded. Dataset {}: LabelSets and TimeLines will be fixed.'.format(dataset))
        LabelSets = [v['LabelSet'] for k, v in yml.items()]
        TimeLines = [v['TimeLines'] for k, v in yml.items()]
        labelset_num = len(LabelSets)
        timeline_num = [len(v['TimeLines']) for k,v in yml.items()]
    else: 
        raise Exception('\n==> Invalid .yml file.')

    return LabelSets, TimeLines, labelset_num, timeline_num


def load_filenames_labels(split):
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if split == 'train':
    filenames = glob.glob('../../dataset/tiny-imagenet-200/train/*/images/*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif split == 'val':
    with open('../../dataset/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = '../../dataset/tiny-imagenet-200/val/images/' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))

  return filenames_labels


def build_label_dicts():
  label_dict, class_description = {}, {}
  with open('../../dataset/tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('../../dataset/tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t')
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc
  return label_dict, class_description


def preload_dataset(dataset):
    # to avoid reload the dataset
    if dataset == 'EMNIST':
        emnist = spio.loadmat("/home/lvfan/dataset/emnist/emnist-byclass.mat")
        x_train = emnist["dataset"][0][0][0][0][0][0]
        x_train = x_train.astype(np.float32)
        # load training labels
        y_train = emnist["dataset"][0][0][0][0][0][1]
        
        # load test dataset
        x_test = emnist["dataset"][0][0][1][0][0][0]
        x_test = x_test.astype(np.float32)
        
        # load test labels
        y_test = emnist["dataset"][0][0][1][0][0][1]
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif dataset == 'CIFARTOY':
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif dataset == 'CIFAR':
        cifar = tf.keras.datasets.cifar100
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
    elif dataset == 'TINY':
        data_npz = '../../dataset/tiny-imagenet-200/tiny.npz'
        if os.path.exists(data_npz) == True:
            _data = np.load(data_npz)
            x_train, y_train, x_test, y_test = _data['arr_0'], _data['arr_1'], _data['arr_2'], _data['arr_3']
            print('npz data loaded.')
        else:
            train = load_filenames_labels('train')
            x_train, y_train = [], []
            for train_image in train:
                img = tf.io.read_file(train_image[0])
                img = tf.image.decode_jpeg(img, channels=3).numpy()
                x_train.append(img)
                label = train_image[1]
                label = tf.compat.v1.string_to_number(label, tf.int32).numpy()
                y_train.append(label)
            x_train = np.array(x_train)
            y_train = np.array(y_train)

            test = load_filenames_labels('val')
            x_test, y_test = [], []
            for test_image in test:
                img = tf.io.read_file(test_image[0])
                img = tf.image.decode_jpeg(img, channels=3).numpy()
                x_test.append(img)
                label = test_image[1]
                label = tf.compat.v1.string_to_number(label, tf.int32).numpy()
                y_test.append(label)
            x_test = np.array(x_test)
            y_test = np.array(y_test)
            x_train, x_test = x_train / 255.0, x_test / 255.0
            np.savez(data_npz, x_train, y_train, x_test, y_test)
    else:
        raise NotImplementedError
    return (x_train, y_train, x_test, y_test)

class ConstructDataStream:
    def __init__(self, dataset, task_num, batch_size, mem_per_class, with_mem, preload_data=None, labelset=None, timeline=None):
        super(ConstructDataStream, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.mem_per_class = mem_per_class
        self.TaskNum = task_num
        self.with_mem = with_mem
        self.preload_data = preload_data
        
        # For Rehearsal
        self.task_mem = []
        self.image_mem = {}
        self.label_mem = {}
        self.mask_mem = {}
        
        self.ConstructLabelSet(labelset) # 1. Construct label set
        self.ConstructDataset() # 2. Construct Dataset
        self.ConstructTimeLine(timeline) # 3. Construct TimeLine
        self.ConstructStream() # 4. Construct DatStream

    
    def ConstructLabelSet(self, labelset):
        if labelset == None: # if no given labelset, create one
            if self.dataset == 'CIFARTOY':
                self.LabelSet = [[0,1,2,3,4],[5,6,7,8,9]]
                self.TotalClass = 10
                self.ClassNum = [len(i) for i in self.LabelSet]
            elif self.dataset == 'CIFAR':
                self.TotalClass = 100
                LabelSet = list(range(self.TotalClass))
                np.random.shuffle(LabelSet)
                # Random Split
                self.ClassNum = []
                self.LabelSet = []
                sum_class = 0
                for i in range(self.TaskNum): # construct 20 ranInt, whoes sum is 100
                    self.ClassNum.append(np.random.randint(2, max(2, min(15, self.TotalClass-sum_class-2*(19-i)))))
                    sum_class = np.sum(self.ClassNum)
                np.random.shuffle(self.ClassNum)
                
                sum_class = 0
                for i in range(self.TaskNum): # Construct label set accroding to the classnum
                    self.LabelSet.append(LabelSet[sum_class:sum_class+self.ClassNum[i]])
                    sum_class+=self.ClassNum[i]
            elif self.dataset == 'TINY':
                self.TotalClass = 200
                LabelSet = list(range(self.TotalClass))
                np.random.shuffle(LabelSet)
                # Random Split
                self.ClassNum = []
                self.LabelSet = []
                sum_class = 0
                for i in range(self.TaskNum):  # construct 20 ranInt, whoes sum is 100
                    self.ClassNum.append(
                        np.random.randint(2, max(2, min(25, self.TotalClass - sum_class - 2 * (19 - i)))))
                    sum_class = np.sum(self.ClassNum)
                np.random.shuffle(self.ClassNum)

                sum_class = 0
                for i in range(self.TaskNum):  # Construct label set accroding to the classnum
                    self.LabelSet.append(LabelSet[sum_class:sum_class + self.ClassNum[i]])
                    sum_class += self.ClassNum[i]
        else:
            assert len(labelset) > 0, 'Invalid LabelSet!'
            self.LabelSet = labelset
            self.ClassNum = [len(i) for i in self.LabelSet]
            self.TotalClass = np.sum(self.ClassNum)

        assert self.TaskNum == len(self.LabelSet)
        self.mem_size = self.mem_per_class*self.TotalClass
        self.ref_mem_size = int(.5*self.mem_per_class*self.TotalClass)
        self.mix_mem_size = self.mem_size-self.ref_mem_size
    
    def ConstructDataset(self):
        if self.preload_data != None:
            x_train, y_train, x_test, y_test = self.preload_data
        else:
            x_train, y_train, x_test, y_test = preload_dataset(self.dataset)
        # Construct different datasets
        self.TrainSet, self.TestSet, self.MemSet = [], [], []
        self.MaskSet = np.zeros((len(self.LabelSet), self.TotalClass), dtype=np.float) # float64
        for i, labels in enumerate(self.LabelSet):
            x_train_task, y_train_task = [], []
            x_test_task , y_test_task  = [], []
            x_mem_task  , y_mem_task   = [], []
            for label in labels: # for each label
                self.MaskSet[i][label] = 1
                nbpick_train = np.where(y_train==label)
                x_train_task.append(x_train[nbpick_train[0]])
                y_train_task.append(y_train[nbpick_train[0]])
                nbpick_test = np.where(y_test==label)
                x_test_task.append(x_test[nbpick_test[0]])
                y_test_task.append(y_test[nbpick_test[0]])
                if self.with_mem == 'r':
                    sample_idx = np.random.choice(nbpick_train[0], size=self.mem_per_class, replace=False)
                else: # The max size of each class is the half of the total, because the smallest task contains 2/4 classes
                    if self.dataset == 'TINY':
                        # sample_num = min(int(self.mem_size / 4), len(nbpick_train[0]))
                        sample_num = min(int(self.mem_size / 4), len(nbpick_train[0]))
                        sample_idx = np.random.choice(nbpick_train[0], size=sample_num, replace=False)
                    else:
                        # sample_num = min(int(self.mem_size / 2), len(nbpick_train[0]))
                        sample_num = min(int(self.mem_size / 2), len(nbpick_train[0]))
                        sample_idx = np.random.choice(nbpick_train[0], size=sample_num, replace=False)
                x_mem_task.append(x_train[sample_idx])
                y_mem_task.append(y_train[sample_idx])
            # print("concat")
            x_train_task = np.concatenate(x_train_task, 0)
            y_train_task = np.concatenate(y_train_task, 0)
            x_test_task  = np.concatenate(x_test_task, 0)
            y_test_task  = np.concatenate(y_test_task, 0)
            # print("done")
            if self.with_mem == 'r':
                x_mem_task   = np.concatenate(x_mem_task, 0)
                y_mem_task   = np.concatenate(y_mem_task, 0)
           
            self.TrainSet.append((x_train_task, y_train_task))
            self.TestSet.append((x_test_task, y_test_task))
            self.MemSet.append((x_mem_task, y_mem_task) if self.with_mem=='r' else (x_mem_task, y_mem_task, self.MaskSet[i]))

    def ConstructTimeLine(self, timeline):
        if timeline == None:
            # if no timeline provied, randomly generate
            self.TimeLine = []
            e_max = 0
            for i in range(self.TaskNum):
                d = math.ceil(len(self.TrainSet[i][1])/self.batch_size)
                if len(self.TimeLine) == 0:
                    s = 0
                else:
                    s = np.random.randint(s, max(e+1, e_max+1))
                e = s + d - 1
                e_max = e if e > e_max else e_max
                self.TimeLine.append((s, e))
        else:
            self.TimeLine = timeline
        self.TotalBatch = 0
        for s, e in self.TimeLine:
            self.TotalBatch = e if e > self.TotalBatch else self.TotalBatch
        self.TotalBatch += 1
        print("Total batch num:", self.TotalBatch)

    def ConstructStream(self):
        self.TrainStream = [iter(tf.data.Dataset.from_tensor_slices(set).shuffle(len(set[1])).batch(self.batch_size)) for set in self.TrainSet]
        self.TestStream = [tf.data.Dataset.from_tensor_slices(set).batch(self.batch_size) for set in self.TestSet]
        if self.with_mem == 'r':
            self.MemStream = [iter(tf.data.Dataset.from_tensor_slices(set).shuffle(self.mem_per_class*self.TotalClass).batch(self.batch_size).repeat()) for set in self.MemSet]

    def UpdataeMemory(self, task, with_mem, batch_score=0, mem_split='ref', data_cache=None):
        if with_mem=='ur':
            # randomly sample data from the training set for this task
            x_mem_task, y_mem_task, mask_mem_task = [], [], []
            image_mem, label_mem, mask_mem = self.MemSet[task]
            for l in range(len(self.LabelSet[task])):
                sample_idx = np.random.choice(image_mem[l].shape[0], size=self.mem_per_class, replace=False)
                x_mem_task.append(image_mem[l][sample_idx])
                y_mem_task.append(label_mem[l][sample_idx])
                mask_mem_task.append(np.array([mask_mem]*self.mem_per_class))
            self.image_mem[task] = x_mem_task
            self.label_mem[task] = y_mem_task
            self.mask_mem[task] = mask_mem_task
            
            self.task_mem.append(task)
            image_mem_set, label_mem_set, mask_mem_set = [], [], []
            for t in self.task_mem:
                image_mem_set.append(np.concatenate(self.image_mem[t], 0))
                label_mem_set.append(np.eye(self.TotalClass)[np.concatenate(self.label_mem[t]).reshape(-1)])
                mask_mem_set.append(np.concatenate(self.mask_mem[t], 0))
            image_mem_set = np.concatenate(image_mem_set, 0)
            label_mem_set = np.concatenate(label_mem_set, 0)
            mask_mem_set = np.concatenate(mask_mem_set, 0)
            mem_set = (image_mem_set, label_mem_set, mask_mem_set)
            self.MemStream = iter(tf.data.Dataset.from_tensor_slices(mem_set).shuffle(self.mem_per_class*self.TotalClass).batch(self.batch_size).repeat())
            # print('==> Memory updated.')
        elif with_mem=='ur_reduce':
            # 1 How many memory is available for this task
            total_class = len(self.LabelSet[task])
            for t in self.task_mem:
                total_class += len(self.LabelSet[t])
            new_mem_per_class = self.mem_size//total_class

            # 2 randomly sample data from the training set for this task
            x_mem_task, y_mem_task, mask_mem_task = [], [], []
            image_mem, label_mem, mask_mem = self.MemSet[task]
            if new_mem_per_class > image_mem[0].shape[0]:
                new_mem_per_class = image_mem[0].shape[0]
            for l in range(len(self.LabelSet[task])):
                sample_idx = np.random.choice(image_mem[l].shape[0], size=int(new_mem_per_class), replace=False)
                x_mem_task.append(image_mem[l][sample_idx])
                y_mem_task.append(label_mem[l][sample_idx])
                mask_mem_task.append(np.array([mask_mem]*new_mem_per_class))
            self.image_mem[task] = x_mem_task
            self.label_mem[task] = y_mem_task
            self.mask_mem[task] = mask_mem_task

            # 3 remove old data
            for t in self.task_mem:
                for l in range(len(self.LabelSet[t])):
                    mem_sample = np.random.choice(self.image_mem[t][l].shape[0], size=int(new_mem_per_class), replace=False)
                    self.image_mem[t][l] = self.image_mem[t][l][mem_sample]
                    self.label_mem[t][l] = self.label_mem[t][l][mem_sample]
                    self.mask_mem[t][l] = self.mask_mem[t][l][mem_sample]
            
            self.mem_per_class = new_mem_per_class
            
            self.task_mem.append(task)
            image_mem_set, label_mem_set, mask_mem_set = [], [], []
            for t in self.task_mem:
                image_mem_set.append(np.concatenate(self.image_mem[t], 0))
                label_mem_set.append(np.eye(self.TotalClass)[np.concatenate(self.label_mem[t]).reshape(-1)])
                mask_mem_set.append(np.concatenate(self.mask_mem[t], 0))
            image_mem_set = np.concatenate(image_mem_set, 0)
            label_mem_set = np.concatenate(label_mem_set, 0)
            mask_mem_set = np.concatenate(mask_mem_set, 0)
            mem_set = (image_mem_set, label_mem_set, mask_mem_set)
            self.MemStream = iter(tf.data.Dataset.from_tensor_slices(mem_set).shuffle(self.mem_per_class*self.TotalClass).batch(self.batch_size).repeat())
            # print('==> Memory updated.')        
        else:
            raise Exception('Invalid memory type.')
    
    def UpdataeMemory2(self, task, with_mem, batch_score=0, mem_split='ref', data_cache=None):

        assert with_mem=='ur_reduce'
        
        # 1 How many memory is available for this task
        total_class = len(self.LabelSet[task])
        for t in self.task_mem:
            total_class += len(self.LabelSet[t])
        new_mem_per_class = self.mem_size//total_class

        # 2 randomly sample data from the training set for this task
        x_mem_task, y_mem_task, mask_mem_task = [], [], []
        image_mem, label_mem, mask_mem = self.MemSet[task]
        if new_mem_per_class > image_mem[0].shape[0]:
            new_mem_per_class = image_mem[0].shape[0]
        for l in range(len(self.LabelSet[task])):
            sample_idx = np.random.choice(image_mem[l].shape[0], size=int(new_mem_per_class), replace=False)
            x_mem_task.append(image_mem[l][sample_idx])
            y_mem_task.append(label_mem[l][sample_idx])
            mask_mem_task.append(np.array([mask_mem]*new_mem_per_class))
        self.image_mem[task] = x_mem_task
        self.label_mem[task] = y_mem_task
        self.mask_mem[task] = mask_mem_task

        # 3 remove old data
        for t in self.task_mem:
            for l in range(len(self.LabelSet[t])):
                mem_sample = np.random.choice(self.image_mem[t][l].shape[0], size=int(new_mem_per_class), replace=False)
                self.image_mem[t][l] = self.image_mem[t][l][mem_sample]
                self.label_mem[t][l] = self.label_mem[t][l][mem_sample]
                self.mask_mem[t][l] = self.mask_mem[t][l][mem_sample]
        
        self.mem_per_class = new_mem_per_class
        
        self.task_mem.append(task)
        image_mem_set, label_mem_set, mask_mem_set = [], [], []
        for t in self.task_mem:
            image_mem_set.append(np.concatenate(self.image_mem[t], 0))
            label_mem_set.append(np.eye(self.TotalClass)[np.concatenate(self.label_mem[t]).reshape(-1)])
            mask_mem_set.append(np.concatenate(self.mask_mem[t], 0))
        image_mem_set = np.concatenate(image_mem_set, 0)
        label_mem_set = np.concatenate(label_mem_set, 0)
        mask_mem_set = np.concatenate(mask_mem_set, 0)
        self.mem_set = (image_mem_set, label_mem_set, mask_mem_set)

    def PickABatchFromMem(self):
        idxs = list(range(self.mem_set[-1].shape[0]))
        batch_idx = np.random.choice(idxs, self.batch_size, replace=False)
        batch_image, batch_label, batch_mask = self.mem_set[0][batch_idx], self.mem_set[1][batch_idx], self.mem_set[2][batch_idx]
        return batch_image, batch_label, batch_mask, batch_idx


    def __len__(self):
        return len(self.ClassNum)

    def __stream_num__(self):
        return None
