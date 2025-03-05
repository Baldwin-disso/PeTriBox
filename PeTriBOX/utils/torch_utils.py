# Copyright 2025 - Lena Conesson, Baldwin Dumortier, Gabriel Krouk, Antoine Liutkus, Clément Carré  
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.    

import pandas as pd 
import numpy as np 
import torch
import torch.utils.data
from torch.utils.data import Subset
from pathlib import Path
import argparse
from collections import OrderedDict
import json
import os
import math
from sklearn.metrics import accuracy_score
import random 
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from Bio import PDB
from Bio import SeqIO 
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqUtils



def filter_substatedict_w_submodule_name(state_dict, submodulename='transformer'):
    return  {k: v for k, v in state_dict.items() if k.startswith(submodulename)}

def map_state_dict_to_cpu(state_dict): # move state dict to GPU without loading the statedict to the correct device
    d = {}
    for k, v in state_dict.items():
        if isinstance(v,dict):
            d.update({k:map_state_dict_to_cpu(v)})
        elif isinstance(v,torch.Tensor):
            d.update({k:v.cpu()})
        else:
            d.update({k:v})
    return d



def load_state_dict_soft(model, state_dict, strict=False):
    """
    Load a state_dict into a model, handling extra and missing parameters.
    
    Args:
        model (nn.Module): The model to load the state_dict into.
        state_dict (dict): The state_dict containing pre-trained weights.
        strict (bool): If True, the function raises an error for missing or unexpected parameters.
                       If False, the function ignores missing or unexpected parameters.
    """
    model_state_dict = model.state_dict()
    
    # Create a new OrderedDict adapted to the model
    new_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].size() == value.size():
                new_state_dict[key] = value
            else:
                print(f"\t Ignoring parameter {key} due to size mismatch: model size {model_state_dict[key].size()}, state_dict size {value.size()}")
        else:
            print(f"\t Ignoring unexpected parameter {key} in state_dict.")
    
    # Load the adapted state_dict into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    # Check for missing parameters
    missing_keys = set(model_state_dict.keys()) - set(new_state_dict.keys())
    if missing_keys:
        print(f"\t Missing parameters in state_dict: {missing_keys}")
        if strict:
            raise RuntimeError(f"\t Missing parameters in state_dict: {missing_keys}")

    # Check for unexpected parameters
    unexpected_keys = set(state_dict.keys()) - set(model_state_dict.keys())
    if unexpected_keys:
        print(f"\t Unexpected parameters in state_dict: {unexpected_keys}")
        if strict:
            raise RuntimeError(f"\t Unexpected parameters in state_dict: {unexpected_keys}")

    print("\t State dict loaded successfully.")





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def cat_tensors_within_dict(x):
    r = {}  
    for k in x.keys():
        r.update({ k : torch.cat(x[k],dim=0)})
    return r 


def get_batch_accuracy(inputs, labels, estimates):

    # estimate : select values with max probability
    estimates_map = estimates.max(dim=2)[1] # [1] allows to select values and not indices 
    # original : rebuild original sequence from target y and estimates
    original = torch.where(labels!=-100, labels, inputs)
    # accumulation variables
    original_focused = []
    estimate_map_focused = []
    for i in range(original.shape[0]): # loop on samples of batch
        # compute focused only,  convert to list and append to data
        original_focused += (labels[i][torch.nonzero(labels[i]!=-100, as_tuple=True)]).detach().to('cpu').tolist()
        estimate_map_focused += (estimates_map[i][torch.nonzero(labels[i]!=-100, as_tuple=True)]).detach().to('cpu').tolist()

    return accuracy_score(original_focused, estimate_map_focused) 

def split_dataset(dataset, split_rates, seed=42):
    # splitting dataset
        train_len = math.floor(split_rates[0] * len(dataset) )
        valid_len =  math.floor(
            (split_rates[1]/(split_rates[1] + split_rates[2])) *  (len(dataset) - train_len) 
        )
        test_len = len(dataset) - (train_len + valid_len) 
        (trainset, validset, testset)  = torch.utils.data.random_split(dataset, [train_len, valid_len, test_len], generator=torch.Generator().manual_seed(seed))
        print("\t splitted  dataset")
        return  { 'train':trainset, 'valid':validset, 'test':testset }

def load_dataset_dict(dataset, split_path):
        split_indices = torch.load(split_path)
        return  {name: Subset(dataset, indices) for name, indices in split_indices.items()}


def save_split(dataset_dict, split_path): 
    torch.save({ name: value.indices for (name,value) in dataset_dict.items() }, split_path)

def compute_split_path(data_path_list, seed):
    
    if len(data_path_list) == 1 or Path(data_path_list[0]).is_dir() :
        split_base_path =  Path(data_path_list[0])
    elif Path(data_path_list[0]).is_file():
        split_base_path = Path(data_path_list[0]).parent / Path(data_path_list[0]).parent.name

    # generate split_path adding seed in the name
    split_path = split_base_path.with_name(split_base_path.stem + f'_split{seed}').with_suffix('.pt')

    return split_path


class SummaryOverWriter:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step_values = {}

    def add_scalar(self, tag, value, step):
        if step not in self.step_values:
            self.step_values[step] = value
        else:
            self.step_values[step] = value
        self.writer.add_scalar(tag, value, step)


class PeriodicLogger(object):
    def __init__(self,
        log_path,
        log_size,
        name = 'other'
    ):
        self.log_size = log_size
        self.writer = SummaryWriter(log_path)
        self.counter = 0
        self.name = name

    def log(self, log_value):
        self.counter += 1
        # if counter has reached period, log and reset counter
        if self.counter % self.log_size == self.log_size - 1:
            self.writer.add_scalar(
                self.name,
                log_value,
                self.counter
            )


class AccuracyLogger(object):
    def __init__(self,
        log_path,
        log_size,
        name = 'accuracy',
        counter = 0
    ):
        self.log_size = log_size
        self.writer = SummaryWriter(log_path)
        self.counter = counter
        self.name = name

    def log(self, inputs, labels, estimates):
        self.counter += 1
        # if counter has reached period, log and reset counter
        if self.counter % self.log_size == self.log_size - 1:
            acc = get_batch_accuracy(inputs, labels, estimates)
            self.writer.add_scalar(
                self.name,
                acc,
                self.counter
            )

class LossLogger(object):
    def __init__(self,
        log_path,
        log_size,
        loss_name = 'train',
        running_loss=0.0,
        counter=0, 
        device='cpu'
    ):
        self.log_size = log_size
        self.counter = counter
        self.writer = SummaryWriter(log_path)
        self.running_loss = torch.tensor(running_loss,dtype=torch.float32,device=device)
        self.loss_name = loss_name
        self.device = device

    def log(self, batch_loss):
        # store mean loss and update counter
        self.running_loss += batch_loss/self.log_size
        self.counter += 1
        # if counter has reached period, log and reset counter
        if self.counter % self.log_size == self.log_size - 1:
            self.writer.add_scalar(
                self.loss_name + ' loss',
                self.running_loss,
                self.counter
            )
            self.running_loss = torch.tensor(0.0,dtype=torch.float32, device = self.device)

    
class LRLogger(object):
    def __init__(self,
        log_path,
        log_size,
        running_lr=0.0,
        counter = 0,
    ):
        self.log_size = log_size
        self.counter = counter
        self.writer = SummaryWriter(log_path)
        self.running_lr = running_lr

    def log(self, running_lr):
        # store mean loss and update counter
        self.running_lr = running_lr
        self.counter += 1
        # if counter has reached period, log and reset counter
        if self.counter % self.log_size == self.log_size - 1:
            self.writer.add_scalar(
                'LR',
                self.running_lr,
                self.counter
            )

 
class TraceLogger(object):
    '''log string trace in file '''
    def __init__(self,
        log_id = 0, 
        folder_path = './'
    ):
        #import pdb; pdb.set_trace()
        self.log_path = Path(folder_path,'trace_{}_{}.log'.format(
            log_id,
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            )
        )
        self.log_path.touch(exist_ok=True) # create file if it does not exist
    
    def __call__(self, message):
        #HACK
        if True: # HACK
            pass #HACK
        else: #HACK : original code
            with open(self.log_path,mode='a') as f: # write
                f.write(datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S.%f --> ") + message + '\n')




        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class EarlyStopping(object):
    """Early Stopping Monitor"""

    def __init__(self, mode="min", min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if mode == "min":
            self.is_better = lambda a, best: a < best - min_delta
        if mode == "max":
            self.is_better = lambda a, best: a > best + min_delta


class freeze_scheduler(object):
    def __init__(
        self,
        model_parameters,
        freeze_offset=None
    ):
        self.model_parameters = model_parameters
        self.freeze_offset = freeze_offset

    def set_freeze_state(self, epoch):
        if self.freeze_offset is not None and epoch < self.freeze_offset:
            self.freeze()
        else :
            self.unfreeze()

    def freeze(self):
        for param in self.model_parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model_parameters():
            param.requires_grad = True
