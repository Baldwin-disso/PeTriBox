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

# base
import os
import json
import argparse
import copy
import tqdm
import math
from pathlib import Path
from itertools import islice

# ext
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from transformers import get_polynomial_decay_schedule_with_warmup 
from transformers import AdamW

# own 

import PeTriBOX.model.models as models
import PeTriBOX.utils.bio_utils as bio_utils
import PeTriBOX.data.tok as tok
import PeTriBOX.data.data as data
import PeTriBOX.data.collate as collate
import PeTriBOX.utils.torch_utils as torch_utils
import PeTriBOX.opt.criterions as criterions
from PeTriBOX.utils.python_utils import partition_keys_along_group_of_keys
from PeTriBOX.utils.torch_utils import LossLogger, AccuracyLogger, LRLogger, AverageMeter
from PeTriBOX.utils.torch_utils import map_state_dict_to_cpu
from PeTriBOX.utils.bio_utils import AMINO_ACIDS
             
class Trainer(object):
    def __init__(
        self,
        dnn, # model pointer
        datasets_dict, # dataset dict
        optimizer,
        scheduler,
        criterion,
        resume_state,    
        other_args,
        collator=None, # collate function/callable  
        device='cpu',
        dist_cfg=None # config of distribution if any
    ):
        

        # input gathering
        self.dnn = dnn
        self.datasets_dict = datasets_dict
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.criterion = criterion
        self.resume_state = resume_state
        self.device = device
        self.collator = collator

        self.task = other_args["task"]
        self.epochs = other_args["epochs"]
        self.subbatch_size = other_args["subbatch_size"]
        self.clip = other_args["clip"]
        self.max_batches = other_args["max_batches"]
        self.dist_cfg = dist_cfg
        self.workers = 0 if self.distributed else other_args["workers"]
        self.shuffle = other_args["shuffle"]

        # compute current epoch when resuming
        self.starting_epoch=len(resume_state["train_loss_history"]) 
           
        # set paths
        self.save_path = str(Path(other_args["root_path"], other_args["model_name"]))
        
        self.log_path = str(Path(other_args["root_path"],'runs',other_args["model_name"])) if other_args["log"] else None

        
        # train samplers, loaders and loggers 
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets_dict["train"],
            num_replicas=self.dist_cfg.size,
            rank=self.dist_cfg.rank,
            shuffle=self.shuffle
        ) if self.distributed else None

       
        self.train_loader = torch.utils.data.DataLoader(
            datasets_dict["train"], 
            batch_size=self.subbatch_size,
            num_workers=self.workers,
            collate_fn=self.collator,
            pin_memory=self.distributed,
            sampler = train_sampler
        )
        self.train_logger = LossLogger(
            self.log_path,
            other_args['log_steps'],
            running_loss= resume_state['loggers_train_loss'],
            loss_name = 'train',
            counter = resume_state['loggers_train_count'],
            device = self.device
        ) if self.log_path and (not self.distributed or self.dist_cfg.rank==0) else None

        self.lr_logger = LRLogger(
            self.log_path,
            other_args['log_steps'],
            running_lr=self.optimizer.param_groups[0]['lr'],
            counter = resume_state['loggers_train_count']
        ) if self.log_path and (not self.distributed or self.dist_cfg.rank==0) else None
    
        # valid samplers, loaders and loggers
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            datasets_dict["valid"],
            num_replicas=self.dist_cfg.size,
            rank=self.dist_cfg.rank,
            shuffle=self.shuffle
        ) if self.distributed else None
        self.valid_loader = torch.utils.data.DataLoader(
            datasets_dict["valid"],
            batch_size=self.subbatch_size,
            num_workers=self.workers,
            collate_fn = self.collator,
            pin_memory=self.distributed,
            sampler = valid_sampler
        )
        self.valid_logger = LossLogger(
            self.log_path,
            other_args['log_steps'],
            running_loss=resume_state['loggers_valid_loss'],
            loss_name = 'valid',
            counter = resume_state['loggers_valid_count'],
            device = self.device
        ) if self.log_path and (not self.distributed or self.dist_cfg.rank==0) else None
         
        self.accuracy_logger = AccuracyLogger(
            self.log_path,
            other_args['log_steps'],
            name = 'accuracy',
            counter = resume_state['loggers_valid_count']
        )  if self.log_path and\
            (not self.distributed or self.dist_cfg.rank==0) and\
            self.task == "MLM"\
            else None
        
    
    
    # run a whole epoch for training or validation
    def train_epoch(
        self,
    ):
        desc = "train"
        self.dnn.train()
        loader = self.train_loader
        loader = islice(loader, self.max_batches) if self.max_batches else loader
        loss_logger = self.train_logger
        lr_logger = self.lr_logger
        pbar = tqdm.tqdm(loader, desc=desc)
        losses = AverageMeter()
        for item in pbar:
            # gather data
            x = {k : t.to(self.device, non_blocking=self.distributed) 
                for k, t in item['inputs'].items()}
            y = item['labels'].to(self.device,non_blocking=self.distributed)
            # forward
            self.optimizer.zero_grad()
            y_hat = self.dnn(**x)
            loss = self.criterion(y_hat, y)
            # backward and opt
            loss.backward()
            self.optimizer.step() 
            if self.clip is not None: 
                nn.utils.clip_grad_norm_(self.dnn.parameters(), self.clip)
            self.scheduler.step()
            # logging
            with torch.no_grad():
                if loss_logger:
                    loss_logger.log(loss) # log loss
                if lr_logger:
                    lr_logger.log(self.optimizer.param_groups[0]['lr']) # log lr
                losses.update(loss, x[next(iter(x))].size(0))
        return losses.avg

        # run a whole epoch for training or validation
    def eval_epoch(
        self,
    ):
        desc = "valid"
        self.dnn.eval()   
        loss_logger = self.valid_logger
        loader = self.valid_loader
        loader = islice(loader, self.max_batches) if self.max_batches else loader

        pbar = tqdm.tqdm(loader, desc=desc)
        losses = AverageMeter()
        with torch.no_grad():
            for item in pbar:
                # gather data
                x = {k : t.to(self.device, non_blocking=self.distributed) 
                    for k, t in item['inputs'].items()}
                y = item['labels'].to(self.device,non_blocking=self.distributed)
                # forward
                y_hat = self.dnn(**x)       
                loss = self.criterion(y_hat, y) 
                # average loss
                if loss_logger:
                    loss_logger.log(loss)
                if self.task =="MLM" and self.accuracy_logger is not None:
                    self.accuracy_logger.log(inputs=x["seqs"], labels=y, estimates=y_hat)
                losses.update(loss, x[next(iter(x))].size(0))
        return losses.avg

    def train(self, save=True): 
        t = tqdm.trange(self.starting_epoch, self.epochs, disable=False)
        if self.starting_epoch == 0:
            valid_loss = self.eval_epoch()
            if self.distributed:
                dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM) 
        for epoch in t:  
            # train one epoch
            train_loss = self.train_epoch()
            if self.distributed:
                dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
            self.resume_state["train_loss_history"].append(train_loss.item())
            # valid one epoch
            valid_loss = self.eval_epoch()
            if self.distributed:
                dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM) 
            self.resume_state["valid_loss_history"].append(valid_loss.item())
            # save epoch if requested
            if save:
                self.save()

        

    def save(self):
        # save only if not distributed or good rank
        if not self.distributed or self.dist_cfg.rank == 0:
            # save model 
            base_state_dict = self.dnn.module.base.state_dict()  \
                if self.dist_cfg and self.dist_cfg.rank == 0 \
                else self.dnn.base.state_dict()
            head_state_dict = self.dnn.module.head.state_dict()  \
                if self.dist_cfg and self.dist_cfg.rank == 0 \
                else self.dnn.head.state_dict()
            
            torch.save( 
                {
                'base': map_state_dict_to_cpu(base_state_dict), 
                'head': map_state_dict_to_cpu(head_state_dict)
                },
                os.path.join(self.save_path, 'model.pt')
            )

            # save optimizer and scheduler states
            torch.save(
                {
                    'optimizer': map_state_dict_to_cpu(self.optimizer.state_dict()),
                    'scheduler': map_state_dict_to_cpu(self.scheduler.state_dict())
                },
                os.path.join(self.save_path, 'resume.pt')
            )


            # update logger states
            self.resume_state["loggers_train_count"] = self.train_logger.counter
            self.resume_state["loggers_valid_count"] = self.valid_logger.counter
            self.resume_state["loggers_train_loss"] = self.train_logger.running_loss.item()
            self.resume_state["loggers_valid_loss"] = self.valid_logger.running_loss.item()   
            # write
            with open(os.path.join(self.save_path, 'resume.json'), 'w') as outfile:
                outfile.write(json.dumps(self.resume_state, indent=4))

    @property 
    def distributed(self):
        return self.dist_cfg is not None
    




def main():
   
    parser = argparse.ArgumentParser(description='pretrain ppi')
    
  
    ### Unsaved args 
    UNSAVED_ARGS = ["data_path", "root_path", "model_name", "from_pretrained" , "resume", "disable_cuda"]
    parser.add_argument('--data-path', type=str, nargs='+',
                        help='path of dataset') 
    parser.add_argument('--root-path', type=str, default = './',
                        help='folder path where trained model are stored')
    parser.add_argument('--model-name', type=str, default='PeTriBERT',
                        help='name of trained model')
    parser.add_argument('--from-pretrained', type=str, default=None,
                        help='name of trained model')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training, bypass other arguments')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    
    # hyperparameters necessary for training (but not for model loading)
    GEN_ARGS = ["workers", "device_id", "seed", "log", "log_steps", "epochs", "batch_size", "max_batches"]
    # general 
    parser.add_argument('--workers', type=int, default=3,
                        help='numbers of workers')
    parser.add_argument('--device-id', type=int, default=0,
                        help='device id when using cuda ') 
    parser.add_argument('--seed', type=int, default=42,
                        help='seed of pseudo-random generation') 
    parser.add_argument('--log', action='store_true', default=False,
                        help='enable tensorboard logging')
    parser.add_argument('--log-steps', type=int, default=100,
                        help='number of batches before logging loss and saving model')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, # SOT 8192,  specki = 8
                        help='batch size')
    parser.add_argument('--max-batches', type=int, default=None, 
                        help='Debug option : maximum number of batches allowed in one epoch. ')
    

    DATA_ARGS = ["split_rates", "iter_max", "random_trunk",  
                "global_rotation_augmentation", "global_translation_augmentation", "translation_noise_augmentation", "rotation_noise_augmentation",
                "focused_rate", "shuffle"]
    # Dataset options                 
    parser.add_argument('--split-rates', type=float, default=[0.8, 0.1, 0.1],
                        nargs=3, help='dataset split rates')
    parser.add_argument('--iter-max', type=int, default=350000,
                        help='maximum number of samples to iter. Cached dataset only')
    parser.add_argument('--random-trunk', action='store_true', default=False,
                        help='Use random truncation when tokenizing if sequence is too big')              
    parser.add_argument('--global-rotation-augmentation', type=bool, default=True,
                        help='numbers of workers')
    parser.add_argument('--global-translation-augmentation', type=int, default=20,
                        help='numbers of workers')
    parser.add_argument('--translation-noise-augmentation', type=float, default=1.0, # average of minimal distance over sequences is 3.2
                        help='numbers of workers')    
    parser.add_argument('--rotation-noise-augmentation', type=float, default=10, # default correspond to sigma = 10 degree
                        help='numbers of workers')   
    parser.add_argument('--focused-rate', type=float, default=0.15,  # SOT 0.15
                        help='rate of focused element for masking procedure')
    parser.add_argument('--shuffle', action='store_true', default=False, 
                        help='shuffle dataset')
    
    
    OPT_ARGS = ["optimizer", "criterion", "clip", "init_lr", "end_lr", "betas", "weight_decay",
                "epsilon", "warmup", "ending", "ppi_boosting", "boosting_threshold", "loss_ponderation" ]
    # optimisation hyperparameters
    parser.add_argument('--optimizer', type = str, default='adamw',
                        help='adamw (CPU or GPU) or lamb (GPU only)') # adamw or lamb
    parser.add_argument('--criterion', type = str, default='CrossEntropyLoss',
                        help='loss function used for training') # adamw or lamb
    parser.add_argument('--clip', type=float, default=1.0, 
                        help='clip gradient value') 
    parser.add_argument('--init-lr', type=float, default=1e-3, # SMALL =  1e-3, BASE = 2.5e-3
                        help='initial learning rate, defaults to 1e-3')
    parser.add_argument('--end-lr', type=float, default=1e-7,
                        help='end lr, default to 1e-7')
    parser.add_argument('--betas', type=float, nargs= 2,  default=[0.9, 0.999],
                        help='beta1, default 0.9')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='beta1, default 0.9')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='epsilon') 
    parser.add_argument('--warmup', type=int, default=30000, # fixed for batch-size
                        help='warmup steps, defautl 30000')
    parser.add_argument('--ending', type=int, default=250000, # SOT 100000
                        help='warmup steps, defautl 220000')
    parser.add_argument('--ppi-boosting', action='store_true', default=False, 
                        help='enable boosting in ppi case')
    parser.add_argument('--boosting-threshold', type=int, default=10, 
                        help='boosting threshold in ppi case ')
    parser.add_argument('--loss-ponderation', type=float, nargs= 2, default=None, 
                        help='Ponderation loss (ppi boosting only) ')
    

    ### MODEL specicif parameters (necessary for loading the model)
    MODEL_ARGS = ["model_cls_name", "task","seq_len", "attention_type", "n_layers", "n_heads", "n_heads_bias", 
                    "query_dimensions", "value_dimensions", "point_dimensions",  "feed_forward_dimensions",
                    "embedding_type", "rotation_embedding_type",  "learnable_embedding_type", "neighbors"]
    MODEL_COMPUTED_ARGS = ["d_model", "vocab_size"] # computed first time
    MODEL_ARGS =   MODEL_ARGS +  MODEL_COMPUTED_ARGS          
    parser.add_argument('--model-cls-name', type=str, default='PeTriBERT', 
                        help='name of trained model')    
    parser.add_argument('--task', type=str, default='MLM', 
                        help='name of trained model')
    parser.add_argument('--attention-type', type=str, default='full', 
                        help='name of trained model') 
    parser.add_argument('--seq-len', type=int, default=1024, 
                        help='sequence lenght')        
    parser.add_argument('--n-layers', type=int, default=5,  # SMALL = 5, BASE = 12
                        help='number of layers')
    parser.add_argument('--n-heads', type=int, default=12,  # SOT 12 (d_model = heads * query = 768)
                        help='number of heads')
    parser.add_argument('--n-heads-bias', type=int, default=4,  # SOT 12 (d_model = heads * query = 768)
                        help='number of heads')
    parser.add_argument('--query-dimensions', type=int, default=64,  # SOT 64
                        help='query dimensions')
    parser.add_argument('--value-dimensions', type=int, default=64,  # SOT 64
                        help='value dimensions')
    parser.add_argument('--point-dimensions', type=int, default=1,  # SOT 1
                        help='point dimensions for Invariant point transformer')
    parser.add_argument('--feed-forward-dimensions', type=int, default=3072,  # SOT 3072
                        help='feed forward dimension')

    parser.add_argument('--embedding-type', type=str, default='uni', # none, uni, tri or unitri 
                        help='enable tensorboard logging')
    parser.add_argument('--rotation-embedding-type', type=str, default='none', # 'none', 'normal', 'dummy'
                        help='enable tensorboard logging')                  
    parser.add_argument('--learnable-embedding-type', type=str, default='learnable_weights_and_MLP',
                        help='enable tensorboard logging')


    args, _ = parser.parse_known_args() 




    # ARGS and STATEs    
    """
    ARGS=(UNSAVED,SAVED=(MODEL,TRAIN=(SYS,DATA,OPT)))
    
    """
    print("================================================================================================")
    print("===================================Starting training script=====================================")
    print("================================================================================================")

    TRAIN_ARGS = GEN_ARGS + DATA_ARGS + OPT_ARGS 
    SAVE_ARGS = TRAIN_ARGS + MODEL_ARGS
    
    print("\n - partitionning of input script")

    unsaved_args, save_args = \
        partition_keys_along_group_of_keys( vars(args),  [UNSAVED_ARGS, SAVE_ARGS ])
    resume_state = {}

    

    # SLURM ARRAY management : override args and model name with array args
    if 'SLURM_ARRAY_TASK_COUNT' in os.environ : # condition to see if slurm_array is used
        print("\n - slurm array is used, using slurm_array.json for parameter grid search ")
        slurm_array_task_id=int(os.environ["SLURM_ARRAY_TASK_ID"])
        # model name override
        unsaved_args.update({'model_name': (unsaved_args['model_name'] + '_array' + str(slurm_array_task_id))}) # override model name

        # input args override
        with open(Path('slurm_array.json'), 'r') as stream:
            json_array_args = json.load(stream)
        json_task_args = json_array_args[str(slurm_array_task_id)]
        # update 
        save_args.update( {k : json_task_args[k]  for k in json_task_args if k in save_args} )

    # model folder creation
    output_path = Path(unsaved_args['root_path'], unsaved_args['model_name'])
    output_path.mkdir(parents=True, exist_ok=True)

    

    # Load or create kwargs and resume states 
    if unsaved_args["resume"]:

        ## 1) override train_args with train.json content
        print('\n - resuming training : loading kwargs and resume state')
        # load json
        with open(Path(output_path, 'train.json'), 'r') as stream:
            json_train_args = json.load(stream)
        print('\t train.json loaded')

        # load json
        with open(Path(output_path, 'model.json'), 'r') as stream:
            json_model_args = json.load(stream)
        print('\t model.json loaded')
        

        json_args = {**json_model_args, **json_train_args }

        # manage args
        print(f'''\t Note : The following parameters are never loaded from the json checkpoint and are instead taken from the input args : {unsaved_args}''')

        # display parameters of the code that are not present in checkpoint (for retro-compatibility issues)
        not_in_chkpnt_list = [ k for k in save_args.keys() if k not in json_args.keys()  ]
        if bool(not_in_chkpnt_list):
            print('\n \t WARNING : when resuming,\
            the following parameters were not found \
            in the json checkpoint and therefore \
            are taken from the input args : {}'.format(not_in_chkpnt_list))

        save_args.update(json_args)

        ## 2) load pretrain_state
        # load json
        with open(Path(output_path, 'resume.json'), 'r') as stream:
            resume_state = json.load(stream)
        print('\t resume state loaded')
    else: 
        # 2) resume state creation
        print('\n - new resume state creation')
        resume_state['train_loss_history'] = []
        resume_state['valid_loss_history'] = []
        resume_state['loggers_train_count'] = 0
        resume_state['loggers_valid_count'] = 0
        resume_state['loggers_train_loss'] = 0.0
        resume_state['loggers_valid_loss'] = 0.0

    # split save_args into train, model, etc
    model_args, gen_args, data_args, opt_args = \
        partition_keys_along_group_of_keys( save_args,  [MODEL_ARGS, GEN_ARGS, DATA_ARGS, OPT_ARGS ])


    # print config
    print("\n - Using following configuration :")

    print("\t system args")
    for item in gen_args.items():
        print("\t \t - {} : {} ".format(item[0], item[1]))
 
    print("\t data args")
    for item in data_args.items():
        print("\t \t  - {} : {} ".format(item[0], item[1]))
    
    print("\t opt args")
    for item in opt_args.items():
        print("\t \t - {} : {} ".format(item[0], item[1]))
    
    print("\t model args")
    for item in model_args.items():
        print("\t \t - {} : {} ".format(item[0], item[1]))


    # setting seed  
    print("\n - setting seed ")
    torch.manual_seed(gen_args["seed"])
    np.random.seed(gen_args["seed"])
    print(f"\t seed set to {gen_args['seed']}")
    

    print("\n - setting device")
    device = None
    if "SLURM_JOB_ID" not in os.environ: # regular
        distributed = False
        if not unsaved_args["disable_cuda"] and torch.cuda.is_available():
            device_type = 'cuda'
            device_id = gen_args["device_id"]
            torch.cuda.set_device(device_id)
            device = torch.device('cuda')
        else:
            device_type = 'cpu'
            device_id = None
            device = torch.device('cpu')
        dist_cfg = None
        subbatch_size = gen_args["batch_size"] 
    else: # distributed
        print("\t Using distributed Model")
        distributed = True
        import idr_torch
        import torch.distributed as dist    
        dist.init_process_group(backend='nccl', 
                                init_method='env://', 
                                world_size=idr_torch.size, 
                                rank=idr_torch.rank)
        gen_args['device_id'] = idr_torch.local_rank
        device_id = gen_args['device_id']
        torch.cuda.set_device(device_id)
        device = torch.device("cuda")
        device_type = 'cuda'
        dist_cfg = idr_torch
        subbatch_size = gen_args["batch_size"] // idr_torch.size 
    print(f"\t used device type {device_type}")
    if device_type=="cuda":
        print(f"\t used device id {device_id}")

    

    # TODO use a specific loader function to load different types of dataset and collator if necessary
    print("\n - dataset loading ")


    if model_args["task"] == "ppi":
        folder_path = unsaved_args["data_path"][0]
        lut_file_path =  unsaved_args["data_path"][1]
        dataset = data.HDF5Dataset(
            hdf5_dir = folder_path,
            lut_file_path = lut_file_path,
            fixed_len = model_args["seq_len"],
            random_trunk = data_args["random_trunk"], 
            boosting = opt_args["ppi_boosting"],
            boosting_threshold = opt_args["boosting_threshold"]
        )
    else :
        dataset = data.Prot3DBDataset(
            file_path_list=unsaved_args["data_path"],
            fixed_len = model_args["seq_len"],
            random_trunk = data_args["random_trunk"],
            global_rotation_augmentation=data_args["global_rotation_augmentation"],
            global_translation_augmentation=data_args["global_translation_augmentation"],
            translation_noise_augmentation=data_args["translation_noise_augmentation"],
            rotation_noise_augmentation= data_args["rotation_noise_augmentation"]
        )
    # compute split path
    ## if only one file, take if as basename
    ## if several file, use folder name as base name (inside itself)
  

    split_path = torch_utils.compute_split_path(unsaved_args["data_path"], gen_args["seed"])
    # load if split exists, otherwise create the split and save
    if split_path.exists():
        print(f"\n \t - loading split from {split_path}")
        datasets_dict = torch_utils.load_dataset_dict(dataset, split_path)
    else: 
        print(f"\n \t - creating split from seed {gen_args['seed']} and saving split to {split_path}")
        datasets_dict = torch_utils.split_dataset(dataset, data_args["split_rates"], seed=gen_args["seed"])
        torch_utils.save_split(datasets_dict, split_path)
    
    
    # collate
    print("\n - instanciate collator")
    if model_args["task"] == "ppi" :
        data_collator = collate.CollatorForPPI(
            dataset.tokenizer,
        )
    else:
        data_collator = collate.CollatorForLM(
            dataset.tokenizer,
            mlm_probability = data_args["focused_rate"]
        )
    



    # model loading
    print("\n - loading model (base and head)")
    if not 'd_model' in model_args: model_args.update({'d_model' : model_args['n_heads'] *  model_args['query_dimensions']})
    if not 'vocab_size' in model_args: model_args.update({'vocab_size':len(dataset.tokenizer.all_tokens)})
    # case where finetuning starts but is not resumed
    if bool(unsaved_args["from_pretrained"]) and not unsaved_args["resume"]:
        override_kwargs = model_args
        checkpoint_path = Path(unsaved_args['root_path'], unsaved_args["from_pretrained"])
        checkpoint_part_to_load = ['base'] # this avoid to load head weights
        strict = False # this allow partial loading when finetuning
    # case for resume (either pretraining of finetuning)
    elif  unsaved_args["resume"]:
        override_kwargs = None
        checkpoint_path = output_path
        checkpoint_part_to_load = ['base', 'head']
        strict = True
    # case starting from scratch
    else:
        override_kwargs = model_args
        checkpoint_path = None 
        checkpoint_part_to_load = ['base', 'head']
        strict = True

   

     
    dnn, model_args = models.load(
        kwargs=override_kwargs, 
        checkpoint_path=checkpoint_path, 
        checkpoint_part_to_load = checkpoint_part_to_load,
        strict=strict,
        device_type=device_type, 
        device_id=device_id, 
        distributed=distributed
    )

    
    print('\t model has {} trainable parameters'.format(torch_utils.count_parameters(dnn)))

    print("\n - loading optimizer and scheduler")
    if args.optimizer == 'adamw' :  
        optimizer = AdamW(
            dnn.parameters(),
            lr=opt_args["init_lr"],
            betas=opt_args["betas"],
            weight_decay=opt_args["weight_decay"],
            correct_bias=False
        )
       

    # scheduler
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer, 
        opt_args["warmup"], 
        opt_args["ending"], 
        lr_end=opt_args["end_lr"], 
        power=1.0, 
        last_epoch=-1
        )
    
    # loss
    if opt_args["criterion"] == "CrossEntropyLoss":     
        
        class_weights = torch.tensor(opt_args['loss_ponderation'], dtype=torch.float32, device=device)\
            if opt_args['loss_ponderation'] is not None \
            else None
        criterion = criterions.CrossEntropyLossFlat(ignore_index=-100)        
    else:
        print("loss not implemented")
    

    # loading weights
    if args.resume:
        print("\t resuming optimizer and scheduler states")
        checkpoint = torch.load(Path(output_path, 'resume.pt'), device) 
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
 

    # trainer
    print("\n - initialising trainer")
    trainer = Trainer(
        dnn, # model pointer
        datasets_dict, # dataset dict
        optimizer,
        scheduler,
        criterion,
        resume_state,    
        collator=data_collator, # collate function/callable
        other_args= {
            "root_path" : unsaved_args["root_path"], 
            "model_name": unsaved_args["model_name"],
            "task": model_args["task"], 
            "subbatch_size": subbatch_size,
            "shuffle": data_args["shuffle"],
            **gen_args, 
            **opt_args  
            },
        device=device, # torch device
        dist_cfg=dist_cfg # config object for distribution
    )

    # save args before training necessary
    
    if not unsaved_args["resume"]:
        print("\n - saving train and model kwargs (this is done only when starting a new training from scratch)")
        with open(os.path.join(output_path, 'train.json'), 'w') as outfile:
                outfile.write(json.dumps( {**gen_args, **data_args, **opt_args},
                indent=4
            ))

        with open(os.path.join(output_path, 'model.json'), 'w') as outfile:
                outfile.write(json.dumps(model_args, indent=4))


    # train
    print("\n - training")
    print("\t starting training")
    trainer.train()
    print("\t end of training")

if __name__=='__main__':
    main()
