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
from itertools import islice
import argparse
from pathlib import Path
import tqdm
import json
#ext
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import classification_report, top_k_accuracy_score, confusion_matrix
import numpy as np
# own
import PeTriBOX.utils.torch_utils as torch_utils
import PeTriBOX.model.models as models
import PeTriBOX.data.data as data
import PeTriBOX.test.tests as tests
from PeTriBOX.utils.python_utils import partition_keys_along_group_of_keys


class Tester(object):
    def __init__(
        self,
        dnn, # model pointer
        datasets_dict, # dataset dict
        test_collator,
        test_model,
        other_args,
        device='cpu', # torch device
    ):

        self.dnn = dnn 
        self.datasets_dict = datasets_dict
        self.device = device
        self.test_path = Path(other_args['root_path'], other_args['model_name'] , other_args['test_name'])
   
        self.task = other_args["task"]
        self.batch_size = other_args["batch_size"]
        self.max_batches = other_args["max_batches"]
        self.workers = other_args["workers"]
        self.accumulate_batches = other_args["accumulate_batches"]

        # computed
        self.testset = self.datasets_dict['test']
        self.tokenizer = self.testset.dataset.tokenizer
        self.testloader = torch.utils.data.DataLoader(
            self.testset, 
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=test_collator
        )
        # test model 
        self.collator = test_collator             
        self.test_model = test_model

        self.collator_post_processor = test_collator.post_process \
            if hasattr(self.collator, 'post_process') and callable(getattr(self.collator, 'post_process')) \
            else (lambda *x : x) 


    def test(self):
        # test handling
        if self.test_path is not None and Path(self.test_path, 'test.pt').exists():
            print("\t loading test data, test results are just read, not run")
            # TODO remove comment when json fixed
            #with open(Path(self.test_path, 'test.json'), 'r') as stream:
            #    test_args  = json.load(stream)
            test_results = torch.load(Path(self.test_path, 'test.pt'))
            pred_batches = []
        else: 
            test_results, pred_batches = self._test() # run test code
            if self.test_path is not None:
                Path(self.test_path).mkdir(exist_ok=True, parents=True)
                torch.save(test_results, Path(self.test_path, 'test.pt'))

        return test_results, pred_batches
    
    

    def _test(self):
        desc = "test"
        self.dnn.eval()   
        loader = self.testloader
        loader = islice(loader, self.max_batches) if self.max_batches else loader

        pbar = tqdm.tqdm(loader, desc=desc)
        pred_batches = []
        self.test_model.reset()
        with torch.no_grad():
            for item in pbar:
                # gather data
                x = {k : t.to(self.device) 
                    for k, t in item['inputs'].items()}
                y = item['labels'].to(self.device)
                # forward
                y_hat = self.dnn(**x)   

                # collator post process 
                x, y, y_hat = self.collator_post_processor(x, y, y_hat)    

                # detach 
                x = {k : t.detach().to('cpu') 
                    for k, t in x.items()}
                y = y.detach().to('cpu')
                y_hat = y_hat.detach().to('cpu')
                pred_batch = {'inputs':x, 'labels':y, 'estimates':y_hat }

                # perform test 
                self.test_model.update(pred_batch)

                # (optionnally) accumulate
                if self.accumulate_batches: 
                    pred_batches.append(pred_batch) 

            # final global operation on test data        
            test_results = self.test_model.post_process()

        return test_results, pred_batches
    

    # call plot function of test model
    def plot(self, test_results):
        self.test_model.plot(test_results)





def main():

    parser = argparse.ArgumentParser(description='pretrain ppi')
  
    
    UNSAVED_ARGS = ["data_path", "root_path", "model_name",  "raw_name",  "test_name", "disable_cuda", "accumulate_batches"]

    parser.add_argument('--data-path', type=str, nargs='+',
                        default=['/media/speckbull/data/prot3DBX_w_rot.pt'],
                        help='path of dataset') 
    parser.add_argument('--root-path', type=str, default = 'weights/',
                        help='folder path where trained model are stored')
    parser.add_argument('--model-name', type=str, default='PeTriBERT',
                        help='name of trained model')
    parser.add_argument('--raw-name', type=str, default='raw',
                        help='name of folder/file containing raw data')
    parser.add_argument('--test-name', type=str, default='test1',
                        help='name of model/folder containing test_data')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='Disable CUDA')
    parser.add_argument('--accumulate-batches', action='store_true', default=False,
                        help='Disable CUDA')
    
    
    RAW_ARGS = ["workers", "device_id", "batch_size", "max_batches",
        "split_and_use_test_split" , "subset_interval", "subset_mult", "random_trunk"]
    parser.add_argument('--workers', type=int, default=0,
                        help='numbers of workers')
    parser.add_argument('--device-id', type=int, default=0,
                        help='device id')                     
    parser.add_argument('--batch-size', type=int, default=8, # SOT 8192
                        help='batch size')
    parser.add_argument('--max-batches', type=int, default=None, 
                        help='batch size')
    parser.add_argument("--split-and-use-test-split", type=bool, default= True,
                        help="If true, split and use test split of input data using seed from loaded model" )
    parser.add_argument("--subset-interval", 
                        type=int,
                        default=None, 
                        nargs=2,
                        help='interval of test')
    parser.add_argument('--subset-mult', type=int,
                        default=3000,  # base, unique, sensitivity
                        help='multiplicity for unique and sensitivity tests')
    parser.add_argument("--random_trunk", type=bool, default= False,
                        help="If true, use trunkate function of dataset" )
    
    
    TEST_ARGS = ['test_collator_cls', 'test_model_cls']
    parser.add_argument("--test-collator-cls", type=str, )
    parser.add_argument("--test-model-cls", type=str, )
    
    args, _ = parser.parse_known_args()


    print("================================================================================================")
    print("===================================Starting testing script=====================================")
    print("================================================================================================")

    print("\n - partitionning of inputs")

    unsaved_args, raw_args, test_args = \
        partition_keys_along_group_of_keys( vars(args),  [UNSAVED_ARGS, RAW_ARGS, TEST_ARGS])
    

    print("\n - Setting and checking model to test")    
    checkpoint_path = Path(args.root_path, args.model_name)
    assert checkpoint_path.exists(), "train a model first"
    with open(Path(checkpoint_path, 'train.json'), 'r') as stream:
            train_args  = json.load(stream)

    print("\n - Checking if raw data or test data already exists")
    test_path = Path(checkpoint_path, unsaved_args["test_name"])
    test_path.mkdir(parents=True, exist_ok=True)

     # setting seed  
    print("\n - setting seed ")

    torch.manual_seed(train_args["seed"])
    np.random.seed(train_args["seed"])
    print(f"\t seed set to {train_args['seed']}")


    print("\t test data do not exist by that name, running test")

    # device config
    print("\n - device configuration")
    device_type = 'cuda' if not args.disable_cuda and torch.cuda.is_available() else 'cpu' 
    device = torch.device(device_type)
    if device_type == 'cuda':
        torch.cuda.set_device(args.device_id)
    print(f"\t - device type :  {device_type}") 
    
    # load model     
    dnn, model_args = models.load( 
        checkpoint_path=checkpoint_path, 
        device_type=device_type, 
        device_id=args.device_id
    )

    print("\n - Loading test dataset")

    dataset = data.Prot3DBDataset(
        file_path_list =unsaved_args["data_path"],
        fixed_len=model_args["seq_len"],
        random_trunk=raw_args["random_trunk"],
        global_rotation_augmentation=False,
        global_translation_augmentation=None,
        translation_noise_augmentation=None,
        rotation_noise_augmentation=None
    )
     # compute split path
    ## if only one file, take if as basename
    ## if several file, use folder name as base name (inside itself)
    split_path = torch_utils.compute_split_path(unsaved_args["data_path"], train_args["seed"])

    # load if split exists, otherwise create the split and save
    if split_path.exists():
        print(f"\n \t - loading split from {split_path}")
        datasets_dict = torch_utils.load_dataset_dict(dataset, split_path)
    else: 
        print(f"\n \t -  creating split from seed {train_args['seed']} and saving split to {split_path}")
        datasets_dict = torch_utils.split_dataset(dataset, train_args["split_rates"], seed=train_args["seed"])
        torch_utils.save_split(datasets_dict, split_path)
    
    # load test
    test_args.update({"tokenizer": dataset.tokenizer})
    test_collator, col_kwargs, test_model, test_model_kwargs = tests.load_test(kwargs=test_args)
    
    # trainer
    tester = Tester(
        dnn, # model pointer
        datasets_dict, # dataset dict
        test_collator = test_collator,
        test_model = test_model, # collate function/callable
        other_args= {
                "root_path" : unsaved_args['root_path'],
                "model_name" : unsaved_args['model_name'],
                "test_name" : unsaved_args['test_name'],
                "accumulate_batches":  unsaved_args['accumulate_batches'],
                "task" : model_args["task"],
                **raw_args,
        },
        device=device, # torch device
    )

    
    # run test
    test_results, pred_batches = tester.test()

    tester.plot(test_results)

    import pdb; pdb.set_trace()

if __name__=='__main__':
    main()
