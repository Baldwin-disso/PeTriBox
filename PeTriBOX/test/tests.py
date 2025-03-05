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

import PeTriBOX.data.tok as tok
import PeTriBOX.data.collate as collate
import PeTriBOX.opt.criterions as criterions
from PeTriBOX.utils.python_utils import instanciate_from_mapping_and_kwargs
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, top_k_accuracy_score
import torch.nn.functional as F
import numpy as np




def load_test(test_collator_cls = None, test_model_cls = None,  kwargs=None):
    TEST_COLLATOR_MAPPING = {
        'Collatorforlm' : collate.CollatorForLM
    }
    TEST_MODEL_MAPPING = {
        'LMreport' : LMReportTest
    }

    kwargs = dict(kwargs) # create a shwallow copy of kwargs (to avoid "popping" on a global level)
   
    # handle 'model' and 'task' in kwargs or not  
    if 'test_collator_cls' in kwargs :
        if test_collator_cls is not None : 
            print("\t test_collator_cls is overrided by test_collator_clsfound in kwargs")
        test_collator_cls = kwargs.pop('test_collator_cls')
    
    if 'test_model_cls' in kwargs :
        if test_model_cls is not None : 
            print("\t test_model_cls is overrided by test_model_cls found in kwargs")
        test_model_cls = kwargs.pop('test_model_cls')
            

    # 2 model loading
    print(f"\t loading test collator: \n \t \t base model : {test_collator_cls} \n \t \t test model : {test_model_cls}")
    
    test_collator, col_kwargs = instanciate_from_mapping_and_kwargs(test_collator_cls, kwargs, mapping =TEST_COLLATOR_MAPPING )
    test_model, model_kwargs = instanciate_from_mapping_and_kwargs(test_model_cls, kwargs, mapping =TEST_MODEL_MAPPING)
   

    return test_collator, col_kwargs, test_model, model_kwargs



    
class LMReportTest(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.num_classes = self.tokenizer.vocab_size
        self.reset()

    # call when instanciating and/or at the beggining of loop
    def reset(self):
        # Initialize counters and accumulators for metrics
        self.cross_entropy_accumulator = 0
        self.accuracy_accumulator = 0
        #self.top3_accuracy_accumulator = 0
        self.precision_accumulator = 0
        self.recall_accumulator = 0
        self.f1_accumulator = 0
        
        
        self.class_precisions = np.zeros(self.num_classes)
        self.class_recalls = np.zeros(self.num_classes)
        self.class_f1s = np.zeros(self.num_classes)
        self.class_supports = np.zeros(self.num_classes)

        self.num_samples = 0
        self.num_batches = 0

        self.final_results = None

    # called on every batch within dataloader loop
    def update(self, batch):
        
        y_true, y_pred, logits, nb_valid_samples = self._process_batch(batch)
        self._update(y_true, y_pred, logits, nb_valid_samples)


    # called at the end of test loop
    def post_process(self):
        """
        Compute the average precision, recall, and F1 score across all batches.
        """
        avg_cross_entropy = self.cross_entropy_accumulator / self.num_batches
        avg_accuracy = self.accuracy_accumulator / self.num_samples
        #avg_top3_accuracy = self.top3_accuracy_accumulator / self.num_samples

        avg_precision = self.precision_accumulator / self.num_samples
        avg_recall = self.recall_accumulator / self.num_samples
        avg_f1 = self.f1_accumulator / self.num_samples
        
        # Compute per-class precision, recall, and F1 score
        avg_class_precision = self.class_precisions / self.class_supports
        avg_class_recall = self.class_recalls / self.class_supports
        avg_class_f1 = self.class_f1s / self.class_supports

        
        self.final_result = {
            "cross_entropy": avg_cross_entropy,
            "accuracy": avg_accuracy,
            #"top3_accuracy": avg_top3_accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "class_precision": avg_class_precision.tolist(),
            "class_recall": avg_class_recall.tolist(),
            "class_f1": avg_class_f1.tolist(),
        }
        return self.final_result

    # call to display results
    def plot(self, final_results):
        if final_results is not None:
            self._plot(final_results)
        else:
            self._plot(self.final_results)


    ## private functions

    # actual plotting function
    def _plot(self, final_results):
        print(final_results)

    # main part of update method
    def _update(self, y_true, y_pred, logits, nb_valid_samples):
        """
        Update precision, recall, F1 score, accuracy, top-3 accuracy, and cross-entropy loss for the current batch.
        """
        #### Step 1 compute metrics
        # Cross-entropy loss
        crossentropy_loss = F.cross_entropy(
            torch.tensor(logits), torch.tensor(y_true), reduction='mean'
        ).item()

        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Top-3 Accuracy
        #import pdb; pdb.set_trace()
        #top3_accuracy = top_k_accuracy_score(y_true, logits, k=3)

        # precision recall f1
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # per class
        precision_per_class = precision_score(y_true, y_pred, labels=range(self.num_classes), average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=range(self.num_classes), average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=range(self.num_classes), average=None, zero_division=0)

        # Compute support (number of true samples for each class)
        support_per_class = np.bincount(y_true, minlength=self.num_classes)

       

        ### step 2 accumulate metrics
        self.cross_entropy_accumulator += crossentropy_loss
        self.accuracy_accumulator += accuracy * nb_valid_samples
        #self.top3_accuracy_accumulator += top3_accuracy * nb_valid_samples

        self.precision_accumulator += precision * nb_valid_samples
        self.recall_accumulator += recall * nb_valid_samples
        self.f1_accumulator += f1 * nb_valid_samples
 
        
        self.class_precisions += precision_per_class * support_per_class
        self.class_recalls += recall_per_class * support_per_class
        self.class_f1s += f1_per_class * support_per_class
        self.class_supports += support_per_class

        #step 3 update counters
        self.num_samples +=  nb_valid_samples
        self.num_batches += 1

     

    # other part of update method  
    def _process_batch(self, batch):
        # Extract true labels and predictions from the batch
        y_true = batch["labels"].cpu().numpy()
        logits = batch["estimates"].cpu().numpy()
        y_pred = np.argmax(logits, axis=2) # argmax along dimension of estimation

        # Apply the focus mask to filter out padding tokens (-100)
        mask = (y_true != -100)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # compute number of valid sample for the batch
        nb_valid_samples = mask.sum()

        # Mask logits to only include valid tokens
        # Use broadcasting to ensure the mask is applied to all logits
        mask = mask[..., None]  # Add a new dimension for logits
        logits = logits[mask.squeeze(-1)]

        return y_true, y_pred, logits, nb_valid_samples

