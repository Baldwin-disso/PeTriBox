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


from pathlib import Path
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, GELU
import numpy as np
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import  FullMask, LengthMask
from PeTriBOX.model.biasedbert import  BiasedBERTEncoder, MultiBiasedBERTEncoder
from PeTriBOX.utils.bio_utils import uncompress_rotation_matrice
from PeTriBOX.utils.torch_utils import load_state_dict_soft
from PeTriBOX.model.embeddings import PeTriEmbedding, PeTriPOVEmbedding, PeTriPPIEmbedding
from PeTriBOX.utils.python_utils import  instanciate_from_mapping_and_kwargs





def load(kwargs=None, 
        checkpoint_path=None, 
        checkpoint_part_to_load = ["base", "head"],
        strict=True,
        device_type = 'cpu', 
        device_id=0, 
        distributed=False, 
    ):
    """
        Function to load the deep learning model and its weights.
        Note : 
        - when using kwargs, at least "model_cls_name" and "MLM" should be provided in kwargs.
        - when using checkpoint path, hyperparameters comes from model.json and kwargs is not used
        unless  "kwargs_override_checkpoint" is set
        - device_id and device type, designate if cuda should be used and on which device
        - distributed should be set to one to wrap model in DataDistributedParallel from pytorch
        - strict indicates if weights should exactly fit to the model, otherwise it allows partial loading of the weights
        - checkpoint_part_to_load, indicates which part of the weights of the model should be loaded 

    """
    # register models
    MODEL_MAPPING = {
        'PeTriBERT': PeTriBERT,
        'PeTriPOV': PeTriPOV,
        'PeTriMPOV': PeTriMPOV,
        'PeTriPPI': PeTriPPI,
    }
    # register task
    TASK_MAPPING = {
        'MLM': MLMHead,
        'ppi': PPIHead
    }

    # 0 handle inputs : 
    # handle checkpoint_path json file vs kwargs in arguments
    
    if checkpoint_path and Path(checkpoint_path, 'model.json').exists(): 
        
        with open(Path(checkpoint_path, 'model.json'), 'r') as stream:
            json_kwargs = json.load(stream)
        if kwargs:     
            json_kwargs.update(kwargs)
            print("\t Beware, kwargs are overriding kwargs from checkpoint")
           
        base_kwargs = dict(json_kwargs) # always use base from checkpoint 
        head_kwargs = dict(json_kwargs)# override json args with kwargs when finetuning
    elif kwargs: 
        print("\t loading model from kwargs, no checkpoint")
        base_kwargs = dict(kwargs)
        head_kwargs = dict(kwargs)
    else: 
        raise ValueError("Either 'kwargs' or 'checkpoint_path' with 'model.json' should exist")  # kwargs or checkpoint should exists
        

    model_cls_name = base_kwargs.pop('model_cls_name')
    task_name = head_kwargs.pop('task')
            


    # 1 device setting
    device_type = device_type if  device_type == 'cpu' or (device_type == 'cuda' and  torch.cuda.is_available()) else 'cpu' 

    if torch.cuda.is_available() and device_type == 'cuda':
        torch.cuda.set_device(device_id)
 
    device = torch.device(device_type if device_type == 'cpu' else f'cuda:{device_id}')

    
    # 2 model loading
    print(f"\t loading model: \n \t \t base model : {model_cls_name} \n \t \t task : {task_name}")
    
    base, base_kwargs = instanciate_from_mapping_and_kwargs(model_cls_name, base_kwargs, mapping=MODEL_MAPPING,)
    head, head_kwargs = instanciate_from_mapping_and_kwargs(task_name, head_kwargs, mapping=TASK_MAPPING)
    # redefine model kwargs from used kwargs 
    kwargs = {'model_cls_name' : model_cls_name, 'task' : task_name,**base_kwargs, **head_kwargs }
    # compose base and head
    dnn = PeTriComposite(base, head).to(device)


    # 3 handling case where model is distributed
    if distributed: 
        print("\t used distributed version of model")
        print(f"\t with device : {device_id} ")
        from torch.nn.parallel import DistributedDataParallel as DDP
        dnn = DDP(
            dnn, 
            device_ids=[device_id], 
            find_unused_parameters=True
        )
    
    # 4 handling checkpoint state loading
    if checkpoint_path:
        print(f"\t loading weights from {checkpoint_path} on {checkpoint_part_to_load} ")
        checkpoint = torch.load(Path(checkpoint_path, 'model.pt'), device)
        #resumepoint = torch.load(Path(checkpoint_path, 'resume.pt'), f'cuda:{device_id}')
        if distributed:  # handling model wrapper
            if "base" in checkpoint_part_to_load:
                load_state_dict_soft(dnn.module.base, checkpoint['base'], strict=strict)
            if "head" in checkpoint_part_to_load:
                load_state_dict_soft(dnn.module.head, checkpoint['head'], strict=strict)
        else:
            
            if "base" in checkpoint_part_to_load:
                load_state_dict_soft(dnn.base, checkpoint['base'], strict=strict)
            if "head" in checkpoint_part_to_load:
                load_state_dict_soft(dnn.head, checkpoint['head'] , strict=strict )
    return dnn, kwargs


# base module used to add specific function to pytorch basic modules
class BaseModule(nn.Module):
    def infer(self, *args, **kwargs):
        self.eval() 
        with torch.no_grad():
            output = self.forward(*args, **kwargs)  # Propagation avant
        return output
    
        
# This a class made to compose transformer and its head
class PeTriComposite(BaseModule):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head 
    
    def forward(self,*args,**kwargs):
        x = self.base(*args,**kwargs)
        x = self.head(x)
        return x
    
    def sample(self, *args, **kwargs):
        return self.sample(*args, **kwargs)



######################################
####    HEADS (Task) definition      
######################################  
class MLMHead(BaseModule):
    def __init__(
        self,
        d_model, 
        vocab_size,
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return x

class PPIHead(BaseModule):
    def __init__(self, 
        d_model
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, 2) #(768, 2)
    
    def forward(self, x):
        x = x[:,0]  # pooled output (1, 1024, 768) -> (1, 768)
        x = self.linear(x) # (1, 768) -> (1, 2)
        return x


##########################################
#### MODEL CLASS DEFINITIONS
#########################################

class PeTriBERT(BaseModule):
    def __init__(
        self, 
        vocab_size,
        seq_len,
        attention_type="full", 
        n_layers=5,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        point_dimensions=1,
        feed_forward_dimensions=3072,
        embedding_type = 'uni',
        rotation_embedding_type='normal',
        learnable_embedding_type = 'learnable_weights_and_MLP'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attention_type = attention_type
        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.point_dimensions = point_dimensions
        self.feed_forward_dimensions = feed_forward_dimensions

        assert n_layers >= 2 ; "layers should be greater than 2" 
        # 1 embedding and positionnal encoder

        self.embedding = PeTriEmbedding(
            vocab_size, 
            seq_len, 
            self.d_model, 
            embedding_type=embedding_type, 
            rotation_embedding_type=rotation_embedding_type,
            learnable_embedding_type = learnable_embedding_type
        )

        print('\t using  PeTriBERT model with {} attention'.format(attention_type))
        self.transformer = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            query_dimensions=self.query_dimensions,
            value_dimensions=self.value_dimensions,
            feed_forward_dimensions=self.feed_forward_dimensions,
            attention_type=attention_type,
            activation="gelu"
        ).get()

    def forward(
        self,
        seqs,
        coords=None, 
        rots=None, 
        segment_info=None,
        attention_mask=None, 
        ):
        x = self.embedding(
            seqs,
            coords=coords,
            rots=rots,
            segment_label=segment_info
            )
        if attention_mask is not None:   
            attention_mask = LengthMask(attention_mask.sum(dim=-1),max_len=self.seq_len) 
                    
        x = self.transformer(x, length_mask=attention_mask)     
        return x

    def forward_embedding(self, x, segment_info=None):
        return self.embedding(x, segment_info)

    @property
    def d_model(self):
        return self.n_heads * self.query_dimensions



class PeTriPOV(BaseModule):
    def __init__(
        self, 
        vocab_size,
        seq_len, 
        n_layers=5,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        point_dimensions=1,
        feed_forward_dimensions=3072,
        learnable_embedding_type='learnable_weights_and_MLP'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.point_dimensions = point_dimensions
        self.feed_forward_dimensions = feed_forward_dimensions
        self.learnable_embedding_type = learnable_embedding_type

        assert n_layers >= 2 ; "layers should be greater than 2" 

        # 1 embedding and positionnal encoder
        self.embedding = PeTriPOVEmbedding(
            vocab_size, 
            seq_len, 
            self.d_model, 
            n_heads=n_heads,
            learnable_embedding_type=self.learnable_embedding_type
        )
    
        # 2 transformer encoder
        self.transformer = BiasedBERTEncoder(
            hidden=self.n_heads*self.query_dimensions,
            n_layers=self.n_layers, 
            attn_heads=self.n_heads,
            dropout=0.1
        )
        

    def forward(
        self,
        seqs,
        coords=None, 
        rots=None, 
        attention_mask=None, 
        segment_info=None
        ):
        
        x, attention_bias = self.embedding(
            seqs,
            coords=coords,
            rots=rots,
            segment_label=segment_info
        )
        x = self.transformer(x, attention_bias=attention_bias, mask=attention_mask)  
        return x

    def forward_embedding(self, x, segment_info=None):
        return self.embedding(x, segment_info)

    @property
    def d_model(self):
        return self.n_heads * self.query_dimensions


class PeTriMPOV(BaseModule):
    def __init__(
        self, 
        vocab_size,
        seq_len, 
        n_layers=5,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        point_dimensions=1,
        feed_forward_dimensions=3072,
        learnable_embedding_type='learnable_weights_and_MLP',
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.point_dimensions = point_dimensions
        self.feed_forward_dimensions = feed_forward_dimensions
        self.learnable_embedding_type = learnable_embedding_type


        assert n_layers >= 2 ; "layers should be greater than 2" 

        # 1 embedding and positionnal encoder
        self.embedding = PeTriPOVEmbedding(
            vocab_size, 
            seq_len, 
            self.d_model, 
            n_heads=n_heads,
            learnable_embedding_type=self.learnable_embedding_type
        )
    
        # 2 transformer encoder
        self.transformer = MultiBiasedBERTEncoder(
            hidden=self.n_heads*self.query_dimensions,
            n_layers=self.n_layers, 
            attn_heads=self.n_heads,
            dropout=0.1)
        

    def forward(
        self,
        seqs,
        coords=None, 
        rots=None, 
        attention_mask=None, 
        segment_info=None
        ):
        
        x, attention_bias = self.embedding(
            seqs,
            coords=coords,
            rots=rots,
            segment_label=segment_info
        )
        x = self.transformer(x, attention_bias=attention_bias, mask=attention_mask)  
        return x

    def forward_embedding(self, x, segment_info=None):
        return self.embedding(x, segment_info)

    @property
    def d_model(self):
        return self.n_heads * self.query_dimensions



class PeTriPPI(BaseModule):
    def __init__(
        self, 
        vocab_size,
        seq_len, 
        n_layers=5,
        n_heads=12,
        query_dimensions=64,
        value_dimensions=64,
        point_dimensions=1,
        feed_forward_dimensions=3072,
        learnable_embedding_type='learnable_weights_and_MLP'
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.query_dimensions = query_dimensions
        self.value_dimensions = value_dimensions
        self.point_dimensions = point_dimensions
        self.feed_forward_dimensions = feed_forward_dimensions
        self.learnable_embedding_type = learnable_embedding_type

        assert n_layers >= 2 ; "layers should be greater than 2" 

        # 1 embedding and positionnal encoder
        self.embedding = PeTriPPIEmbedding(
            vocab_size, 
            seq_len, 
            self.d_model, 
            n_heads=n_heads,
            learnable_embedding_type=self.learnable_embedding_type
        )
    
        # 2 transformer encoder
        self.transformer = BiasedBERTEncoder(
            hidden=self.n_heads*self.query_dimensions,
            n_layers=self.n_layers, 
            attn_heads=self.n_heads,
            dropout=0.1)
        

    def forward(self, seqs, coords, rots, attention_mask, plddt, pae, disto_log, tm, segment_size):
        x, attention_bias = self.embedding(seqs, coords, rots, plddt, pae, disto_log, tm, segment_size)
        x = self.transformer(x, attention_bias=attention_bias, mask=attention_mask)
        return x

    def forward_embedding(self, x, segment_info=None):
        return self.embedding(x, segment_info)

    @property
    def d_model(self):
        return self.n_heads * self.query_dimensions