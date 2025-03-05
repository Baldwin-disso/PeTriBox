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

from PeTriBOX.utils.python_utils import instanciate_from_mapping_and_kwargs
import torch.nn.functional as F
import torch
import PeTriBOX.data.tok as tok

def load_sampler( sampler_kwargs=None):
    SAMPLER_MAPPING = {
        'TemperatureSampler': TemperatureSampler,
        'NucleusSampler': NucleusSampler
    }
    # assert model checkpoint is in kwargs
    assert "sampler_cls_name" in sampler_kwargs, "sampler_cls_name should be in kwargs" 


    # instanciate predictor
    sampler_cls_name = sampler_kwargs.pop('sampler_cls_name')
    sampler, sampler_kwargs = instanciate_from_mapping_and_kwargs(
        sampler_cls_name, 
        sampler_kwargs, 
        mapping=SAMPLER_MAPPING
        )
    

    return sampler, {'sampler_cls_name':sampler_cls_name, **sampler_kwargs}



def draw_from_logits(logits, draws=1):
    probs = F.softmax(logits,dim=-1) # use softmax to compute probabilities
    return draw_from_probs(probs, draws=draws)


def draw_from_probs(probs, draws=1):
    cumul = probs.cumsum(dim=-1) 
    cumul = cumul.repeat_interleave(draws , dim=0)
    draw_seed = torch.rand(cumul.shape[:-1], device=probs.device).unsqueeze(-1)
    draw = (draw_seed > cumul).sum(dim=-1)
    return draw



class Sampler:
    def __init__(self):
        self.tokenizer = tok.SimpleTokenizer()

    def sample(self, logits, indices=None, draws=1, force_special_to_zero=True):
        pass 

    def prepare_logits(self, logits, indices, force_special_to_zero=True):
        # restrain to set of indices only
        logits = logits[:,indices] if indices is not None else None
        # force logits of special tokens to minus infinity
        if force_special_to_zero and logits is not None:
            logits[:,:,self.tokenizer.special_idx] = float('-inf')
        return logits 


class TemperatureSampler(Sampler):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def sample(self, logits, indices, draws=1, force_special_to_zero=True):
        # restrain logits to current indices and force special token to zero
        logits = super().prepare_logits(logits, indices, force_special_to_zero=force_special_to_zero)
        
        if logits is not None : 
            # divide logits using temperature
            logits = logits/self.temperature
            # draw from logits
            draw = draw_from_logits(logits, draws=draws)
        else: 
            draw = None
        return draw
        


class NucleusSampler(Sampler):
    def __init__(self, p=0.9):
        super().__init__()
        self.p = p

    
    def sample(self, logits, indices, draws=1, force_special_to_zero=True):
        logits = super().prepare_logits(logits, indices, force_special_to_zero=force_special_to_zero)
        probs = F.softmax(logits,dim=-1)
    
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
        mask = cumulative_probs <= self.p
    
        # allow to have true for the first False (and also for true)
        mask[..., 1:] = mask[..., :-1].clone() | mask[..., 1:]
        mask[..., 0] = True
        
        # apply mask
        selected_probs = sorted_probs * mask.float()
        
        # renormalize to obtain probabilities
        norm_probs = selected_probs / selected_probs.sum(dim=-1, keepdim=True)
        
        # create output
        nucleous_probs = torch.zeros_like(probs)
        
        # reassign probabilities to original indices
        nucleous_probs.scatter_(-1, sorted_indices, norm_probs)
        
        draw = draw_from_probs(nucleous_probs, draws=draws)
        return draw


        

