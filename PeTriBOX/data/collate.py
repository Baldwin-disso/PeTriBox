# Copyright 2025 - Lena Conesson, Baldwin Dumortier, Gabriel Krouk, Antoine Liutkus, Clément Carré  
#
# # Licensed to the Apache Software Foundation (ASF) under one
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

import torch
import math

class CollatorForLM(object):
    def __init__(self, 
        tokenizer,
        mlm_probability = 0.15,
    ):
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        print("\t LM collator created with mlm probability of {}".format(self.mlm_probability))

    def __call__(self, batch):
        t = self.tokenizer
        seqs = []
        coords = []
        rots = []
        # 1) stack batch into coords, inputs and labels
        for b in batch:
            seqs.append(b[0])
            coords.append(b[1])
            if b[2] is not None:
                rots.append(b[2])
        
        coords = torch.stack(coords)
        seqs = torch.stack(seqs)
        rots = torch.stack(rots) if bool(rots) else None
        labels = seqs.clone()
        
        
        # 2) attention_mask
        attention_mask =  torch.ones(labels.shape, dtype=torch.bool) # Full tensor of True values
        attention_mask = attention_mask & (labels != t.pad_idx) & (labels != t.end_idx)

        # 3) mask inputs and labels

        # token mask
        special_token_mask =  torch.zeros(labels.shape, dtype=torch.bool) # Full tensor of True values
        for l in t.special_idx :
            special_token_mask = special_token_mask | (labels == l) # put False where special token l is found

        # probability matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability) 
        probability_matrix.masked_fill_(special_token_mask, value=0.0) # put 0.0 as probability to location of special tokens
        masked_indices = torch.bernoulli(probability_matrix).bool() # define masked indexes based on probability

        # remove unmasked index from label to avoid gradient 
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        seqs[indices_replaced] = t.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(low=t.normal_idx[0], high=t.normal_idx[-1], size=labels.shape, dtype=torch.long)
        seqs[indices_random] = random_tokens[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        

        return {
            'inputs' : { 'seqs': seqs,  'coords': coords, 'rots':rots , 'attention_mask': attention_mask},
            'labels' : labels
        }
    
   

    
class CollatorForPPI(object):
    def __init__(self, 
        tokenizer,
    ):

        self.tokenizer = tokenizer


    def __call__(self, batch):
        
        t = self.tokenizer
        seqs = []
        coords = []
        rots = []
        plddt = []
        pae = []
        tm = []
        disto_log = []
        segment_size = []
        labels = []
    
        # 1) Stack batch into coords, inputs, and labels
        for b in batch:
            seqs.append(b['seqs'])
            coords.append(b['coords'])
            rots.append(b['rots'])
            plddt.append(b['plddt'])
            pae.append(b['pae'])
            tm.append(b['tm'])
            disto_log.append(b['disto_log'])
            segment_size.append(b['segment_size'])
            labels.append(b['label'])
        coords = torch.stack(coords)
        seqs = torch.stack(seqs)
        rots = torch.stack(rots)
        plddt = torch.stack(plddt).float()
        pae = torch.stack(pae)
        tm = torch.stack(tm).float()
        disto_log = torch.stack(disto_log)
        segment_size = torch.stack(segment_size)
        labels = torch.stack(labels)
        # 2) Attention mask
        attention_mask = torch.ones(seqs.shape, dtype=torch.bool) # Full tensor of True values
        attention_mask = attention_mask & (seqs != t.pad_idx) & (seqs != t.end_idx) # Mask padding and end tokens

        return {
            'inputs': {
                'seqs': seqs,
                'coords': coords,
                'rots': rots,
                'plddt': plddt,
                'pae': pae,
                'tm': tm,
                'disto_log': disto_log,
                'segment_size': segment_size,
                'attention_mask': attention_mask
            }, # Dict de tensors
            'labels': labels # Tensor
        }


class CollatorForUniqueProteinPrediction(object):
    def __init__(
        self, 
        tokenizer,
        mlm_probability
    ):
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        print("\t collator for unique protein prediction created")

    def __call__(self, batch):
        t = self.tokenizer
        # get data and set shape
        t = self.tokenizer
        seqs = []
        coords = []
        rots = []
        for b in batch:
            seqs.append(b[0])
            coords.append(b[1])
            if b[2] is not None:
                rots.append(b[2])
        coords = torch.stack(coords)
        inputs = torch.stack(seqs)
        rots = torch.stack(rots) if bool(rots) else None
        labels = inputs.clone()

        

        # compute sizes
        batch_size = inputs.shape[0] # number of duplicates
        seq_len = inputs.shape[1]
        prot_len = sum(sum((inputs[0]!=t.start_idx) & (inputs[0]!=t.pad_idx) & (inputs[0]!=t.end_idx))) # real protein length
        nb_focused = math.floor(self.mlm_probability * prot_len)

        

        #  attention_mask
        attention_mask =  torch.ones(inputs.shape, dtype=torch.bool) # Full tensor of True values
        attention_mask = attention_mask & (inputs != t.pad_idx) & (inputs != t.end_idx)

        for i in range(batch_size):
            start = min(1 + i*nb_focused, prot_len + 1 - nb_focused)
            end = min(1 + (i+1) * nb_focused, prot_len+1)
            labels[i, start:end] = inputs[i,start: end]
            inputs[i, start:end] = t.mask_idx
        
    
        return {
            'inputs' : { 'seqs': inputs,  'coords': coords, 'rots':rots , 'attention_mask': attention_mask},
            'labels' : labels
        }
    
    @staticmethod
    def post_process(inputs, labels, estimates):  
        bs = estimates.shape[0]
        res_seqs = inputs['seqs'].clone()
        res_estimates =  estimates[0, None].clone() # clone first line
        res_labels = labels[0,None].clone()
        res_coords = inputs['coords'][0][None].clone()
        res_rots =  inputs['rots'][0][None].clone() if rots is not None else None
       
        # loop on every "sample of the batch" and re-unite part where labels exist
        for b in range(bs):
            res_estimates[0, labels[b]!=-100 ] = estimates[b, labels[b]!=-100 ]
            res_labels[0, labels[b]!=-100 ] = labels[b, labels[b]!=-100 ]

        res_inputs = {'seqs': res_seqs,  'coords': res_coords, 'rots':res_rots}

        return res_inputs, labels, estimates

    


class CollatorForSensitivitySampling(object):
    def __init__(
        self, 
        tokenizer,
        mlm_probability,
        position_sigma=5,
        rotation_sigma= 0.1,
        batch_size_out = None
    ):
        self.mlm_probability = mlm_probability
        self.tokenizer = tokenizer
        self.batch_size_out = batch_size_out
        self.batch_simulated_locations = []
        self.position_sigma = position_sigma
        self.rotation_sigma = rotation_sigma
        self.mask_indices = None
        print("\t collator for Edge Sampling created")

    def __call__(self, batch):
        
        t = self.tokenizer
        seqs = []
        coords = []
        rots = []
        # 1) stack batch into coords, inputs and labels
        for b in batch:
            seqs.append(b[0])
            coords.append(b[1])
            if b[2] is not None:
                rots.append(b[2])
        coords = torch.stack(coords)
        inputs = torch.stack(seqs)
        rots = torch.stack(rots) if bool(rots) else None
        labels = inputs.clone()

        # compute sizes
        #import pdb; pdb.set_trace()
        seq_len = seqs[0].shape[1]
        
        prot_len = sum(sum((seqs[0]!=t.start_idx) & (seqs[0]!=t.pad_idx) & (seqs[0]!=t.end_idx)  )) # real protein length
        nb_focused = math.floor(self.mlm_probability * prot_len)

        #  attention_mask
        attention_mask =  torch.ones(inputs.shape, dtype=torch.bool) # Full tensor of True values
        attention_mask = attention_mask & (inputs != t.pad_idx) & (inputs != t.end_idx)


        # token mask
        special_token_mask =  torch.zeros(inputs.shape, dtype=torch.bool) # Full tensor of True values
        for l in t.special_idx :
            special_token_mask = special_token_mask | (labels[0] == l) # put False where special token l is found

        # probability matrix
        if self.mask_indices is None:
            probability_matrix = torch.full(inputs[0].shape, self.mlm_probability) 
            probability_matrix.masked_fill_(special_token_mask, value=0.0) # put 0.0 as probability to location of special tokens
            self.masked_indices = torch.bernoulli(probability_matrix).bool() # define masked indexes based on probability

        # remove unmasked index from label to avoid gradient 
        labels[:,~self.masked_indices] = -100

        # replace masked indiceds with masked tokens inputs
        inputs[:,self.masked_indices] = t.mask_idx

        # add random noise on coordinates for masked indexes
        position_draw_shape = coords[1:,self.masked_indices,...].shape
        
        position_draw = torch.normal(
            torch.zeros(position_draw_shape), 
            self.position_sigma*torch.ones(position_draw_shape)
        )
        coords[1:,self.masked_indices,...] += position_draw

        # add random noise on rotation parameters for masked indexes
        rotation_draw_shape = rots[1:,self.masked_indices,...].shape
        rotation_draw = torch.normal(
            torch.zeros(rotation_draw_shape), 
            self.rotation_sigma*torch.ones(rotation_draw_shape)
        )
        rots[1:,self.masked_indices,...] += rotation_draw


    
        return {
            'inputs' : { 'seqs': inputs,  'coords': coords, 'rots':rots , 'attention_mask': attention_mask},
            'labels' : labels
        }
    
  
class CollatorForGibbsSampling(object):
    def __init__(
        self, 
        tokenizer,
        focused_idx,
        angle_units=8,
    ):
        self.tokenizer = tokenizer
        self.focused_idx = focused_idx
        self.angle_units = angle_units
        self.sample_counter = 0
        print("\t collator for Gibbs Sampling created")

        # compute tensor of all rotations
        angles = 2*math.pi*torch.arange(0, 1, 1/angle_units) # spread angles on the unit circles
        angles_cos = torch.cos(angles)
        angles_sin = torch.sin(angles)
        c = torch.cartesian_prod(angles_cos,angles_cos,angles_cos)
        s = torch.cartesian_prod(angles_sin, angles_sin, angles_sin)
        self.replacement_rotations = torch.stack((c,s),dim=-1)

    def reset(self):
         self.sample_counter = 0

    def __call__(self, batch):
        t = self.tokenizer

        # 1) stack batch into coords, inputs, and rotations
        seqs = []
        coords = []
        rots = []
        for b in batch:
            seqs.append(b[0])
            coords.append(b[1])
            if b[2] is not None:
                rots.append(b[2])
        coords = torch.stack(coords)
        inputs = torch.stack(seqs)
        rots = torch.stack(rots) if bool(rots) else None
        batch_size = inputs.shape[0]

        
        # Pre-allocate inputs and label
        #import pdb; pdb.set_trace()
        labels = -100*torch.ones(inputs.shape)

        #  attention_mask
        attention_mask =  torch.ones(inputs.shape, dtype=torch.bool) # Full tensor of True values
        attention_mask = attention_mask & (inputs != t.pad_idx) & (inputs != t.end_idx)

        # remove unmasked index from label to avoid gradient 
        labels[:,self.focused_idx] = inputs[:,self.focused_idx]

        # replace masked indiceds with masked tokens inputs
        inputs[:,self.focused_idx] = t.mask_idx

        # rotations 
        rots[:,self.focused_idx] = self.replacement_rotations[
            self.sample_counter:self.sample_counter+batch_size
        ]
        #update counter 
        self.sample_counter = self.sample_counter + batch_size
        if self.sample_counter >= self.angle_units**3:
            self.reset()
       
    
        return {
            'inputs' : { 'seqs': inputs,  'coords': coords, 'rots':rots , 'attention_mask': attention_mask},
            'labels' : labels
        }
