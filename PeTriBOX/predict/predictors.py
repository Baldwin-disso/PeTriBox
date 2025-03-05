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

from PeTriBOX.model.models import load
from PeTriBOX.utils.python_utils import instanciate_from_mapping_and_kwargs
import PeTriBOX.data.tok as tok
import PeTriBOX.utils.bio_utils as butils
from PeTriBOX.data.data import convert_pkl_metric_dict
from PeTriBOX.predict.samplers import load_sampler
from PeTriBOX.predict.indices_iterators import load_indices_iterator
import torch
import torch.nn.functional as F
import json
import pickle
from pathlib import Path
import pandas as pd
import copy
from PeTriBOX.utils.bio_utils import PROTEIN_CHAIN_IDS

def load_predictor(predictor_kwargs=None, device_type = 'cpu', device_id=1 ):
    PREDICTOR_MAPPING = {
        'residue distribution':MonomerMaskedLogitsPredictor,
        #'residue prediction':MonomerNewResiduePredictor,
        'sequence prediction':MonomerNewSequencePredictor,
        'sequence prediction multi':MonomerNewSequencePredictorMult,
        'binder prediction multi':DimerNewSequencePredictorMult,
        'PPI prediction':PPIPredictor
    }
    # assert model checkpoint is in kwargs
    assert "predictor_cls_name" in predictor_kwargs, "predictor_cls_name should be in kwargs"
    assert "model_checkpoint" in predictor_kwargs, "model_checkpoint should be in kwargs" 


    # instanciate predictor
    predictor_cls_name = predictor_kwargs.pop('predictor_cls_name')
    predictor, predictor_kwargs = instanciate_from_mapping_and_kwargs(predictor_cls_name, predictor_kwargs, mapping=PREDICTOR_MAPPING )
    

    return predictor, {'predictor_cls_name':predictor_cls_name, **predictor_kwargs}


# base class, that loads model
class PredictorBase(object):
    def __init__(self, model_checkpoint, device_type='cpu', device_id=0):
        #device
        self.device_type = device_type
        self.device_id = device_id
        # load DNN
        self.dnn, _ = load(checkpoint_path=model_checkpoint, device_type=device_type, device_id=device_id)
        # set tokenizer with fixed len
        with open(Path(model_checkpoint, 'model.json' ), 'r') as stream:
            json_args = json.load(stream)
        self.seq_len = json_args["seq_len"]
        self.tokenizer = tok.SimpleTokenizer(fixed_len=self.seq_len)


class MonomerMaskedLogitsPredictor(PredictorBase):
    def __init__(self, model_checkpoint, device_type='cpu', device_id=0) -> None:
        super().__init__(model_checkpoint, device_type=device_type, device_id=device_id)

        
    def __call__(self, pdb_file_or_frames=None, chain='A' , seq=None, res_idxs = None):
        # get device
        device = torch.device('cpu' if self.device_type == 'cpu' else f'{self.device_type}:{self.device_id}')


        # build frame
        frames = butils.pdb_to_frames(pdb_file_or_frames)[chain] if isinstance(pdb_file_or_frames, str) else copy.deepcopy(pdb_file_or_frames)
        
        
        # tokenize
        if seq is not None: # override seq in frames by input sequence
            frames['seqs'] = seq
        frames['seqs'], frames['coords'], frames['rots'], = self.tokenizer.encode(
            frames['seqs'], coords=frames['coords'], rots=frames['rots'], 
            batchify=True
        )

        # add attention mask
        attention_mask =  torch.ones(frames['seqs'].shape, dtype=torch.bool) 
        attention_mask =  attention_mask & (frames['seqs'] != self.tokenizer.pad_idx) & (frames['seqs'] != self.tokenizer.end_idx)
        frames = {**frames,  'attention_mask': attention_mask}

        # send to device
        frames = {k:v.to(device) for k,v in frames.items()}  
       
        # mask
        if res_idxs:
            frames['seqs'][:,res_idxs] = self.tokenizer.mask_idx

        # infer 
        yhat = self.dnn.infer(**frames) 
        
        return yhat



class MonomerNewSequencePredictor(PredictorBase):
    def __init__(self, sampler_kwargs, indices_iterator_kwargs, model_checkpoint, device_type='cpu', device_id=0) -> None:
        super().__init__(model_checkpoint, device_type=device_type, device_id=device_id)
        self.sampler, _ = load_sampler(sampler_kwargs)
        self.indices_iterator, _ = load_indices_iterator(indices_iterator_kwargs)

        

    def __call__(self, pdb_file_or_frames=None, chains=['A'], seq=None, indices=None, pdb_out=None):
         # get device
        device = torch.device('cpu' if self.device_type == 'cpu' else f'{self.device_type}:{self.device_id}')

        # build frame
        if isinstance(pdb_file_or_frames, str):
            frames = butils.pdb_to_frames(pdb_file_or_frames)[chain]  
        else:
            frames = copy.deepcopy(pdb_file_or_frames)
        
        
        # tokenize
        if seq is not None: # override seq in frames by input sequence
            frames['seqs'] = seq
        frames['seqs'], frames['coords'], frames['rots'], = self.tokenizer.encode(
            frames['seqs'], coords=frames['coords'], rots=frames['rots'], 
            batchify=True
        )

        # add attention mask
        attention_mask =  torch.ones(frames['seqs'].shape, dtype=torch.bool) 
        attention_mask =  attention_mask & (frames['seqs'] != self.tokenizer.pad_idx) & (frames['seqs'] != self.tokenizer.end_idx)
        frames = {**frames,  'attention_mask': attention_mask}

        # send to device
        frames = {k:v.to(device) for k,v in frames.items()}  
       

        # start loop
        
        if indices is None:
            print("\t predicting on the whole sequence  ")
            _, indices = torch.nonzero(self.tokenizer.unpadded_tokens_mask(frames['seqs']), as_tuple=True)
            indices = indices.tolist()
            
        else: 
            indices = [i+1 for i in indices] # to skip start tokens


        
        for subindices in self.indices_iterator(indices):
            # mask sequence based on indices
            frames["seqs"][:,subindices] = self.tokenizer.mask_idx

            # infer
            yhat = self.dnn.infer(**frames)
            
            # sample 
            new_seq = self.sampler.sample(yhat, indices=subindices)

            # update 
            frames["seqs"][:,subindices] = new_seq

        

        frames["seqs"], frames["coords"],  frames["rots"] = \
            self.tokenizer.decode(frames["seqs"], coords = frames["coords"],  rots = frames["rots"])
        
        if pdb_out and isinstance(pdb_file_or_frames, str) :
            butils.brute_replace_seq_in_pdb(frames["seqs"], pdb_file_or_frames, out_pdb=pdb_out )

        return frames


class MonomerNewSequencePredictorMult(PredictorBase):
    def __init__(self, sampler_kwargs, indices_iterator_kwargs, model_checkpoint, device_type='cpu', device_id=0) -> None:
        super().__init__(model_checkpoint, device_type=device_type, device_id=device_id)
        self.sampler, _ = load_sampler(sampler_kwargs)
        self.indices_iterator, _ = load_indices_iterator(indices_iterator_kwargs)
        
    def __call__(self, pdb_files_or_frames=None, chain='A', seqs=None, indices=None, pdbs_out=None, draws = 1):
         # get device
        device = torch.device('cpu' if self.device_type == 'cpu' else f'{self.device_type}:{self.device_id}')

        # build frame
        if isinstance(pdb_files_or_frames, list) and  isinstance(pdb_files_or_frames[0], str):
            frames_list = []
            for pdb_file in pdb_files_or_frames:
                frames_list.append(butils.pdb_to_frames(pdb_file)[chain])
        else:
            frames_list = copy.deepcopy(pdb_files_or_frames)

        # TODO check size
        
        
        # tokenize
        if seqs is not None: # override seq in frames by input sequence
            for frames, seq in zip(frames_list, seqs): 
                frames['seqs'] = seq


        for frames in frames_list:
            frames['seqs'], frames['coords'], frames['rots'], = self.tokenizer.encode(
                frames['seqs'], coords=frames['coords'], rots=frames['rots'], 
                batchify=True
            )

            # add attention mask
            attention_mask =  torch.ones(frames['seqs'].shape, dtype=torch.bool) 
            attention_mask =  attention_mask & (frames['seqs'] != self.tokenizer.pad_idx) & (frames['seqs'] != self.tokenizer.end_idx)
            frames = {**frames,  'attention_mask': attention_mask}

        # batchify frames batch
        frames_batch = { k:[] for k in frames_list[0] }
        for f in frames_list:
            for k in frames_batch:
                frames_batch[k].append(f[k])
        frames_batch = {k:torch.cat(v) for k,v in frames_batch.items()}

       
        # manage indices to be processed
        if indices is None:
            print("\t predicting on the whole sequence  ")
            _, indices = torch.nonzero(self.tokenizer.unpadded_tokens_mask(frames_batch['seqs']), as_tuple=True)
            indices = indices.tolist()
            
        else: 
            indices = [i+1 for i in indices] # to skip start tokens


        
        # send data to device
        frames_batch = {k:v.to(device) for k,v in frames_batch.items()}  

        # prepare result holder : clone and make an interleave repeat of input
        frames_batch_out = copy.deepcopy(frames_batch)
        frames_batch_out = {k:v.repeat_interleave(draws,  dim=0) for k,v in frames_batch.items()}

        # loop on indices iterator
        for subindices in self.indices_iterator(indices):
            # create a temporary clone  (on same device)
            frames_batch_aux =  copy.deepcopy(frames_batch)

            # mask sequence based on indices
            frames_batch_aux["seqs"][:,subindices] = self.tokenizer.mask_idx

            # infer
            yhat = self.dnn.infer(**frames_batch_aux)
            
            # sample 
            new_seq = self.sampler.sample(yhat, indices=subindices, draws=draws)
            

            # update 
            frames_batch_out["seqs"][:,subindices] = new_seq

        # send back output and input to cpu
        frames_batch = {k:v.to('cpu') for k,v in frames_batch.items()}
        frames_batch_out = {k:v.to('cpu') for k,v in frames_batch_out.items()}


        #  unbatchify
        keys = frames_batch_out.keys()
        values = zip(*frames_batch_out.values()) 
        frames_list = [dict( zip(keys, v ))  for v in values  ]
                 
        #  token decoding
        for frames in frames_list: 
            frames["seqs"], frames["coords"],  frames["rots"] = \
                self.tokenizer.decode(frames["seqs"], coords=frames["coords"],  rots=frames["rots"])
        
        # TODO save as multiple pdbs
        if pdbs_out and isinstance(pdb_files_or_frames, list) and isinstance(pdb_files_or_frames[0], str) :
            assert len(pdbs_out) == (draws * len(pdb_files_or_frames))
            # repeat original file list
            pdbs_in = []
            for item in pdb_files_or_frames:
                pdbs_in.extend([item] * draws)

            # create files
            for frames, pdb_in, pdb_out,   in zip(frames_list, pdbs_in, pdbs_out):
                butils.brute_replace_seq_in_pdb(frames["seqs"], pdb_in, out_pdb=pdb_out )

        return frames_list




class DimerNewSequencePredictorMult(PredictorBase):
    def __init__(self, sampler_kwargs, indices_iterator_kwargs, model_checkpoint, device_type='cpu', device_id=0) -> None:
        super().__init__(model_checkpoint, device_type=device_type, device_id=device_id)
        self.sampler, _ = load_sampler(sampler_kwargs)
        self.indices_iterator, _ = load_indices_iterator(indices_iterator_kwargs)
        
    # seqs = [ {'A': seqA, 'B': seqB}  ]
    def __call__(self, pdb_files_or_frames=None, ligand_is_A=True, seqs=None, indices=None, pdbs_out=None, draws=1):

         # get device
        device = torch.device('cpu' if self.device_type == 'cpu' else f'{self.device_type}:{self.device_id}')

        # build frame
        if isinstance(pdb_files_or_frames, list) and  isinstance(pdb_files_or_frames[0], str):
            frames_list = []
            for pdb_file in pdb_files_or_frames:
                frames_list.append(butils.pdb_to_frames(pdb_file))
                
        else:
            frames_list = copy.deepcopy(pdb_files_or_frames)

        # TODO check size 
        
        
        # tokenize
        if seqs is not None: # override seq in frames by input sequence
            for frames, seq in zip(frames_list, seqs): 
                frames['A']['seqs'] = seq['A']
                frames['B']['seqs'] = seq['B']


        merged_frames_list= []        
        for frames in frames_list:
            merged_frames ={k:[] for k in frames['A'] }
            (merged_frames['seqs'], merged_frames['coords'], merged_frames['rots'], merged_frames['segment'] 
            )\
            = self.tokenizer.encode(
                frames['A']['seqs'], coords=frames['A']['coords'], rots=frames['A']['rots'], 
                seq2=frames['B']['seqs'], coords2=frames['B']['coords'], rots2=frames['B']['rots'],
                batchify=True,
                return_segment=True
            )

            # add attention mask
            attention_mask =  torch.ones(merged_frames['seqs'].shape, dtype=torch.bool) 
            attention_mask =  attention_mask & (merged_frames['seqs'] != self.tokenizer.pad_idx) & (merged_frames['seqs'] != self.tokenizer.end_idx)
            merged_frames = {**merged_frames,  'attention_mask': attention_mask}
            merged_frames_list.append(merged_frames)
            
        # batchify frames batch
        frames_batch = { k:[] for k in merged_frames_list[0] }
        for f in merged_frames_list:
            for k in frames_batch:
                frames_batch[k].append(f[k])
        frames_batch = {k:torch.cat(v) for k,v in frames_batch.items()}

       
        # manage indices to be processed
        if indices is None:
            print(f"\t predicting on the whole binder (chain {'A' if ligand_is_A else 'B'})  ")
            ligand_mask = (merged_frames['segment'] == 0) if ligand_is_A else (merged_frames['segment'] == 1)
            _, indices = torch.nonzero(ligand_mask, as_tuple=True)
            indices = indices.tolist()
            
        else: 
            indices = [i+1 for i in indices] # to skip start tokens


        
        # send data to device
        frames_batch = {k:v.to(device) for k,v in frames_batch.items()}  

        # prepare result holder : clone and make an interleave repeat of input
        frames_batch_out = copy.deepcopy(frames_batch)
        frames_batch_out = {k:v.repeat_interleave(draws,  dim=0) for k,v in frames_batch.items()}

        # loop on indices iterator
        for subindices in self.indices_iterator(indices):
            # create a temporary clone  (on same device)
            frames_batch_aux =  copy.deepcopy(frames_batch)
            frames_batch_aux.pop("segment") # HACKK

            # mask sequence based on indices
            frames_batch_aux["seqs"][:,subindices] = self.tokenizer.mask_idx

            # infer
            yhat = self.dnn.infer(**frames_batch_aux)
            
            # sample 
            new_seq = self.sampler.sample(yhat, indices=subindices, draws=draws)
            
            # update 
            frames_batch_out["seqs"][:,subindices] = new_seq

        # send back output and input to cpu
        frames_batch = {k:v.to('cpu') for k,v in frames_batch.items()}
        frames_batch_out = {k:v.to('cpu') for k,v in frames_batch_out.items()}

        
        #  unbatchify
        keys = frames_batch_out.keys()
        values = zip(*frames_batch_out.values()) 
        merged_frames_list = [dict( zip(keys, v ))  for v in values  ]
                 
        #  token decoding  
        frames_list = []
        for  merged_frames in  merged_frames_list: 
            frames = {'A':dict(), 'B':dict()} # new frame dict
            frames['A']["seqs"], frames['A']["coords"],  frames['A']["rots"], \
            frames['B']["seqs"], frames['B']["coords"],  frames['B']["rots"] = \
                self.tokenizer.decode(merged_frames["seqs"], coords = merged_frames["coords"],  rots = merged_frames["rots"], segment = merged_frames['segment'])
            frames_list.append(frames) # append new frames
        
        # save as multiple pdbs
        if pdbs_out and isinstance(pdb_files_or_frames, list) and isinstance(pdb_files_or_frames[0], str) :
            assert len(pdbs_out) == (draws * len(pdb_files_or_frames))
            # repeat original file list
            pdbs_in = []
            for item in pdb_files_or_frames:
                pdbs_in.extend([item] * draws)

            # create files # TODO brute replace
            for frames, pdb_in, pdb_out,   in zip(frames_list, pdbs_in, pdbs_out):
                butils.brute_replace_seq_in_pdb(frames['A']["seqs"] + frames['B']["seqs"], pdb_in, out_pdb=pdb_out )

        return frames_list




class PPIPredictor(PredictorBase):
    def __init__(self, model_checkpoint, device_type='cpu', device_id=0) -> None:
        super().__init__(model_checkpoint, device_type=device_type, device_id=device_id)
        self.tokenizer = tok.PPITokenizer(
            fixed_len=self.seq_len 
        )
        self.boosting = True
        
    # seqs = [ {'A': seqA, 'B': seqB}  ]
    def __call__(self, pkl_metrics_file_or_obj=None,  out_csv=None):
        device = torch.device('cpu' if self.device_type == 'cpu' else f'{self.device_type}:{self.device_id}')

  
        # load metrics and convert
        with open(pkl_metrics_file_or_obj, 'rb') as file:
                data = pickle.load(file)
        
        data, _, pred_inter = convert_pkl_metric_dict(data, pae_interaction_threshold=10.0, boosting=self.boosting, outputs_pred_inter=True)
        
        # tokenize
        x =  self.tokenizer.encode(data, batchify=True)

        # add attention mask
        
        attention_mask =  torch.ones(x['seqs'].shape, dtype=torch.bool) 
        attention_mask =  attention_mask & (x['seqs'] != self.tokenizer.pad_idx) & (x['seqs'] != self.tokenizer.end_idx)
        x = {**x,  'attention_mask': attention_mask}


        # send to device
        x = {k:v.to(device) for k,v in x.items() if v is not None}  
       
        # infer
        yhat = self.dnn.infer(**x)

        # compute probability  
        prob = F.softmax(yhat, dim=-1)[...,1].item()
        
        # retrive results, and fill csv if necessary
        if out_csv is not None:
            self.append_to_csv(out_csv, data, pred_inter, prob, separator = ';')        

        return prob
    
    def append_to_csv(self, csv_path, data, pred_inter, prob, separator = ';'):
        columns_subset = ['sq_ligand', 'sq_protein', 'ptm', 'iptm']
        petrippi_str = 'boosting' if self.boosting else 'prediction'

        # dict to save
        update_dict = {k:data[k] for k in columns_subset}
        # convert tensor
        update_dict.update({k:data[k].item() for k in update_dict if isinstance(data[k],torch.Tensor)})
        

        update_dict.update({
            'pae_interaction_prediction':pred_inter,
            f'PeTriPPI_{petrippi_str}_probability':prob, 
        })

       
        # Convert data to a DataFrame
        df_new = pd.DataFrame([update_dict])
        
        
        if Path(csv_path).exists():
            df_existing = pd.read_csv(csv_path, sep=separator)
            existing_columns = df_existing.columns.tolist()
            df_new = df_new.reindex(columns=existing_columns)
            df_new.to_csv(csv_path, mode='a', index=False, header=False, sep=separator)
        else:
            df_new.to_csv(csv_path, mode='w', index=False, header=True, sep=separator)



