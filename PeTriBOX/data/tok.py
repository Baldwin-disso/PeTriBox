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

import torch
import math
from PeTriBOX.utils.bio_utils import AMINO_ACIDS
import  PeTriBOX.utils.bio_utils as butils



class SimpleTokenizer(object):
    def __init__(self,
        fixed_len = None, # if not None define the lenght of the sequence
        random_trunk = False,
        use_separator=False
    ):
        self.normal_tokens = AMINO_ACIDS 
        self.start_token = '['; self.sep_token = '|'; self.pad_token = '-'; self.end_token = ']'; self.mask_token = '?'  
        self.special_tokens = [ self.start_token, self.sep_token ,self.pad_token, self.end_token, self.mask_token ]

        self.fixed_len = fixed_len
        self.random_trunk = random_trunk
        self.use_separator=use_separator
                
        print("\t tokenizer created with fixed len : {}".format(self.fixed_len))

    @property
    def all_tokens(self):
        return [*self.special_tokens, *self.normal_tokens]
    
    @property
    def vocab_size(self):
        return len(self.all_tokens)
    
    @property
    def token_to_idx_dict(self):
        #import pdb; pdb.set_trace()
        return { t:i for i, t in enumerate(self.all_tokens)  }
    
    def token_to_idx(self, sequence, tensorize = False, add_batch_dim=False): 
        #import pdb; pdb.set_trace()
        seq =  [self.token_to_idx_dict[e] for e in sequence]
        if tensorize: 
            seq = torch.tensor(seq, dtype=torch.long) 
            if add_batch_dim:
                seq = seq[None]
        return seq
        
    @property
    def idx_to_token_dict(self):
        return {i:t for i, t in enumerate(self.all_tokens)}
    
    def idx_to_token(self, sequence):
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist() 
        return ''.join([self.idx_to_token_dict[e] for  e in sequence])
    

    # idx properties
    @property
    def special_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.special_tokens ]

    @property
    def normal_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.normal_tokens ]

    @property
    def all_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.all_tokens ]


    @property
    def start_idx(self):
        return self.token_to_idx_dict[self.start_token]

    @property
    def sep_idx(self):
        return self.token_to_idx_dict[self.sep_token]

    @property
    def pad_idx(self):
        return self.token_to_idx_dict[self.pad_token]
    
    @property
    def end_idx(self):
        return self.token_to_idx_dict[self.end_token]
    
    @property
    def mask_idx(self):
        return self.token_to_idx_dict[self.mask_token]



    def encode(self, seq, coords=None, rots=None, seq2=None, coords2=None, rots2=None, batchify=False, return_segment=False ): 
    
        # 1 manage type of sequences and encode
        #import pdb; pdb.set_trace()
        if isinstance(seq, torch.Tensor):
            # raw to str 
            seq = butils.rawidx_to_aa(seq.squeeze())
            seq = self.token_to_idx(seq, tensorize=True)
        elif isinstance(seq, str):
            seq = self.token_to_idx(seq, tensorize=True) 
        else:
            raise ValueError("seq must be string or tensor")
        
        
        if isinstance(seq2, torch.Tensor):
            # raw to str 
            seq2 = butils.rawidx_to_aa(seq2.squeeze())
            seq2 = self.token_to_idx(seq2, tensorize=True)
        elif isinstance(seq2, str):
            seq2= self.token_to_idx(seq2, tensorize=True) 
        elif seq2 is None: 
            seq2 = None
            
        
        coords = coords.squeeze() if coords is not None else None
        coords2 = coords2.squeeze() if coords2 is not None else None
        rots = rots.squeeze() if rots is not None else None
        rots2 = rots2.squeeze() if rots2 is not None else None

        # segment 
        seg = torch.zeros(seq.shape)
        seg2 = torch.ones(seq2.shape) if seq2 is not None else None


        # 2 add special tokens and truncate
        big_seq  = torch.cat((torch.tensor([self.start_idx]), seq   )) 
        big_seg  = torch.cat((torch.tensor([2]), seg   )) 
        big_coords = torch.cat((torch.zeros(1,3), coords)) if coords is not None else None
        big_rots = torch.cat((torch.zeros(1,3), rots)) if rots is not None else None
        if seq2 is not None and  self.use_separator: # optional : add separator
            big_seq = torch.cat((big_seq,  torch.tensor([self.sep_idx])  ))
            big_seg = torch.cat((big_seg,  torch.tensor([2])  ))
            big_coords = torch.cat((big_coords, torch.zeros(1,3) )) if coords2 is not None else None
            big_rots = torch.cat((big_rots, torch.zeros(1,3))) if rots2 is not None else None
        if seq2 is not None: # add 2 sequence
            big_seq = torch.cat((big_seq,  torch.tensor([self.sep_idx]), seq2  ))
            big_seg = torch.cat((big_seg,  torch.tensor([2]), seg2  ))
            big_coords = torch.cat((big_coords, torch.zeros(1,3), coords2 )) if coords2 is not None else None
            big_rots = torch.cat((big_rots, torch.zeros(1,3), rots2)) if rots2 is not None else None
        big_seq = torch.cat((big_seq, torch.tensor([self.end_idx])))
        big_seg = torch.cat((big_seg, torch.tensor([2])))
        big_coords = torch.cat((big_coords, torch.zeros(1,3))) if big_coords is not None else None
        big_rots = torch.cat((big_rots, torch.zeros(1,3))) if big_rots is not None else None

        seq_len = seq.shape[0] + 2  # size of non truncated encoded seq 
        big_len = big_seq.shape[0]        

        # Truncate if necessary
        if self.fixed_len: # truncate or pad
             
            enc_seq = self.pad_idx * torch.ones(self.fixed_len, dtype=torch.long)
            enc_seg = 2 * torch.ones(self.fixed_len, dtype=torch.long)
            enc_coords =  torch.zeros(self.fixed_len,3) if big_coords is not None else None
            enc_rots =  torch.zeros(self.fixed_len,3) if big_rots is not None else None
            
            if  big_len < self.fixed_len : # padding case
                start_index = 0 
                end_index = big_len
            else: # trunking case 
                # set min and  bound for start index so that (for determinist and random case of trunk)
                # - fixed len vector should fit in big seq
                # - separation token should be present
                bmin = max(0, ((seq_len) - self.fixed_len)) # minimal admissible value  for start index to keep separation index
                bmax = min( (big_len - self.fixed_len), seq_len  ) # maximal admissible value for start index
                 
                start_index = start_index = int(torch.randint(bmin, bmax, (1,))) if self.random_trunk and bmax > bmin else  (bmin+bmax)//2 

                end_index = start_index + self.fixed_len
            enc_seq[0:end_index-start_index] = big_seq[start_index:end_index]
            enc_seg[0:end_index-start_index] = big_seg[start_index:end_index]
            if big_coords is not None:
                enc_coords[0:end_index-start_index] = big_coords[start_index:end_index]
            if big_rots is not None:
                enc_rots[0:end_index-start_index] = big_rots[start_index:end_index] 
        else:
            enc_seq, enc_coords, enc_rots, enc_seg = big_seq, big_coords, big_rots, big_seg

        if batchify: # add batch dimension
            enc_seq = enc_seq.squeeze().unsqueeze(0)
            enc_seg = enc_seg.squeeze().unsqueeze(0)
            enc_coords = enc_coords.squeeze().unsqueeze(0) if enc_coords is not None else None
            enc_rots = enc_rots.squeeze().unsqueeze(0) if enc_rots is not None else None

        if return_segment:
            return enc_seq, enc_coords, enc_rots, enc_seg
        else :
            return enc_seq, enc_coords, enc_rots

    def decode(self, seq, coords=None, rots=None, segment=None):
        # squeeze
        seq = seq.squeeze()
        coords = coords.squeeze() if coords is not None else None
        rots = rots.squeeze() if rots is not None else None
        # remove special tokens
        seq_dec = seq[self.unpadded_tokens_mask(seq)] # remove pad if any
        coords_dec = coords[self.unpadded_tokens_mask(seq)] if coords is not None else None
        rots_dec = rots[self.unpadded_tokens_mask(seq)] if rots is not None else None
        # slice in 2 if separator token is present
        if (seq_dec == self.sep_idx).any():
            ind = torch.nonzero( seq_dec == self.sep_idx,as_tuple=True)[0]
            seq_dec2 = self.idx_to_token(seq_dec[ind+1:]) # decode
            coords_dec2 = coords_dec[ind+1:]
            rots_dec2 = rots_dec[ind+1:]
            seq_dec = self.idx_to_token(seq_dec[:ind])  # decode
            coords_dec = coords_dec[:ind]
            rots_dec = rots_dec[:ind]
            return seq_dec, coords_dec, rots_dec, seq_dec2, coords_dec2, rots_dec2
        elif segment is not None:
            # case where dimer
            if (segment==1).any():
                seq_dec2 = self.idx_to_token(seq_dec[segment==1])
                coords_dec2 = coords_dec[segment==1]
                rots_dec2 = rots_dec[segment==1]
            # all cases
            seq_dec = self.idx_to_token(seq_dec[segment==0])
            coords_dec = coords_dec[segment==0]
            rots_dec = rots_dec[segment==0]

            # return depend on case
            if (segment==1).any():
                return seq_dec, coords_dec, rots_dec, seq_dec2, coords_dec2, rots_dec2
            else: 
                return seq_dec, coords_dec, rots_dec 
        else:
            # decode sequence
            seq_dec = self.idx_to_token(seq_dec)  # decode
            return seq_dec, coords_dec, rots_dec

    def unpadded_tokens_mask(self, seq):
        # note : does not take sep pad into account
        return (seq != self.start_idx) & (seq != self.pad_idx)  & (seq != self.end_idx)

    # TODO check 
    def attention_mask(self, seq):
        return  (seq != self.pad_idx)  


    def unpadded_tokens_indices(self, seq):
        return torch.nonzero(self.unpadded_tokens_mask(seq), as_tuple=True)



class PPITokenizer(object):
    def __init__(self,
        fixed_len = None, # if not None define the lenght of the sequence
        random_trunk = False
    ):
        self.normal_tokens = AMINO_ACIDS 
        self.start_token = '['; self.sep_token = '|'; self.pad_token = '-'; self.end_token = ']'; self.mask_token = '?'  
        self.special_tokens = [ self.start_token, self.sep_token ,self.pad_token, self.end_token, self.mask_token ]

        self.fixed_len = fixed_len
        self.random_trunk = random_trunk
                
        print("\t tokenizer created with fixed len : {}".format(self.fixed_len))

    @property
    def all_tokens(self):
        return [*self.special_tokens, *self.normal_tokens]
    
    @property
    def vocab_size(self):
        return len(self.all_tokens)
    
    @property
    def token_to_idx_dict(self):
        return { t:i for i, t in enumerate(self.all_tokens)  }
    
    def token_to_idx(self, sequence, tensorize = False, add_batch_dim=False): 
        seq =  [self.token_to_idx_dict[e] for e in sequence]
        if tensorize: 
            seq = torch.tensor(seq, dtype=torch.long) 
            if add_batch_dim:
                seq = seq[None]
        return seq
        
    @property
    def idx_to_token_dict(self):
        return {i:t for i, t in enumerate(self.all_tokens)}
    
    def idx_to_token(self, sequence):
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist() 
        return ''.join([self.idx_to_token_dict[e] for  e in sequence])
    

    # idx properties
    @property
    def special_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.special_tokens ]

    @property
    def normal_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.normal_tokens ]

    @property
    def all_idx(self):
        return [ self.token_to_idx_dict[token] for token in self.all_tokens ]


    @property
    def start_idx(self):
        return self.token_to_idx_dict[self.start_token]

    @property
    def sep_idx(self):
        return self.token_to_idx_dict[self.sep_token]

    @property
    def pad_idx(self):
        return self.token_to_idx_dict[self.pad_token]
    
    @property
    def end_idx(self):
        return self.token_to_idx_dict[self.end_token]
    
    @property
    def mask_idx(self):
        return self.token_to_idx_dict[self.mask_token]



    def encode(self, data, batchify=False): 
        seq = data['sq_ligand']
        seq2 = data['sq_protein']
        coords = data['coords']
        rots = data['rots']
        plddt = data['plddt'].float()
        pae = data['pae']
        disto_bin = data['disto_bin']
        disto_log = data['disto_log']
        ptm = data['ptm']
        iptm = data['iptm']
        label = data['label']
        tm = torch.stack((ptm, iptm))
        # 1 manage type of sequences and encode
        if 'U' in seq or 'O' in seq:
            seq = seq.replace('U', '?').replace('O', '?')
        if 'U' in seq2 or 'O' in seq2:
            seq2 = seq2.replace('U', '?').replace('O', '?')

        if isinstance(seq, torch.Tensor):
            # raw to str 
            seq = butils.rawidx_to_aa(seq.squeeze())
            seq = self.token_to_idx(seq, tensorize=True)
        elif isinstance(seq, str):
            seq = self.token_to_idx(seq, tensorize=True) 
        else:
            raise ValueError("seq must be string or tensor")
        
        
        if isinstance(seq2, torch.Tensor):
            # raw to str 
            seq2 = butils.rawidx_to_aa(seq2.squeeze())
            seq2 = self.token_to_idx(seq2, tensorize=True)
        elif isinstance(seq2, str):
            seq2= self.token_to_idx(seq2, tensorize=True) 
        elif seq2 is None: 
            seq2 = None
        total_size = len(seq) + len(seq2) +2 if seq2 is not None else len(seq)
        segment_size = torch.full((total_size,), 2)
        segment_size[1:len(seq)+1] = 0
        segment_size[len(seq)+1:len(seq)+len(seq2)+1] = 1
        
        coords = coords.squeeze() if coords is not None else None
        rots = rots.squeeze() if rots is not None else None
        plddt = plddt.squeeze() if plddt is not None else None
        pae = pae.squeeze() if pae is not None else None
        disto_bin = disto_bin.squeeze() if disto_bin is not None else None
        disto_log = disto_log.squeeze() if disto_log is not None else None
        tm = tm.squeeze() if ptm is not None else None
        label = label.squeeze() if label is not None else None

        # 2 add special tokens and truncate
        big_seq  = torch.cat((torch.tensor([self.start_idx]), seq   )) 
        big_coords = torch.cat((torch.zeros(1,3), coords)) if coords is not None else None
        big_rots = torch.cat((torch.zeros(1,3), rots)) if rots is not None else None
        big_plddt = torch.cat((torch.zeros(1), plddt)) if plddt is not None else None

        if pae is not None :
            big_pae =  torch.zeros(pae.shape[0] + 2, pae.shape[1] + 2)
            big_pae[1:-1, 1:-1] = pae
        else:
            big_pae = None
            
        if disto_log is not None:
            big_log = torch.zeros(disto_log.shape[0] + 2, disto_log.shape[1] +  2, disto_log.shape[2])
            big_log[1:-1, 1:-1, :] = disto_log
        else:
            big_log = None
        
        if seq2 is not None:
            big_seq = torch.cat((big_seq, seq2))
        big_seq = torch.cat((big_seq, torch.tensor([self.end_idx])))
        big_coords = torch.cat((big_coords, torch.zeros(1,3))) if big_coords is not None else None
        big_rots = torch.cat((big_rots, torch.zeros(1,3))) if big_rots is not None else None
        big_plddt = torch.cat((big_plddt, torch.zeros(1))) if big_plddt is not None else None
       
        
        seq_len = seq.shape[0] + 2  # size of non truncated encoded seq 
        big_len = big_seq.shape[0]
        # Truncate if necessary
        if self.fixed_len: # truncate or pad
             
            enc_seq = self.pad_idx * torch.ones(self.fixed_len, dtype=torch.long)
            enc_plddt = self.pad_idx * torch.ones(self.fixed_len, dtype=torch.float32) if big_plddt is not None else None
            enc_segment_size = self.pad_idx * torch.ones(self.fixed_len, dtype=torch.long) if segment_size is not None else None
            enc_coords =  torch.zeros(self.fixed_len,3) if big_coords is not None else None
            enc_rots =  torch.zeros(self.fixed_len,3) if big_rots is not None else None
            enc_pae = torch.zeros(self.fixed_len,self.fixed_len) if big_pae is not None else None
            enc_disto = torch.zeros(self.fixed_len,self.fixed_len,64) if big_log is not None else None

            if  big_len < self.fixed_len : # padding case
                start_index = 0 
                end_index = big_len
            else: # trunking case 
                # set min and  bound for start index so that (for determinist and random case of trunk)
                # - fixed len vector should fit in big seq
                # - separation token should be present
                bmin = max(0, ((seq_len) - self.fixed_len)) # minimal admissible value  for start index to keep separation index
                bmax = min( (big_len - self.fixed_len), seq_len  ) # maximal admissible value for start index
                 
                start_index = start_index = int(torch.randint(bmin, bmax, (1,))) if self.random_trunk and bmax > bmin else  (bmin+bmax)//2 

                end_index = start_index + self.fixed_len
            enc_seq[0:end_index-start_index] = big_seq[start_index:end_index]
            enc_plddt[0:end_index-start_index] = big_plddt[start_index:end_index]
            enc_segment_size[0:end_index-start_index] = segment_size[start_index:end_index] if segment_size is not None else None
            if big_coords is not None:
                enc_coords[0:end_index-start_index] = big_coords[start_index:end_index]
            if big_rots is not None:
                enc_rots[0:end_index-start_index] = big_rots[start_index:end_index] 
            if big_pae is not None:
                enc_pae[0:end_index-start_index, 0:end_index-start_index] = big_pae[start_index:end_index, start_index:end_index]
            if big_log is not None:
                enc_disto[0:end_index-start_index, 0:end_index-start_index, :] = big_log[start_index:end_index, 0:end_index-start_index, :]
        else:
            enc_seq, enc_coords, enc_rots, enc_plddt, enc_pae, enc_disto, tm, enc_segment_size, label = big_seq, big_coords, big_rots, big_plddt, big_pae, big_log, tm, segment_size, label

        if batchify: # add batch dimension
            enc_seq = enc_seq.squeeze().unsqueeze(0)
            enc_coords = enc_coords.squeeze().unsqueeze(0) if enc_coords is not None else None
            enc_rots = enc_rots.squeeze().unsqueeze(0) if enc_rots is not None else None
            enc_plddt = enc_plddt.squeeze().unsqueeze(0) if enc_plddt is not None else None
            enc_pae = enc_pae.squeeze().unsqueeze(0) if enc_pae is not None else None
            enc_disto = enc_disto.squeeze().unsqueeze(0) if enc_disto is not None else None
            tm = tm.squeeze().unsqueeze(0) if tm is not None else None
            enc_segment_size = enc_segment_size.squeeze().unsqueeze(0) if enc_segment_size is not None else None    
            label = label.squeeze().unsqueeze(0) if label is not None else None   
    
        data_dict = {
            'seqs': enc_seq,
            'coords': enc_coords,
            'rots': enc_rots,
            'plddt': enc_plddt,
            'pae': enc_pae,
            'disto_log': enc_disto,
            'tm': tm,
            'segment_size': enc_segment_size,
            'label': label
        }

        return data_dict
    
    
    
    
    def decode(self, seqs, coords, rots, plddt, pae, disto_log, tm, segment_size):
        # squeeze
        seqs = seqs.squeeze()
        coords = coords.squeeze() if coords is not None else None
        rots = rots.squeeze() if rots is not None else None
        plddt = plddt.squeeze() if plddt is not None else None
        pae = pae.squeeze() if pae is not None else None
        disto_log = disto_log.squeeze() if disto_log is not None else None
        tm = tm.squeeze() if tm is not None else None
        segment_size = segment_size.squeeze() if segment_size is not None else None
        
        # remove special tokens
        seq_dec = seqs[self.unpadded_tokens_mask(seqs)] # remove pad if any
        coords_dec = coords[self.unpadded_tokens_mask(seqs)] if coords is not None else None
        rots_dec = rots[self.unpadded_tokens_mask(seqs)] if rots is not None else None
        # slice in 2 if separator token is present
        if (seq_dec == self.sep_idx).any():
            ind = torch.nonzero( seq_dec == self.sep_idx,as_tuple=True)[0]
            seq_dec2 = self.idx_to_token(seq_dec[ind+1:]) # decode
            coords_dec2 = coords_dec[ind+1:]
            rots_dec2 = rots_dec[ind+1:]
            seq_dec = self.idx_to_token(seq_dec[:ind])  # decode
            coords_dec = coords_dec[:ind]
            rots_dec = rots_dec[:ind]
            return seq_dec, coords_dec, rots_dec, seq_dec2, coords_dec2, rots_dec2
        else:
            # decode sequence
            seq_dec = self.idx_to_token(seq_dec)  # decode
            return seq_dec, coords_dec, rots_dec

    def unpadded_tokens_mask(self, seq):
        # note : does not take sep pad into account
        return (seq != self.start_idx) & (seq != self.pad_idx)  & (seq != self.end_idx)

    # TODO check 
    def attention_mask(self, seq):
        return  (seq != self.pad_idx)  


    def unpadded_tokens_indices(self, seq):
        return torch.nonzero(self.unpadded_tokens_mask(seq), as_tuple=True)