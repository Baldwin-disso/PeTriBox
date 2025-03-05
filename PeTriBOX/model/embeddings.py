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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, LayerNorm, GELU
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import  FullMask, LengthMask
from PeTriBOX.model.basics import MLP3
from PeTriBOX.utils.bio_utils import uncompress_rotation_matrice
import math


def init_embedding_layers(learnable_embedding_type, linear_layer, MLP3_dim, MLP3_batch_norm_dim='1d'):
    # Initialise with random distribution
    torch.nn.init.uniform_(linear_layer.weight, a=0.0, b=1.0)
    mlp3 = None
    # make learnable or not
    if learnable_embedding_type == 'learnable_weights_and_MLP' :
        linear_layer.weight.requires_grad = True
        mlp3 = MLP3(MLP3_dim, batch_norm_dim=MLP3_batch_norm_dim)
        print("\t embedding :  learnable weights and MLP3 created")
    elif learnable_embedding_type == 'learnable_weights_only' :
        linear_layer.weight.requires_grad = True
        mlp3 = None
        print("\t embedding : learnable weights only")
    elif learnable_embedding_type == 'frozen_weights_and_MLP':
        linear_layer.weight.requires_grad = False
        mlp3 = MLP3(MLP3_dim, MLP3_batch_norm_dim)
        print("\t embedding : postionnal embedder 3D with frozen weights and MLP3")
    elif learnable_embedding_type == 'frozen_weights_only':
        linear_layer.weight.requires_grad = False
        mlp3 = None
        print("\t postionnal embedder 3D with frozen_weights_only created")

    return (linear_layer, mlp3)

#######################################
#  Base embedding
#####################################


class PostionalEmbedding3D(nn.Module):
    def __init__(self, d_model, learnable_embedding_type = 'learnable_weights_and_MLP3'):
        super().__init__()
        assert (d_model % 2) == 0, "please  d_model must be even (n_heads or query_dimension even)"
        
        self.linear = nn.Linear(3, d_model//2, bias=False) 

        self.linear, self.mlp = init_embedding_layers(
            learnable_embedding_type, 
            self.linear,
            d_model, 
            MLP3_batch_norm_dim='1d'
        )

    def forward(self, c):
        e = self.linear(c)
        e = torch.cat((e.cos(), e.sin()),dim=-1)
        if self.mlp is not None:
            e = self.mlp(e)
        return e


class RelativePostionalEmbedding3D(nn.Module):
    def __init__(self, n_heads, learnable_embedding_type = 'learnable_weights_and_MLP3'):
        super().__init__()
        
        # define positionnal matrix as module
        self.linear = nn.Linear(3, n_heads, bias=False) 


        self.linear, self.mlp = init_embedding_layers(
            learnable_embedding_type, 
            self.linear,
            n_heads, 
            MLP3_batch_norm_dim='2d'
        )

    def forward(self, c):
        # compute distance matrix
        d = c[:,None,:,:] - c[:,:,None,:] # (b,L,3) : coords -> (b,L,L,3) distance matrix
        # apply linear layer
        e = self.linear(d) # (b,L,L,3) -> (b,L,L,n_heads)
       # apply MLP3
        if self.mlp is not None:
            e = self.mlp(e)
        return e
    

class POVEmbedding(nn.Module):
    """ Embedding used in POV attention 

    The idea here is that each positionnal embedding is computer
    after all the frame of one protein has been rotated and translated so
    that the coords and rotation are viewed from the query viewpoint
    for that reason we have
    cji_tilde = Ri^(-1) ( Cj - ci )  # Cj points, seen frame i as a reference
    Rji_tilde =  Ri^(-1) Rj # Rj rots, seen from frame i as reference 
      
    """
    
    def __init__(self, n_heads, learnable_embedding_type = 'learnable_weights_and_MLP3'):
        super().__init__()
        
        self.linear = nn.Linear(12, n_heads, bias=False) 

        self.linear, self.mlp = init_embedding_layers(
            learnable_embedding_type, 
            self.linear,
            n_heads, 
            MLP3_batch_norm_dim='2d'
        )
    

    def forward(self, c, a):
        B = c.shape[0]
        L = c.shape[1]
        # compute distance matrix
        d = c[:,None,:,:] - c[:,:,None,:] # (B,L,3) : coords -> (B,L,L,3) distance matrix
        # get rotation matix and inverses 
        R =   uncompress_rotation_matrice(a, inverse=False)
        Rinv =  uncompress_rotation_matrice(a, inverse=True) #  B,L,3  ->  B,L,3,3
        
        # apply inverse matrix for each case 
        # For position first 
        #  GOAL Rinv(B,L,3,3) x c(B,L,L,3)    
        # unsqueeze axis 1 and repeat so that  Rinv(B,L,3,3) -> Rinv(B,L,L,3,3)
        # M matmul  Rinv(B,L,L,3,3) x  c(B,L,L,3) -> res(B,L,L,3)
        
        Rinv_repeat =  Rinv.unsqueeze(2).repeat(1,1,L,1,1)
        d = torch.matmul(Rinv_repeat, d.unsqueeze(-1)).squeeze(-1)
        # for rots then
        # GOAL  Rij(b,L,L,3,3) 
        #  unsqueeze axis 2 of R and repeat so that R(B,L,L,3,3) (but with other axis than R1)
        # then matmul  Rinv(b,L,L,3,3) x R(b,L,L,3,3)
        R_repeat = R.unsqueeze(1).repeat((1,L,1,1,1))   

        R = torch.matmul(Rinv_repeat,R_repeat)
        R = R.reshape(B,L,L,9)
        
        # apply linear layer
        e = self.linear(torch.cat((d,R), dim=-1)) # (b,L,L,3) -> (b,L,L,n_heads)
        
       # apply MLP3
        if self.mlp is not None:
            e = self.mlp(e)
        return e



class RotationEmbedding(nn.Module):
    def __init__(self, d_model, learnable_embedding_type = 'learnable_weights_and_MLP3'):
        super().__init__()
        assert (d_model % 2) == 0, "please  d_model must be even (n_heads or query_dimension even)"
        
        self.linear = nn.Linear(9, d_model//2, bias=False)

        self.linear, self.mlp = init_embedding_layers(
            learnable_embedding_type, 
            self.linear,
            d_model, 
            MLP3_batch_norm_dim='1d'
        )

    def forward(self, a):
        assert a is not None, 'no rotatations parameters is available in data, please choose another dataset'
        bs = a.shape[0]
        seq_len = a.shape[1]
        # Compute rotation matrix
        M = uncompress_rotation_matrice(a)
        M = M.reshape(bs, seq_len, 9)
        # forward through linear layer
        e = self.linear(M)
        # cos and sin function
        e = torch.cat((e.cos(), e.sin()),dim=-1)
        # MLP3
        if self.mlp is not None:
            e = self.mlp(e)
        return e



class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        print("\t Base positionnal embedder created ")

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=2)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=2)


######################################
##  Composite Embedding
######################################


class PeTriEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        seq_len, 
        embed_size, 
        nb_heads=12,
        dropout=0.1, 
        embedding_type='uni', 
        rotation_embedding_type='normal',
        learnable_embedding_type='learnable_weights_and_MLP3'):
        
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.embedding_type = embedding_type
        self.rotation_embedding_type = rotation_embedding_type
        self.learneable_encoding_type = learnable_embedding_type
        # 3D and rotation embedding

        self.rotation = None
        if self.embedding_type == 'relative':
            self.position3D = RelativePostionalEmbedding3D(
                n_heads = nb_heads,
                learnable_embedding_type = learnable_embedding_type
            )
        elif embedding_type == 'tri' or embedding_type == 'unitri' : 
            self.position3D = PostionalEmbedding3D(
                d_model=self.token.embedding_dim,
                learnable_embedding_type = learnable_embedding_type
            )
            if rotation_embedding_type=='normal':
                self.rotation = RotationEmbedding(
                    d_model=self.token.embedding_dim,
                    learnable_embedding_type = learnable_embedding_type
                )
            elif rotation_embedding_type=='none':
                self.rotation = None
            else:
                raise ValueError('rotation_embedding_type  must be \'none\', \'normal\' or \'dummy\' ')
        else:
            self.position3D = None
        # normal bert positionnal encoding
        if embedding_type == 'uni' or embedding_type == 'unitri' :        
            self.position = PositionalEmbedding(d_model=self.token.embedding_dim, max_len=seq_len)
        else: 
            self.position = None

        if not(embedding_type=='uni') and not(embedding_type=='tri')\
            and not(embedding_type=='unitri') and not(embedding_type=='relative'):
            raise ValueError('embedding_type  must be \'uni\', \'tri\' or \'unitri\' or \'relative\' ')

        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.layernorm = nn.LayerNorm([seq_len, embed_size])
        self.dropout = nn.Dropout(p=dropout)
        if self.embedding_type == 'relative':
            self.layernorm_rel = nn.LayerNorm([seq_len, nb_heads])
            self.dropout_rel = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.embedding_type = embedding_type
        self.rotation_embedding_type = rotation_embedding_type
        
        print("\t Type of positionnal  embedding: {}".format(self.embedding_type))
        print("\t Type of rotation  embedding: {}".format(self.rotation_embedding_type))

    def forward(self, seqs, coords=None, rots=None, segment_label=None):

        x = self.token(seqs)  
        relative_bias = None
        if self.position is not None:
            x+= self.position(x)
        if self.position3D is not None: 
            assert coords is not None, "should pass coords when using 3D embeddings"
            if self.embedding_type == 'relative':
                relative_bias = self.position3D(coords)
                relative_bias = self.layernorm_rel(relative_bias)
                relative_bias = self.dropout_rel(relative_bias)
            else: 
                x += self.position3D(coords)  
                if self.rotation is not None:
                    assert rots is not None, "should pass rots params when using rots"
                    x += self.rotation(rots)
        if segment_label is not None:
            x += self.segment(segment_label) 
        x = self.layernorm(x)
        x = self.dropout(x)
        if self.embedding_type == 'relative':
            return x, relative_bias
        else:
            return x



class PeTriPOVEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        seq_len, 
        embed_size, 
        n_heads,
        dropout=0.1, 
        learnable_embedding_type='learnable_weights_and_MLP'):
        
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.learneable_encoding_type = learnable_embedding_type
        # 3D and rotation embedding
        
        self.struct_embedding = POVEmbedding(
            n_heads=n_heads,
            learnable_embedding_type=learnable_embedding_type
        )
         
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.layernorm = nn.LayerNorm([seq_len, embed_size])
        self.dropout = nn.Dropout(p=dropout)
        
        self.layernorm_rel = nn.LayerNorm([seq_len, n_heads])
        self.dropout_rel = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        print("\t using POV embedding")    
    
    def forward(self, seqs, coords, rots, segment_label=None):
        # token part
        x = self.token(seqs)
        if segment_label is not None:
            x += self.segment(segment_label) 
        x = self.layernorm(x)
        x = self.dropout(x)  

        # POV bias part
        relative_bias = self.struct_embedding(coords, rots)
        relative_bias = self.layernorm_rel(relative_bias)
        relative_bias = self.dropout_rel(relative_bias)
        return x, relative_bias
       

class PeTriPPIEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size, # 20 + 5 (AA + special tokens)
        seq_len,  # 1024
        embed_size, # 768
        n_heads, # 12
        dropout=0.1, 
        learnable_embedding_type='learnable_weights_and_MLP'):
        
        super().__init__()
        self.learneable_encoding_type = learnable_embedding_type
        # 3D and rotation embedding
        self.struct_embedding = POVEmbedding(
            n_heads=n_heads,
            learnable_embedding_type=learnable_embedding_type
        )
        # Token embedding for sequence
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        
        
        # Linear layers for continuous values
        self.plddt_linear = nn.Linear(1, embed_size)  # L,1 to L,embed_size
        self.pae_linear = nn.Linear(1, n_heads)  # L,L to L,embed_size
        self.disto_log_linear = nn.Linear(64, n_heads)  # L,L to L,embed_size
        self.tm_linear_bias = nn.Linear(2, n_heads)  # N,2 to N,n_heads
        self.tm_linear = nn.Linear(2, embed_size)  # N,2 to N,embed_size
        self.layernorm = nn.LayerNorm([seq_len, embed_size])
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm_rel = nn.LayerNorm([seq_len, n_heads])
        self.dropout_rel = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.seq_len = seq_len
        print("\t using Hapi embedding")    
    
    def forward(self, seqs, coords, rots, plddt, pae, disto_log, tm, segment_size):
        # token embedding part
        x = self.token(seqs) # (1, 1024) -> (1, 1024, 768)
        x += self.segment(segment_size) # (1, 1024) -> (1, 1024, 768) 
        x += self.tm_linear(tm).unsqueeze(1).repeat(1, self.seq_len, 1)  # (1, 2) -> (1, L, d)

        x += self.plddt_linear(plddt.unsqueeze(-1))  # (1, 1024, 1) -> (1, 1024, 768)
        x = self.layernorm(x)
        x = self.dropout(x)  
        
        relative_bias = self.struct_embedding(coords, rots)
        
        pae = pae.unsqueeze(-1)
        relative_bias = relative_bias + self.pae_linear(pae)
        relative_bias = self.disto_log_linear(disto_log) + relative_bias
        relative_bias = relative_bias + self.tm_linear_bias(tm).unsqueeze(1).unsqueeze(1).repeat(1, self.seq_len, self.seq_len, 1)
        relative_bias = self.layernorm_rel(relative_bias)
        relative_bias = self.dropout_rel(relative_bias)

        
        return x, relative_bias

       

