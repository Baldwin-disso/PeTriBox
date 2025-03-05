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

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from PeTriBOX.model.basics import PositionwiseFeedForward, SublayerConnection

# Attention modules 

class AttentionWithBias(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, attention_bias=None, mask=None, dropout=None):
        
        
        scores = (torch.matmul(query, key.transpose(-2, -1)) + attention_bias) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            seq_len = mask.shape[1]
            # mask shaping
            m1 =  mask[...,None].repeat(1,1,seq_len)
            m2 =  mask[...,None].repeat(1,1,seq_len)
            mask = m1*m2
            mask = mask[:,None].repeat(1,scores.shape[1],1,1)
            scores = scores.masked_fill(~mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttentionWithBias(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = AttentionWithBias()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attention_bias=None, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        attention_bias = attention_bias.permute(0,3,1,2) # (b,L,L,h) -> (b,h,L,L)
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, attention_bias=attention_bias, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

# transformers modules
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttentionWithBias(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, attention_bias, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, attention_bias=attention_bias, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BiasedBERTEncoder(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4


        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, attention_bias=None, mask=None):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, attention_bias, mask)

        return x
    


class MultiBiasedBERTEncoder(nn.Module):
    '''
        Biased BERT encoder with multiple  (unshared) bias at each layer.
        n_layers layers of PositionwiseFeedForward() are created and paired with each transformer block
        
        Transformer block N is fed with attention_bias 
        that have through  only one specific associated to transformer block
    '''

    def __init__(self, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mode = "serial"):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.feed_forward_hidden = hidden * 4
        self.mode = mode

        # Create a list of sub-networks for updating the attention bias
        self.bias_update_blocks = nn.ModuleList([
            PositionwiseFeedForward(d_model=attn_heads, d_ff=attn_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        )


    
    def forward(self, x, attention_bias=None, mask=None):
        for transformer, bias_update in zip(self.transformer_blocks, self.bias_update_blocks):
            if attention_bias is not None:
                fed_bias = bias_update(attention_bias) 
                
            x = transformer.forward(x, fed_bias, mask)
        return x



    