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

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import  FullMask, LengthMask


class MLP3(nn.Module):
    def __init__(self, d_model,  batch_norm_dim='1d'):
        super().__init__()
        self.l1 = nn.Linear(d_model, d_model)
        self.t1 = nn.Tanh()
        
        self.l2 = nn.Linear(d_model, d_model)
        self.t2 = nn.Tanh()
        
        self.l3 = nn.Linear(d_model, d_model)
        self.t3 = nn.Tanh()

        if batch_norm_dim =='1d':
            self.b1 = nn.BatchNorm1d(d_model)
            self.b2 = nn.BatchNorm1d(d_model)
        elif batch_norm_dim =='2d':
            self.b1 = nn.BatchNorm2d(d_model)
            self.b2 = nn.BatchNorm2d(d_model)
        
    def forward(self,x):
        # layer 1
        x = self.l1(x)
        x = self.t1(x)
        x = x.transpose(1,-1)
        x = self.b1(x)
        x = x.transpose(1,-1)
        # layer 1
        x = self.l2(x)
        x = self.t2(x)
        x = x.transpose(1,-1)
        x = self.b2(x)
        x = x.transpose(1,-1)
        # layer 1
        x = self.l3(x)
        x = self.t3(x)
        return x


# utility modules
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, skip_connection=True):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.skip_connection = skip_connection

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        if self.skip_connection:
            return x + self.dropout(sublayer(self.norm(x)))
        else: 
            return self.dropout(sublayer(self.norm(x)))
