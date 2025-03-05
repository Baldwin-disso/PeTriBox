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



class CrossEntropyLossFlat:
    def __init__(self, weight=None, ignore_index=-100):
        self.weight = weight  # classes weighting
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=self.weight)

    def __call__(self, y_hat, y):
        y_hat_flat = y_hat.reshape(-1, y_hat.shape[-1])
        y_flat = y.reshape(-1)
        loss = self._criterion(y_hat_flat, y_flat)
        return loss
