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
import random

def load_indices_iterator( iterator_kwargs=None):
    ITERATOR_MAPPING = {
        'SequentialIndiceIterator': SequentialIndiceIterator,
        'OneShotIndiceIterator': OneShotIndiceIterator,
        'RandomUniqueIndiceIterator': RandomIndiceIterator
    }
    # assert model checkpoint is in kwargs
    assert "iterator_cls_name" in iterator_kwargs, "iterator_cls_name should be in kwargs" 


    # instanciate predictor
    iterator_cls_name = iterator_kwargs.pop('iterator_cls_name')
    indice_iterator, iterator_kwargs = instanciate_from_mapping_and_kwargs(
        iterator_cls_name, 
        iterator_kwargs, 
        mapping=ITERATOR_MAPPING
        )
    

    return indice_iterator, {'iterator_cls_name':iterator_cls_name, **iterator_kwargs}

class IndiceIterator:
    def __init__(self):
        pass

    def __call__(self, data):
        self.data = data
        self.current_idx = 0
        self.yielded = False
        return self

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError("This method should be overridden by subclasses")


class SequentialIndiceIterator(IndiceIterator):
    def __init__(self, window_size=1):
        super().__init__()
        self.window_size = window_size

    def __next__(self):
        if self.current_idx < len(self.data):
            next_indices = self.data[self.current_idx:self.current_idx + self.window_size]
            self.current_idx += self.window_size
            return next_indices
        else:
            raise StopIteration


class OneShotIndiceIterator(IndiceIterator):
    def __init__(self):
        super().__init__()

    def __next__(self):
        if not self.yielded:
            self.yielded = True
            return self.data
        else:
            raise StopIteration
        


class RandomIndiceIterator(IndiceIterator):
    def __init__(self, nb_indices=1):
        super().__init__()
        self.nb_indices=1

    def __call__(self, data):
        super().__call__(data)
        self.data = random.shuffle(data)

    def __next__(self):
        if self.current_idx < len(self.data):
            next_indices = self.data[self.current_idx:self.current_idx + self.nb_indices]
            self.current_idx += self.nb_indices
            return next_indices
        else:
            raise StopIteration
