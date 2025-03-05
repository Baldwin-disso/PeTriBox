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

import itertools
import inspect
from functools import reduce

def partition_keys_along_group_of_keys(d , l):
    """
        d : dictionnary
        l : list of list of keys 
        return a list of dictionnaries whose keys are define by each list in l
    """
    # check things up
    l_chained =  list(itertools.chain(*l))
    assert len(l_chained) == len(set(l_chained)), "duplicates keys in second arguments list"
    assert all( [ k in l_chained for k in d ] ), " all key from dictionnary should be in partition"

    undefined_keys = set(l_chained) - set(d)
    if undefined_keys:
        print(f"\t WARNING : during partitioning, the following keys were not found : {undefined_keys}")
    return [ { k:d[k] for k in sl if k in d }  for sl in l  ] # for each sublist of l, extract the dictionnary

def filter_kwargs_for_class(cls, kwargs):
    """
    find necessary kwargs for cls class (and remove others)
    """
   
    class_args = inspect.signature(cls).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in class_args}
    filtered_out_keys = [k for k in kwargs if k not in class_args]
    # non overriden class kwargs
    non_overriden_kwargs = {k for k in class_args if k not in  filtered_kwargs }    
    return filtered_kwargs, filtered_out_keys, non_overriden_kwargs

def instanciate_from_mapping_and_kwargs( name, kwargs, mapping=None):
    """
    -mapping : dict of {name : Class}
    -name : class to instanciate using mapping
    -kwargs : kwargs to instantite class
    """

    cls = mapping[name] if mapping else globals()[name]
    cls_kwargs, filtered_kwargs, non_overriden_kwargs = filter_kwargs_for_class(cls, kwargs)
    instance = cls(**cls_kwargs)
    print( f"\t kwargs in {name} class : {cls_kwargs}")
    print( f"\t filtered out kwargs from {name} : {filtered_kwargs}")
    print( f"\t Non overidden kwargs from {name} : {non_overriden_kwargs}  ")
    return instance, cls_kwargs




def merge_function(functions):
    def f_merge(x):
        return tuple(f(x) for f in functions)
    return f_merge

def compose_functions(functions):
    def f_composed(x):
        return reduce(lambda v, f: f(v), functions, x)
    return f_composed

