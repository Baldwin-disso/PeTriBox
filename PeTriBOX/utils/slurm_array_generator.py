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

import json
import itertools

# 1 input
grid = { 'init_lr':[1e-6, 5e-5, 1e-4, 1e-3 ] , 'warmup': [300, 3000, 30000] }

# 2 generate slurm array 
names = grid.keys()
values_lists = [ l for l in grid.values() ]
values_products = list(itertools.product(*values_lists) )


array = {} # array to build
for i, values_product in enumerate(values_products): # for every item of the cartesian product of values
    d = {}
    for name, value in zip(names, values_product ): # for every parameter of the item
        d.update({name : value})
    array.update({ i : d })


# 3 save
with open('slurm_array.json', 'w') as outfile:
    outfile.write(json.dumps(array, indent=4, sort_keys=True))
