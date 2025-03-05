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

from bisect import bisect
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, Subset
import itertools
from pathlib import Path
import math
import PeTriBOX.utils.bio_utils as bio_utils
import glob
import json
import os 
import bisect
import random 
import h5py
from PeTriBOX.data import tok
from PeTriBOX.data.augmentations import GlobalRotationAugmentor, GlobalTranslationAugmentor, TranslationNoiseAugmentor, RotationNoiseAugmentor, augmentors_compose 




# Dataset
class Prot3DBDataset(Dataset):
    def __init__(
        self,
        file_path_list=None,
        data=None, 
        data_already_tokenized=False,
        fixed_len=None,
        random_trunk=False,
        global_rotation_augmentation = False, 
        global_translation_augmentation= None,
        rotation_noise_augmentation = None,
        translation_noise_augmentation = None
    ): 
        assert file_path_list is None or data is None
        if file_path_list is not None:
            self.data = bio_utils.gather_protDB3D_files(file_path_list)
        if data is not None: 
            self.data= {
                'sequences':data[0],
                'coordinates':data[1],
                'rotations':data[2]
            }

      
        self.data_already_tokenized = data_already_tokenized
        self.tokenizer = tok.SimpleTokenizer(fixed_len=fixed_len, random_trunk=random_trunk)
        self.augmentors = []

        if global_rotation_augmentation: 
            self.augmentors.append(
                GlobalRotationAugmentor()
            )

        if global_translation_augmentation is not None:
            self.augmentors.append(
                GlobalTranslationAugmentor(scaling=global_translation_augmentation)
            )

        if rotation_noise_augmentation is not None:
            self.augmentors.append(
                RotationNoiseAugmentor(sigma=rotation_noise_augmentation)
            )

        if translation_noise_augmentation is not None:
            self.augmentors.append(
                TranslationNoiseAugmentor(sigma=translation_noise_augmentation)
            )
         

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, ixd):
        # retrieve data
        seq = self.data['sequences'][ixd]
        coords =  self.data['coordinates'] [ixd]
        rots =  self.data['rotations'] [ixd] if 'rotations' in self.data else None
        with torch.no_grad():
            # tokenizing if tokenizer exists and if stored data is not already tokenized
            if (self.tokenizer is not None) and (not self.data_already_tokenized) :
                seq, coords, rots = self.tokenizer.encode(seq, coords, rots)

            # augmenting coords
            coords, rots = augmentors_compose(coords, rots, augmentors_list=self.augmentors)
        
        return  seq, coords, rots

    def subdataset_from_query(
        self,
        query=None,
        indexes=None,  
        nb_duplicates=1,
        q_prefix = 'AF-', 
        q_suffix='-F1-model_v1', 
        full_name_with_folder=False):
        '''
            Query : string of list of string containing query
        '''
        assert (query is None or indexes is None)
        if query is not None:
            # convert to list if necessary
            query = [query] if isinstance(query, str) else query 

        
            if full_name_with_folder:
                database =  self.files_names
            else:
                query = [ q_prefix + q + q_suffix  for q in query]
                database = [ Path(s).stem  for s in  self.files_names]

            # retrieve indexes of query
            indexes = [database.index(q) for q in query]
        
        # define subdataset from these indexes
        
        subset = Subset(self, indexes * nb_duplicates)

        return subset

    @property
    def sequences(self):
        return self.data['sequences']

    @property
    def coordinates(self):
        return self.data['coordinates']
    
    @property
    def files_names(self):
        return self.data['files_names']

    @property
    def amino_acids_lut(self):
        return bio_utils.AMINO_ACIDS

        
class Prot3DBCachedDataset(Dataset):
    def __init__(
        self,
        folder_path = None,
        tokenizer = None,
        global_rotation_augmentation = False, 
        global_translation_augmentation= None,
        rotation_noise_augmentation = None,
        translation_noise_augmentation = None,
        file_format= 'alphafold_{}.pt',
        split_rates = [0.8, 0.1, 0.1], 
        split = 'train', 
        iter_max = None
    ): 
        # define trace logger of dataset
        worker_info = torch.utils.data.get_worker_info()

        
        # unpack
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.file_format = file_format
        self.split_rates =  {"train":split_rates[0], "valid":split_rates[1], "test":split_rates[2]} 
        self.split = split
        self.iter_max = math.floor(iter_max * 
            (self.split_rates[self.split]/ self.split_rates['train'])
        ) 
        # virtual epoch and data_cache
        self.indexes_lut = None  # look up table of indices of dataset related to shards
        self.split_slice = None # start and end indexes related to split
        self.iter_slice = None # start and en indexes related to iteration (virtual epoch)
        self.cache_slice = [] # start and end indexes related to data in cache
        self.cache_data = None # cached data

        # call refresh to update indexes_lut and split_slice
        self.refresh_dataset()
        
    
        # augmentors 
        self.augmentors = []

        if global_rotation_augmentation: 
            self.augmentors.append(
                GlobalRotationAugmentor()
            )

        if global_translation_augmentation is not None:
            self.augmentors.append(
                GlobalTranslationAugmentor(scaling=global_translation_augmentation)
            )

        if rotation_noise_augmentation is not None:
            self.augmentors.append(
                RotationNoiseAugmentor(sigma=rotation_noise_augmentation)
            )

        if translation_noise_augmentation is not None:
            self.augmentors.append(
                TranslationNoiseAugmentor(sigma=translation_noise_augmentation)
            )
            



    def __len__(self):
        # refresh dataset
        self.refresh_dataset() 
        return  self.iter_slice[1] - self.iter_slice[0] 

    def __getitem__(self,idx):
        self.refresh_dataset()

        # get dataset split splice, then get absolute index in dataset (index of data without considering split)
        abs_idx = idx + self.iter_slice[0]
        
        # if query abs_idx is not in cache_data
        if (not self.cache_slice) or (not self.cache_data) or (not abs_idx in range(*self.cache_slice)):
            # find shard 
            file_idx = bisect.bisect(self.indexes_lut, abs_idx) -1
            self.cache_slice = [self.indexes_lut[file_idx] , self.indexes_lut[file_idx+1]]
            self.cache_data = torch.load(
                self.full_format.format(file_idx)
            )

        # compute abs_idx relative to cache_data
        cache_idx = abs_idx - self.cache_slice[0]

        # get data
        seq = self.cache_data['sequences'][cache_idx]
        coords =  self.cache_data['coordinates'] [cache_idx]
        rots =  self.cache_data['rotations'] [cache_idx] if 'rotations' in self.cache_data else None
        # tokenizing if tokenizer exists and if stored data is not already tokenized
        if (self.tokenizer is not None) :
            seq, coords, rots = self.tokenizer.encode(seq, coords, rots)

        # augmenting coords
        coords = augmentors_compose(coords, rots, augmentors_list= self.position_augmentors)
        
        return  seq, coords, rots

    
     
    def refresh_dataset(self):
        #import pdb; pdb.set_trace()
        # At initialisation only 
        if self.indexes_lut is None:
            self._load_lut_file()
            self._update_slices()

        # check if _indexes_lut is up to date, otherwise update
        shards_nb = self._get_shards_nb()
        if shards_nb  >  len(self.indexes_lut) - 1:
            self._update_indexes_lut_for_shards(len(self.indexes_lut)- 1, shards_nb)
            self._dump_lut_file()
            self._update_slices()

    def _load_lut_file(self):
        lut_file_path = Path(self.folder_path,'index_lut.json')
        if lut_file_path.exists():
            with open(lut_file_path,'r') as f:
                res = json.load(f) 
        else: 
            res = [0]
        self.indexes_lut = res

    def _get_shards_nb(self):
        file_list = glob.glob(self.full_format.format('*'))
        return len(file_list)

    def _update_indexes_lut_for_shards(self,start_shrd,end_shrd):
        res = []
        acc = self.indexes_lut[-1] # accumumlated value is the last one
        for i in range(start_shrd,end_shrd):
            file_path = self.full_format.format(i)
            x = torch.load(file_path)
            n = acc + len(x['sequences']) # new value is accumulated + number of sequences
            res.append(n)
            acc = n
        # update _indexes_lut
        self.indexes_lut += res
    
    def _dump_lut_file(self):
        lut_file_path = Path(self.folder_path,'index_lut.json')
        with open(lut_file_path,'w') as f:
            f.write(json.dumps(self.indexes_lut, indent=4, sort_keys=True))

    def _update_slices(self):
        full_dataset_size = self.indexes_lut[-1]
        if full_dataset_size > 0:
            # split slice update
            split_partition = [
                0,
                math.floor(full_dataset_size*self.split_rates['train']),
                math.floor(full_dataset_size*(self.split_rates['train']+self.split_rates['valid'])),
                full_dataset_size
            ]
            if self.split == 'train':
                self.split_slice =  split_partition[0:2]
            elif self.split == 'valid':
                self.split_slice = split_partition[1:3]
            elif self.split == 'test':
                self.split_slice = split_partition[2:]

            # iter slice update
            if self.iter_max is None: # iterate over all the dataset
                self.iter_slice = self.split_slice
            else: # slice split data randomly so that it leads to iter_maxindexes_lut
                iter_start = random.randrange(self.split_slice[0], self.split_slice[1] - self.iter_max)
                iter_end = iter_start + self.iter_max
                self.iter_slice = [iter_start, iter_end]


    
    @property
    def full_format(self):
        return os.path.join(self.folder_path, self.file_format)
    







def get_sq_from_name(self, name, lut_file_path):
    """
    Retrieves the sequence associated with a given name from a lookup table (CSV file).
    
    Args:
        name (bytes): The name in byte format.
        lut_file_path (str): Path to the CSV lookup table file.
    
    Returns:
        str: The new protein sequence if 'R' is in the name, otherwise the new ligand sequence.
    """
    name = name.decode()  # Convert from bytes to string
    column = 'num_R' if 'R' in name else 'num_L'
            
    df = pd.read_csv(lut_file_path, sep=',')
    
    for _, line in df.iterrows():
        if line[column] == name:
            return line['new protein Sequence'] if 'R' in name else line['new ligand Sequence']

def list_positive_interactions(csv_file):
    """
    Identifies positive interactions from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file containing the interactions.
    
    Returns:
        list: A list of tuples representing positive interactions (receptor, ligand).
    """
    df = pd.read_csv(csv_file)
    
    positive_interactions = []
    
    for _, row in df.iterrows():
        ligand = row['num_L']
        receptor = row['num_R']
        
        positive_interactions.append((receptor, ligand))
    
    return positive_interactions

def compute_pae_interaction(pae_matrix, bindersize):
    """
    Computes the average interaction PAE by taking the two anti-diagonal blocks of the PAE matrix.

    Parameters:
    - pae_matrix (torch.Tensor): The Predicted Aligned Error (PAE) matrix.
    - bindersize (int): The size of the binder region.

    Returns:
    - float: The mean PAE interaction value.
    """
    N = pae_matrix.shape[0]
    mid = N // 2

    # Extract the anti-diagonal blocks
    block1 = pae_matrix[:bindersize, bindersize:]
    block2 = pae_matrix[bindersize:, :bindersize]

    # Compute the mean of the block values
    pae_interaction_mean = (torch.mean(block1) + torch.mean(block2)) / 2
    return pae_interaction_mean

def pad_tensor(tensor1, tensor2):
    new_size = len(tensor2) - len(tensor1)
    mean = tensor1.mean(dim=0, keepdim=True)
    noise = torch.randn((new_size, 1, tensor1.shape[2]))  
    padding_value = mean + noise

    new_shape = (tensor2.shape[0], tensor1.shape[1], tensor1.shape[2])
    new_tensor = torch.zeros(new_shape)

    new_tensor[:new_size] = padding_value
    new_tensor[new_size:new_size + tensor1.shape[0]] = tensor1

    return new_tensor


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_dir, lut_file_path, fixed_len=None, random_trunk=False, boosting=False, boosting_threshold=10):
        self.lut_file_path = lut_file_path
        self.hdf5_dir = Path(hdf5_dir)
        self.hdf5_files = sorted(self.hdf5_dir.glob('*.hdf5'))
        
        self.hdf5_data = {}
        self.sample_keys = set()
        self.tokenizer = tok.PPITokenizer(fixed_len=fixed_len, random_trunk=random_trunk)
        self.boosting = boosting
        self.boosting_threshold = boosting_threshold
        self.positive_label_list = list_positive_interactions(self.lut_file_path)
        
        # load keys of each HDF5 file
        for hdf5_file in self.hdf5_files:
            with h5py.File(hdf5_file, 'r') as hdf5:
                for key in hdf5.keys():
                    sample_id = key.split('.')[0]
                    self.sample_keys.add((hdf5_file, sample_id))
                    #self.sample_keys.update({int(sample_id): hdf5_file})
        
        self.sample_keys = list(self.sample_keys)
        
    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, idx):
        
        test = False
        idx_recup = idx
        while test == False:
            hdf5_file, sample_id = self.sample_keys[idx_recup]

            with h5py.File(hdf5_file, 'r') as hdf5:
                data = {k.split('.')[-1]: hdf5[k][()] for k in hdf5.keys() if k.startswith(sample_id)}
            data, test = self.get_all_data(data)
            idx_recup -= 1
            
        data_dict = self.tokenizer.encode(data, batchify=False)

        return data_dict
    
   
    def get_all_data(self, res):
        ligand_sq = get_sq_from_name(self, res['ligand'], self.lut_file_path)
        protein_sq = get_sq_from_name(self, res['protein'], self.lut_file_path)
        data_dict, test = convert_pkl_metric_dict(res, 
            ligand_sq, 
            protein_sq, 
            pae_interaction_threshold=self.boosting_threshold, 
            boosting=self.boosting, 
            positive_label_list=self.positive_label_list
            )
        return data_dict, test
    

def convert_NCAC_dict_to_tensor(NCAC_dict):
    '''
        convert dict of type  NCAC_dict = {1 : ( devicearray , devicearray, devicearray )   }  
        to NCAC_tenors ot type Tensor( seq_len, atom_type, coordinates  )
    '''
    NCAC_dict =   { id : torch.stack([torch.tensor(np.array(e)) for e in data]) for (id,data) in  NCAC_dict.items()   }
    NCAC_tensor =  torch.stack(tuple(NCAC_dict.values()), dim=0) 
    return NCAC_tensor



def convert_pkl_metric_dict(res, ligand_sq=None, protein_sq=None, pae_interaction_threshold=10.0, boosting=True, positive_label_list=None, outputs_pred_inter=False):

    # res_types = [ (k, type(v)) for (k,v) in  res.items()]
    # import pdb; pdb.set_trace()
    # res_sizes =  { k :res[k].shape for k in ['coordinates_NCAC', 'distogram_bin_edges', 'distogram_logits', 'plddt', 'predicted_aligned_error'] }
    # print(f'\n res_types : {res_types} \n res_sizes : {res_sizes}')
    # import pdb; pdb.set_trace()

    if isinstance(res['coordinates_NCAC'], dict):
        T = convert_NCAC_dict_to_tensor(res['coordinates_NCAC'])
    else: 
        S = [torch.tensor(x, dtype=torch.float32) for x in res['coordinates_NCAC']] #creation of list of tensors of coordinates
        T = torch.stack(S) # global tensor for coordinates

    plddt = torch.tensor(res['plddt'], dtype=torch.float32)
    pae = torch.tensor(res['predicted_aligned_error'],  dtype=torch.float32)
    if 'distogram' in res: # pickle format
        disto_bin = torch.tensor(np.array(res['distogram']['bin_edges']))
        disto_log = torch.tensor(np.array(res['distogram']['logits']))
    else: # hdf5 format    
        disto_bin = torch.tensor(np.array(res['distogram_bin_edges']))
        disto_log = torch.tensor(np.array(res['distogram_logits']))
    ptm = torch.tensor(np.array(res['ptm']), dtype=torch.float32)
    iptm = torch.tensor(np.array(res['iptm']), dtype=torch.float32)
    
    if ligand_sq is None:
        assert 'sq_ligand' in res, "sq_ligand should be provided"
        ligand_sq = res['sq_ligand']
    if protein_sq is None: 
        assert 'sq_protein' in res, "sq_protein should be provided"
        protein_sq = res['sq_protein']

    res['protein'] = res['protein'].decode() if isinstance(res['protein'],bytes) else res['protein']
    res['ligand'] = res['ligand'].decode() if isinstance(res['ligand'],bytes) else res['ligand']
    
    T = pad_tensor(T, pae)
    t_coords, t_rots = bio_utils.backbone_triangle_to_orthobasis(T[:,1,:], T[:,0,:], T[:,2,:]) 
    t_rots = bio_utils.compress_rotation_matrice(t_rots) 

    # labels
    if positive_label_list :
        if (res['protein'], res['ligand']) in positive_label_list:
            label = 1
        else:
            label = 0
        label = torch.tensor(label)

    test = (len(ligand_sq) + len(protein_sq) == len(plddt))

    # pae__interaction_prediction
    binder_size = len(ligand_sq)
    pred_inter = 0
    pae_inter = compute_pae_interaction(pae, binder_size)
    pred_inter = int((pae_inter <= pae_interaction_threshold))

    if positive_label_list and boosting:
        label = np.abs(label - pred_inter)
    else: 
        label = None
       
    data_dict = {
            'sq_ligand': ligand_sq,
            'sq_protein': protein_sq,
            'coords': t_coords,
            'rots': t_rots,
            'plddt': plddt,
            'pae': pae,
            'disto_bin': disto_bin,
            'disto_log': disto_log,
            'ptm': ptm,
            'iptm': iptm,
            'label': label
        }
    
    if outputs_pred_inter: 
        return data_dict, test, pred_inter
    else:
        return data_dict, test