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

import pandas as pd 
import numpy as np 
import torch
import torch.utils.data
from pathlib import Path
import argparse
import json
import os
from sklearn.metrics import accuracy_score
import random 
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
from Bio import PDB
from Bio import SeqIO 
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqUtils
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import one_to_three
import warnings


import datetime


AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
PROTEIN_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

AA_to_RAWID = { e:i for (i,e) in enumerate(AMINO_ACIDS) } 
RAWID_to_AA = { i:e for (i,e) in enumerate(AMINO_ACIDS) }


def aa_to_rawidx(sequence, tensorize=False):
    seq =  [AA_to_RAWID[e] for e in sequence]
    seq = torch.tensor(seq, dtype=torch.int) if tensorize else seq
    return seq

def rawidx_to_aa(sequence): 
    return ''.join([AMINO_ACIDS[e] for  e in sequence])


def backbone_triangle_to_orthobasis(CA, N, C):
    # center is CA carbon
    r_coord = CA
    # orthogonal basis is built using gram schmidt on CA, N and C carbon
    u = N - CA 
    v = C - CA
    # normalise u
    u = u/torch.norm(u, p=2, dim=-1, keepdim=True)
    # gram schmidt
    aux = v  - (u*v).sum(-1, keepdim=True) * u
    v = aux / torch.norm(aux, p=2, dim=-1, keepdim=True)
    # vectorial product
    w = torch.linalg.cross(u,v)
    r_rot = torch.stack((u,v,w), dim=-1) 
    return r_coord, r_rot


def seq2fasta(names_or_name, seq_list, fasta_path, append=False):
    # manage if name or names are given
    if isinstance(names_or_name,list):
        names = names_or_name
    else:
        names = [ names_or_name + '_{}'.format(i)  for i in range(len(seq_list))]
    # write names to fasta file
    record_list = []
    for name, seq in zip(names, seq_list):
        record = SeqRecord(Seq(seq), name, "", "")
        record_list.append(record)
    open_mode = "a" if append  else "w" 
    with open(fasta_path, open_mode) as output_handle:
        SeqIO.write(record_list, output_handle, "fasta")


def fasta2seq(fasta_path):
    gen = SeqIO.parse(
        fasta_path,
        'fasta'
    )
    seq_list = [ str(record.seq) for record in gen ]
    return seq_list


def gather_protDB3D_files(prot3DB_path_list):
    for c, f in enumerate(prot3DB_path_list):
        f = Path(f)
        assert f.is_file() and f.suffix == '.pt', 'a wrong file list was given'
        l = torch.load(str(f))
        if c == 0: # init with first file 
            gather_dict = { k:[] for k in l}
        for k in gather_dict.keys():
            if k == 'amino_acids_lut':
                gather_dict.update({'amino_acids_lut' : l['amino_acids_lut']})  
            else:  
                gather_dict[k] = list(chain(gather_dict[k], l[k])) 
    return  gather_dict


def pdb_to_biopython_chains(pdb_file):
    assert Path(pdb_file).suffix == '.pdb' or Path(pdb_file).suffix == '.cif'
    if Path(pdb_file).suffix == '.pdb':
        parser = PDB.PDBParser()
    if Path(pdb_file).suffix == '.cif':
        parser = PDB.MMCIFParser()
    structure = parser.get_structure('osef', pdb_file)

    chains = []
    model =  structure[0]
    chains = [chain for chain in model] 
    return chains


def pdb_to_frames(pdb_file):
    assert Path(pdb_file).suffix == '.pdb' or Path(pdb_file).suffix == '.cif'
    if Path(pdb_file).suffix == '.pdb':
        parser = PDB.PDBParser()
    if Path(pdb_file).suffix == '.cif':
        parser = PDB.MMCIFParser()
     
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure('osef', pdb_file)

    chains = []
    model =  structure[0]
    chains = [chain for chain in model]

    results = {}

    for chain in chains: 
        residues = chain.get_residues()
        r_names = []
        r_coords = []
        r_rots = []
        for r in residues: 
            r_name = SeqUtils.seq1(r.get_resname())
            if r_name == 'X':
                continue
            r_coord, r_rot  = backbone_triangle_to_orthobasis(
                torch.tensor(r['CA'].get_coord()),
                torch.tensor(r['N'].get_coord()),
                torch.tensor(r['C'].get_coord())
            )
            r_rot = compress_rotation_matrice(r_rot)
            r_names += r_name
            r_coords.append(r_coord)
            r_rots.append(r_rot)
        # stacking and converting
        r_seq = ''.join(r_names)
        r_coords = torch.stack(r_coords)
        r_rots = torch.stack(r_rots)
        # updating results 
        results.update(
            {chain.id : { 'seqs': r_seq,  'coords': r_coords, 'rots':r_rots } }
        )
    return results




def pdb_to_seqs(pdb_file):
    assert Path(pdb_file).suffix == '.pdb' or Path(pdb_file).suffix == '.cif'
    if Path(pdb_file).suffix == '.pdb':
        parser = PDB.PDBParser()
    if Path(pdb_file).suffix == '.cif':
        parser = PDB.MMCIFParser()
    structure = parser.get_structure('osef', pdb_file)

    names = []
    seqs = []
    for model in structure:
        for chain in model:
            name =  str(model.id) + '_' + chain.id 
            seq = ''.join([
                SeqUtils.seq1(r.get_resname()) 
                for r in chain.get_residues() 
                if SeqUtils.seq1(r.get_resname()) != 'X'
            ])
            names.append(name)
            seqs.append(seq) 
    return seqs


def brute_replace_seq_in_pdb(seq, pdb_file, out_pdb):
    class CA_N_C_O_Select(Select):
        def accept_atom(self, atom):
            return atom.get_name() in ('CA', 'N', 'C', 'O')

    def modify_residue_names(structure, sequence):
        i = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ' and i < len(sequence):
                        new_resname = one_to_three(sequence[i])
                        residue.resname = new_resname
                        i += 1

    # Parse the input PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('model', pdb_file)
    
    # Filter atoms and save the structure
    io = PDBIO()
    io.set_structure(structure)
    filtered_pdb = "filtered.pdb"
    io.save(filtered_pdb, CA_N_C_O_Select())
    
    # Re-parse the filtered structure
    structure = parser.get_structure('model', filtered_pdb)
    
    # Modify residue names based on the sequence
    modify_residue_names(structure, seq)
    
    # Save the modified structure
    io.set_structure(structure)
    io.save(out_pdb)
    
    return out_pdb

    


def replace_values_by_value(x, values_list, value):
    for v in values_list:
        x[x==v] = value
    return x


def compress_rotation_matrice(M):
    mx = torch.atan2(M[...,2,1], M[...,2,2])
    my = torch.atan2(M[...,2,0],  (M[...,1,0]**2 + M[...,0,0]**2)**0.5   )
    mz =  torch.atan2(M[...,1,0],M[...,0,0]) 
    m = torch.stack((mx,my,mz),dim=-1)
    return m


def uncompress_rotation_matrice(m, return_subrotation=False, inverse=False):
    s = m.shape[:-1]
    ones = torch.ones(s).to(m.device)
    zeros = torch.zeros(s).to(m.device)
    tx = m[...,0]; ty = m[...,1]; tz = m[...,2]
    if inverse: # gives the inverse rotation
        tx = -tx; ty = -ty; tz = -tz


    Mx = torch.stack(
        (
        torch.stack((ones,zeros,zeros),dim=-1),
        torch.stack((zeros, torch.cos(tx) , torch.sin(tx)),dim=-1),
        torch.stack((zeros, -torch.sin(tx) , torch.cos(tx)),dim=-1)
        ),
        dim=-1
    )

    My = torch.stack(
        (
        torch.stack((torch.cos(ty) ,  zeros , torch.sin(ty)),dim=-1),
        torch.stack((zeros, ones, zeros),dim=-1),
        torch.stack((-torch.sin(ty), zeros , torch.cos(ty)),dim=-1)
        ),
        dim=-1
    )

    Mz = torch.stack(
        (
        torch.stack((torch.cos(tz)  ,torch.sin(tz) , zeros),dim=-1),
        torch.stack((-torch.sin(tz) , torch.cos(tz), zeros),dim=-1),
        torch.stack((zeros, zeros, ones),dim=-1)
        ),
        dim=-1
    )
    
    if return_subrotation:
        return Mx, My, Mz
    else: 
        M = torch.matmul(torch.matmul(Mz , My) , Mx)
        if inverse : # gives the inverse rotation
            M = torch.matmul(torch.matmul(Mx , My) , Mz)
        return M

