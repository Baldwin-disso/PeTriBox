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
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
from pathlib import Path
from Bio.PDB import PDBParser
import PeTriBOX.utils.bio_utils as bio_utils
import tok
import argparse



#### Raw plot functions



def gradient_colored_3d_line(ax,x,y,z, cmap, vmap):
    """
       function that draw a 3 line using cmap 
    """
    for i,v in enumerate(vmap):
        #import pdb; pdb.set_trace()
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cmap(v))



def plot_protein(
    protein_coords, 
    file_name=None,
    cmap= 'jet',
    vmap = None, 
    colored_edge = True, 
    other_points_coords = None,
    other_points_cmap = 'jet',
    other_points_vmap = None
    ):
    """
        function that plots a protein from its coordinates :
        arguments : 
        - coords -> np.array [len, 3] : coordinates of the residues of a protein
        - file_name -> str : relative file name of format <folder>/<protein>
        - cmap -> str : matplotlib cmap for the protein to draw (jet, plasma etc)
        - vmap -> np.array[len] : array of value for color mapping. when none is given, it colors residues depending on it position the sequence
        - colored_edge : if true, edge of the protein are colored along with cmap/vmap
        - other_points_coords : a list of other 3 points to draw
        - other_points_cmap : cmap for the others points
        - other_points_vmap : vmap for the others points
    """
    # prepare protein data for plot
    protein_coords = protein_coords.to('cpu').detach().numpy()
    L = protein_coords.shape[0]
    x = protein_coords[:,0]
    y = protein_coords[:,1]
    z = protein_coords[:,2] 
    cmap = plt.get_cmap(cmap) 
    if vmap is None:
     vmap = np.arange(L)/L 
    elif torch.is_tensor(vmap):
        vmap = vmap.numpy()
    

    # get protein name
    name = Path(file_name).stem if file_name is not None else None


    # plot protein

    plt.figure()
    ax = plt.axes(projection="3d")
    if not colored_edge: 
        ax.plot3D(x,y,z, label='protein residues')
    else: 
        gradient_colored_3d_line(ax,x,y,z,cmap, vmap)

    ax.scatter3D(x,y,z, c=vmap, cmap=cmap)

    # plot other points
    # prepare scatterd point for plot
    if other_points_coords is not None:
        xs =  other_points_coords[:,0]
        ys =  other_points_coords[:,1]
        zs =  other_points_coords[:,2]
        cmap2 = plt.get_cmap(other_points_cmap)
        other_points_vmap = np.zeros(other_points_coords.shape[0]) if other_points_vmap is None else other_points_vmap

        ax.scatter3D(xs,ys,zs,c=other_points_vmap, cmap=cmap2)

    # add legend and title
    ax.legend()
    plt.title(name)



def plot_subway_style(real_residues_or_probs, other_locs=None, other_probs=None, view_range=(0.0,5.0)):
    """
    real_residues_or_probs : array of residues values or matrix or residues probabilities 
    locations : array (location index) of other locations in between residues 
    probabilities : array (proability, location index) associated probabilities in beetween residues
    """
    # create figure
    fig, ax = plt.subplots()

    # compute probabilities of real residues
    real_locs = np.arange(real_residues_or_probs.shape[-1])
    if real_residues_or_probs.ndim == 1:
        #import pdb; pdb.set_trace()
        real_probs = np.zeros((real_residues_or_probs.max()+1, real_residues_or_probs.size))
        col = np.arange(real_residues_or_probs.size)
        real_probs[real_residues_or_probs, col] = 1
    elif real_residues_or_probs.ndim == 2:
        real_probs = real_residues_or_probs

    if other_locs is not None and other_probs is not None:
        # concatenate and sort data
        # conc
        all_locs = np.concatenate((other_locs,real_locs))
        all_probs = np.concatenate((other_probs,real_probs),axis=1)
        # sort 
        sargs = np.argsort(all_locs)
        all_locs = all_locs[sargs]
        all_probs = all_probs[:,sargs]
    else:
        (all_locs, all_probs) = (real_locs, real_probs)
  
    # plot 
    cum_probs = np.cumsum(all_probs,axis=0)
    for i in range(cum_probs.shape[0]):
        cum_probs_prev = cum_probs[i-1] if i > 0 else np.zeros((cum_probs.shape[1]))
        ax.plot(all_locs, cum_probs[i])
        ax.fill_between(all_locs, cum_probs_prev, cum_probs[i], alpha=0.2)

    # plot vertical lines
    for xc in real_locs:
        plt.axvline(x=xc, color='black', linewidth = 4 , linestyle ='-')

    plt.xlim(view_range)
  


## unitary tests
def unit_tests():
    # TEST 1 protein to plot 
    prot3DB_path = '/Users/baldwin/work/data/prot3DB.pt'
    alphafoldDB_path = '/media/speckbull/data/alphafoldDB'
    prot3DB = torch.load(prot3DB_path)
    n = 0
    seq = prot3DB['sequences'][n]
    coords = prot3DB['coordinates'][n]
    file_name = prot3DB['files_names'][n]

    s_coords = coords + torch.normal(
            torch.zeros(coords.shape),
            torch.ones(coords.shape)
        )


    #plt = plot_protein(coords, file_name=file_name, vmap = np.arange(coords.shape[0]-1,-1,-1), cmap='jet', other_points_coords=s_coords, other_points_vmap=np.arange(s_coords.shape[0]))
    plt = plot_protein(coords)
    plt.show()
    

    # TEST 2 subway style plot
    
    # ex 1
    residues = np.array([ 3, 0, 2, 1, 3 ])
    locations = np.array([ 0.5, 1.5 , 2.3, 2.5, 3.8 ] )
    probs = np.array([
        [0.7, 0.6, 0.2, 0.3, 0.1],
        [0.1, 0.1, 0.4, 0.4, 0.6 ],
        [0.19, 0.1, 0.2, 0.1, 0.1 ],
        [0.01, 0.2, 0.2, 0.2, 0.2 ]
    ])
    #plot_subway_style(residues, locations, probs)

    #ex 2
    real_len = 30
    sim_len = 200
    nb_amino_acids_types = 20 
    residues = np.random.randint(0, high=nb_amino_acids_types, size=real_len, dtype=int)
    locations = np.random.rand(sim_len) * real_len
    probs = np.random.rand(nb_amino_acids_types,sim_len) 
    probs = probs/probs.sum(axis=0,keepdims=True)
    plot_subway_style(residues, locations, probs)
    plot_subway_style(residues, None, None)
    plot_subway_style(probs, None, None)
    plt.show()
    import pdb; pdb.set_trace()
    
   
    

if __name__=='__main__':
    unit_tests()
 
    
   

