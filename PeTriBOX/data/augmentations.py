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

from PeTriBOX.utils import bio_utils
import torch

# data augmentation
class GlobalRotationAugmentor(object):
    def __init__(self):
       print("\t global rotation Augmentor")

    def __call__(self, coords, rots=None): # coords of shape (samples, seq, 3), or (seq, 3) or (3) 
        # make coords dimension = 3
        if coords.dim()==1:
            coords = coords[None,None]
        elif coords.dim()==2:
            coords = coords[None]

        # make rotation dimensions = 3
        if rots is not None:
            if rots.dim()==1:
                rots = rots[None,None]
            elif rots.dim()==2:
                rots = rots[None]
            # rebuilt rotation matrices from angles
            rots = bio_utils.uncompress_rotation_matrice(rots)
        
        # Create augmented rotation matrices 
        # Now coords is of shape (samples, seq, 3)
        # and rots is of shape (samples, seq, 3,3)
        nb = coords.shape[0] 
        seq_len = coords.shape[1]
        pi = torch.acos(torch.zeros(1)).item() * 2

        # randomly generate 2 rotation angles (along x axis and y axis) for every sample
        angle_x = 2*pi*torch.rand(nb)
        angle_y = 2*pi*torch.rand(nb)
        angle_z = 2*pi*torch.rand(nb)
        # blocks of zeros and ones
        ones = torch.ones(nb)
        zeros = torch.zeros(nb)

        # build rotation matrix along  x axis
        x1 = torch.stack((ones, zeros, zeros), dim=1)
        x2 = torch.stack((zeros, torch.cos(angle_x), torch.sin(angle_x)), dim = 1)
        x3 = torch.stack((zeros, -torch.sin(angle_x), torch.cos(angle_x)), dim = 1 )
        x = torch.stack((x1,x2,x3), dim=2) # rotation matrices in tensor of size (nb, 3,3)
        x = x[:,None].repeat(1, seq_len, 1,1) # duplicate tensor so it fits in tensor of size (nb, seq_len, 3, 3)

        # build rotation matrix along y axis 
        y1 = torch.stack((torch.cos(angle_y), zeros, torch.sin(angle_y)), dim=1)
        y2 = torch.stack((zeros, ones, zeros), dim = 1)
        y3 = torch.stack((-torch.sin(angle_y), zeros, torch.cos(angle_y)), dim = 1 )
        y = torch.stack((y1,y2,y3), dim=2) # rotation matrices in tensor of size (nb, 3,3)
        y = y[:,None].repeat(1, seq_len, 1,1) # duplicate tensor so it fits in tensor of size (nb, seq_len, 3, 3)

        # build rotation matrix along z axis 
        z1 = torch.stack((torch.cos(angle_z), torch.sin(angle_z), zeros), dim=1)
        z2 = torch.stack((-torch.sin(angle_z), torch.cos(angle_z), zeros), dim = 1)
        z3 = torch.stack((zeros, zeros, ones), dim = 1 )
        z = torch.stack((z1,z2,z3), dim=2) # rotation matrices in tensor of size (nb, 3,3)
        z = z[:,None].repeat(1, seq_len, 1,1) # duplicate tensor so it fits in tensor of size (nb, seq_len, 3, 3)

        # apply rotations on coords
        coords = coords[...,None] # unsqueeze for matmul
        coords_mean = coords.mean(dim=1,keepdim=True) # mean over sequence dimension
        coords = torch.matmul(x, coords - coords_mean) + coords_mean # substract mean, rotate, add mean back
        coords = torch.matmul(y, coords - coords_mean) + coords_mean # subtract mean rotate, add mean back
        coords = torch.matmul(z, coords - coords_mean) + coords_mean # subtract mean rotate, add mean back
        coords = coords.squeeze() # get back to orginal shape

        # apply rotations on rots
        if rots is not None:
            rots = torch.matmul(x,rots)
            rots = torch.matmul(y,rots)
            rots = torch.matmul(z,rots)
            rots = bio_utils.compress_rotation_matrice(rots)
            rots = rots.squeeze()


        return coords, rots
 

class GlobalTranslationAugmentor(object):
    def __init__(self, scaling = 1):
        self.scaling = scaling
        print("\t translation 3D augmentor created with scaling : {}".format(self.scaling))

    def __call__(self, coords, rots=None):
        # make coords dimension = 3
        if coords.dim()==1:
            coords = coords[None,None]
        elif coords.dim()==2:
            coords = coords[None]

        # make rotation dimensions = 3
        if rots is not None:
            if rots.dim()==1:
                rots = rots[None,None]
            elif rots.dim()==2:
                rots = rots[None]
            
        

        # Now coords is of shape (samples, seq, 3)
        nb = coords.shape[0] 
        seq_len = coords.shape[1]

        # generate translation
        translation = self.scaling * torch.rand(nb,1,3)
        
        # apply translation 
        coords = coords + translation
        coords = coords.squeeze()

        rots = rots.squeeze()

        return coords, rots


class TranslationNoiseAugmentor(object):
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        print("\t Position noise 3D augmentor created with sigma : {}".format(self.sigma))

    def __call__(self, coords, rots=None): 
        # make coords dimension = 3
        if coords.dim()==1:
            coords = coords[None,None]
        elif coords.dim()==2:
            coords = coords[None]

        # make rotation dimensions = 3
        if rots is not None:
            if rots.dim()==1:
                rots = rots[None,None]
            elif rots.dim()==2:
                rots = rots[None]
            

        # First augment add noise to coords
        shape = coords.shape
        noise = torch.normal(
            torch.zeros(shape),
            self.sigma*torch.ones(shape)
        )
        coords = coords + noise
        coords = coords.squeeze()

        rots = rots.squeeze()

        return coords, rots

class RotationNoiseAugmentor(object):
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        print("\t Position noise 3D augmentor created with sigma : {}".format(self.sigma))

    def __call__(self, coords, rots=None): 
        # make coords dimension = 3
        if coords.dim()==1:
            coords = coords[None,None]
        elif coords.dim()==2:
            coords = coords[None]

        # make rotation dimensions = 3
        if rots is not None:
            if rots.dim()==1:
                rots = rots[None,None]
            elif rots.dim()==2:
                rots = rots[None]
            
            # Second add noise to rots 
            shape = rots.shape

            # generate small noise over rots of normal law N(0,sigma)
            noise = torch.normal(
                torch.zeros(shape),
                self.sigma*torch.ones(shape)
            )

            # add noise
            rots = rots + noise

            rots = rots.squeeze()

        return coords, rots


# augmentor composition
def augmentors_compose(*args, augmentors_list):
    if augmentors_list:
        for augmentor in augmentors_list :
                args = augmentor(*args)
    return args
