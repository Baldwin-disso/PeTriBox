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

import argparse
from pathlib import Path
import tqdm
import torch
from PeTriBOX.predict.predictors import load_predictor
import warnings



def write_paths_to_txt(file_path, paths):
    with open(file_path, "w") as file:
        for path in paths:
            file.write(path + "\n")

def append_paths_to_txt(file_path, paths):
    with open(file_path, "a") as file:
        for path in paths:
            file.write(path + "\n")


def read_paths_from_txt(file_path):
    with open(file_path, "r") as file:
        paths = [line.strip() for line in file]
    return paths



def main():
        parser = argparse.ArgumentParser(description='predict binder')
        parser.add_argument(
                "pdb_dir",
                type=str,
                help= "provide a directory full of .pdb files" 
        )
        parser.add_argument(
                "out_dir",
                type=str,
                help= "path of the output generated .pdb directory" 
        )
        parser.add_argument(
                "--draws-per-pdb",
                default=1, 
                type=int,
                help= "number of new generated sequence for each of the input pdb" 
        )

        parser.add_argument(
                "--model-path",
                default='', 
                type=str,
                help= "number of new generated sequence for each of the input pdb" 
        )
        parser.add_argument(
                "--temperature",
                default=1.0,
                type=float,
                help="sampling temperature for sequence generation. sup. 1 for higher diversity, inf. to 1 for more robustness"
        )
        parser.add_argument(
               "--out-file-suffix",
               type=str, 
               default="_design"
        )
        parser.add_argument(
               "--checkpoint_name",
               type=str, 
               default="inverse_folding.checkpoint"
        )

        args = parser.parse_args()

        # inputs
        pdb_dir = Path(args.pdb_dir); out_dir = Path(args.out_dir)
        model_path = args.model_path
        draws_per_pdb = args.draws_per_pdb; temperature = args.temperature
        out_file_suffix = args.out_file_suffix; checkpoint_path = Path(out_dir, args.checkpoint_name)

        out_dir.mkdir(parents=True, exist_ok=True)

        # predictor loading
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(f"prediction will run on {device} ")
             

        predictor_kwargs = {
            "predictor_cls_name": "sequence prediction multi",
            "model_checkpoint": model_path,
            "device_type":device,
            "indices_iterator_kwargs" :{
                "iterator_cls_name" :"SequentialIndiceIterator"
            },
            "sampler_kwargs": {
                "sampler_cls_name": "TemperatureSampler",
                "temperature":temperature
            }
        }

        predictor, predictor_kwargs = load_predictor(predictor_kwargs=predictor_kwargs)

        # manage file to process
        
        pdb_stems = [p.stem for p in pdb_dir.iterdir() if p.suffix == '.pdb']
        if Path(checkpoint_path).exists():
            processed_stems = read_paths_from_txt(checkpoint_path)
            pdb_stems = list(set(pdb_stems) - set(processed_stems))
        pdb_files = [Path(pdb_dir, s + '.pdb') for s in pdb_stems]

        # main loop
        print(f"start of generation, {len(pdb_files)} files to be processed  ")
        for pdb_file in tqdm.tqdm(pdb_files):
            pdbs_out = [str(Path(out_dir, pdb_file.stem + out_file_suffix + f'{d}' + '.pdb' )) for d in range(draws_per_pdb) ]
            predictor(pdb_files_or_frames=[str(pdb_file)], draws=draws_per_pdb,  pdbs_out=pdbs_out)
            if all([Path(p).exists() for p in pdbs_out]):
                print(f"{pdb_file.stem} was processed")
                append_paths_to_txt(checkpoint_path, [pdb_file.stem])
        print("end of binder generation")
        

if __name__ ==  '__main__':
        main()
     



