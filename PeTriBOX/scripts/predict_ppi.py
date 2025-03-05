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
                "pkl_dir",
                type=str,
                help= "provide a directory full of .plk files, obtained from Hapi-ppi" 
        )
        parser.add_argument(
                "out_csv",
                type=str,
                help= "path of the .csv outputs" 
        )
        
        parser.add_argument(
                "--model-path",
                default='', 
                type=str,
                help= "path to weights" 
        )
        
        parser.add_argument(
               "--checkpoint_name",
               type=str, 
               default="ppi.checkpoint"
        )

        args = parser.parse_args()

        # inputs
        pkl_dir = Path(args.pkl_dir); out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(exist_ok=True, parents=True)
        model_path = args.model_path
        checkpoint_path = Path(pkl_dir, args.checkpoint_name)

        # predictor loading
        device = "cuda" if torch.cuda.is_available() else 'cpu'
        print(f"prediction will run on {device} ")
             

        predictor_kwargs = {
            "predictor_cls_name": "PPI prediction",
            "model_checkpoint": model_path,
            
        }
        predictor, predictor_kwargs = load_predictor(predictor_kwargs=predictor_kwargs)
        
        # manage file to process
        pkl_stems = [p.stem for p in pkl_dir.iterdir() if p.suffix == '.pkl']
        if Path(checkpoint_path).exists():
            processed_stems = read_paths_from_txt(checkpoint_path)
            pkl_stems = list(set(pkl_stems) - set(processed_stems))
        pkl_files = [Path(pkl_dir, s + '.pkl') for s in pkl_stems]

        # main loop
        print(f"start of ppi prediction, {len(pkl_files)} files to be processed  ")
        scores = []
        for pkl_file in tqdm.tqdm(pkl_files):
            score = predictor(pkl_metrics_file_or_obj=pkl_file , out_csv=out_csv)
            scores.append(score)
            append_paths_to_txt(checkpoint_path, [pkl_file.stem])
        print("end of ppi prediction")
        

if __name__ ==  '__main__':
        main()
     



