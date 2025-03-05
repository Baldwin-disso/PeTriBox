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
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_model_jsons(root, json_name):
    """
    Loads JSON files from subdirectories within the specified root directory.

    Parameters:
    - root (str): The root directory containing subdirectories with JSON files.
    - json_name (str): The name of the JSON file to look for in each subdirectory.

    Returns:
    - dict: A dictionary where keys are subdirectory names and values are the loaded JSON data.
    """
    data = {}
    root_path = Path(root)
    
    # Iterate over subdirectories
    for folder in root_path.iterdir():
        if folder.is_dir():
            json_file_path = folder / json_name
            
            # Check if the JSON file exists
            if json_file_path.is_file():
                with json_file_path.open('r') as file:
                    try:
                        model_data = json.load(file)
                        data[folder.name] = model_data
                    except json.JSONDecodeError as e:
                        print(f"JSON decoding error in {json_file_path}: {e}")
    
    return data

def merge_json_data(data):
    """
    Merges JSON data into a Pandas DataFrame.

    Parameters:
    - data (dict): A dictionary containing JSON data.

    Returns:
    - pd.DataFrame: A DataFrame with JSON keys as columns and subdirectory names as index.
    """
    # Create a DataFrame from the data
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Replace missing values with NaN
    df = df.replace({None: np.nan})
    
    return df

def main(root, json_name, output_file):
    """
    Main function to load, merge, and save JSON data as a CSV file.

    Parameters:
    - root (str): The root directory where JSON files are searched.
    - json_name (str): The name of the JSON file to read.
    - output_file (str): The name of the output CSV file.
    """
    model_data = load_model_jsons(root, json_name)
    merged_data = merge_json_data(model_data)
    
    # Display the table
    print(merged_data)
    
    # Save to a CSV file
    merged_data.to_csv(output_file, index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a table of keys from JSON files.")
    parser.add_argument("root", help="Root directory where JSON files are searched.")
    parser.add_argument("--json-name", default="model.json", help="Name of the JSON file to read (default: model.json).")
    parser.add_argument("--output-file", default="experiments_hyperparams.csv", help="Name of the output file (default: experiments_hyperparams.csv).")
    
    args = parser.parse_args()
    
    main(args.root, args.json_name, args.output_file)
