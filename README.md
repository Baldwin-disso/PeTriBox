![HAPI](https://drive.google.com/uc?id=1eJGbjZj55VEX6vTdH8ynI3nZz_jV3wsQ)

# **PeTriBOX**

PeTriBOX is a collection of fully trainable protein sequence-structure transformer encoder models, built with PyTorch and adapted from BERT. These models can be used for **protein inverse-folding** (via masked language modeling) or **fine-tuned for various tasks**.

We also provide a fine-tuned model named **PeTriPPI**, which enhances **HAPI-PPI**, a protein-protein interaction (PPI) detection tool in our [HAPI](https://github.com/Baldwin-disso/Alphafold2_hapi) toolbox.

---

## **Installation**

### **1. Install Conda Environment**
Clone the repository and set up the environment:

```bash
cd <Repos>
conda env create -f env/PeTriBOX.yml
conda activate petribox
pip install -e .
```

### **2. Install Dependencies for PeTriPPI**
For PeTriPPI, a specific JAX version is required:

```bash
conda activate petribox
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### **3. Download Weights, Example Data, and Datasets**

Ensure you are in the PeTriBOX repository and have activated the environment:

```bash
cd <Repos>
conda activate petribox
```

#### **Download Pretrained Model Weights**
```bash
gdown https://drive.google.com/uc?id=1334V8Hg-JcZKT2pmoHNsq9U7DujXw_Cc
tar -xzvf weights.tar.gz
rm weights.tar.gz
```

#### **Download Example Data (Required for Example Scripts)**
```bash
gdown https://drive.google.com/uc?id=1su5Na6-PH3T3ZWrgwKwLUqfMjAFJNyf1
tar -xzvf data.tar.gz
rm data.tar.gz
```

#### **Download Full Datasets (Optional, ~10GB Storage Required)**
```bash
gdown https://drive.google.com/uc?id=1XO_YunD8OVS925UD7AI63Tz76A9T9yEk
tar -xzvf datasets.tar.gz
rm datasets.tar.gz
```

The dataset folder contains:
- `alphafoldDB_ftp_CANC.pt` → Full dataset from the **AlphaFoldDB FTP Server**.
- `alphafoldDB_ftp_CANC_split42.pt` → Predefined split of the full dataset.
- `alphafoldDB_ftp_CANC_debug.pt` → A **reduced** dataset for debugging.
- `alphafoldDB_ftp_CANC_debug_split42.pt` → Split of the reduced dataset.

---

## **Inference with Pretrained Models**

### **Running Predictions with Python**

Inference can be performed directly using Python scripts available in the `scripts/` directory. Below are the usage examples for each inference type:

#### 1. **Sequence Design via Inverse Folding**
This script performs inverse folding on `.pdb` files found in the input folder. The output consists of designed sequences that are embedded into `.pdb` files within the output folder. Note that this prediction does not modify the original `.pdb` structure and does not generate lateral chains.

```bash
python scripts/predict_sequence.py \
    /path/to/input_pdbs \
    /path/to/output_pdbs \
    --model-path /path/to/model \
    --draws 5 \
    --temp 1.0 \
    --suffix _designed \
    --checkpoint inverse_folding.checkpoint
```

| Argument | Description |
|----------|-------------|
| `/path/to/input_pdbs` | Path to the folder containing `.pdb` files for inverse folding (positional) |
| `/path/to/output_pdbs` | Path to the folder where designed sequences will be saved (positional) |
| `--model-path` | Path to the model weights to load |
| `--draws` | Number of designed sequences per input structure |
| `--temp` | Sampling temperature for sequence generation |
| `--suffix` | Suffix appended to output files |
| `--checkpoint` | Name of the checkpoint file for resuming |

#### 2. **Binder Sequence Design**
This script generates binder sequences using inverse folding. It requires an existing peptide-protein complex structure in `.pdb` format, passed as a folder. Chain A represents the peptide/ligand, and Chain B represents the receptor/protein.

```bash
python scripts/predict_binder.py \
    /path/to/input_complexes \
    /path/to/output_sequences \
    --model-path /path/to/model \
    --draws 3 \
    --temp 0.85 \
    --suffix _binder \
    --checkpoint inverse_folding.checkpoint
```

| Argument | Description |
|----------|-------------|
| `/path/to/input_complexes` | Folder containing `.pdb` files of protein-peptide complexes (Chain A = ligand, Chain B = receptor) (positional) |
| `/path/to/output_sequences` | Folder where designed binder sequences will be saved (positional) |
| `--model-path` | Path to the model weights to load |
| `--draws` | Number of designed binder sequences per complex |
| `--temp` | Sampling temperature for sequence generation |
| `--suffix` | Suffix appended to output files |
| `--checkpoint` | Name of the checkpoint file for resuming |

#### 3. **Protein-Protein Interaction (PPI) Prediction**
This script performs PPI boosting/prediction using outputs from HAPI-PPI ([HAPI](git@github.com:Baldwin-disso/Alphafold2_hapi.git)). It takes a folder of `.pkl` files (outputs of HAPI-PPI) and provides a probability estimation for misclassification. HAPI-PPI uses a threshold on the **PAE interaction** metric from Alphafold Multimer to predict interactions in a peptide-protein dimer. A high probability output from PeTriPPI suggests that HAPI-PPI might have misclassified the interaction.

```bash
python scripts/predict_ppi.py \
    /path/to/input_pkl \
    /path/to/output_predictions.csv \
    --model-path /path/to/model
```

| Argument | Description |
|----------|-------------|
| `/path/to/input_pkl` | Folder containing `.pkl` files (HAPI-PPI outputs) (positional) |
| `/path/to/output_predictions.csv` | Path to the output `.csv` file with predictions (positional) |
| `--model-path` | Path to the model weights to load |

### Running Examples Using Shell Scripts

For convenience, example scripts are provided in `<Repos>/PeTriBOX/scripts/run_predict_*.sh`. These scripts automate the execution of the commands above with predefined paths and parameters.

- `run_predict_sequence_examples.sh` → Runs `predict_sequence.py` for inverse folding.
- `run_predict_binder_examples.sh` → Runs `predict_binder.py` for binder sequence design.
- `run_predict_ppi_example.sh` → Runs `predict_ppi.py` for PPI prediction.

These scripts can be used as templates and modified as needed to fit specific use cases.



---


## **Training from Scratch**

### **Available Encoders**
PeTriBOX supports different variants for different use cases:

| Model | Description |
|--------|------------|
| PeTriBOX | Standard Transformer model with positional embeddings |
| PeTriPOV | Includes point-of-view (POV) embeddings for 3D structures |
| PeTriMPOV | Multi-biased variant of PeTriPOV |

### **Training**

Training is designed to run on HPC systems like Jean-Zay and may require adaptation for different computing environments.

```bash
python scripts/train.py \
    --root-path my_weights/ \
    --model-name MyPeTriBERT \
    --model-cls-name PeTriBERT \
    --task MLM \
    --data-path datasets/alphafoldDB_ftp_CANC.pt \
    --log
```

| Argument | Description |
|----------|-------------|
| `--root-path` | Directory where the trained model will be saved |
| `--model-name` | Custom name for the trained model |
| `--model-cls-name` | Model class to use (e.g., `PeTriBERT`) |
| `--task` | Task to train the model for (e.g., `MLM` for Masked Language Modeling) |
| `--data-path` | Path to the dataset used for training |
| `--log` | Enables logging for monitoring training progress |
| `--from-pretrained` | Path to a pre-trained model to fine-tune |
| `--resume` | Resumes training from a previous checkpoint, bypassing other arguments |

Other training options (batch size, learning rate, number of epochs, etc.) can be explored by running:
```bash
python scripts/train.py --help
```

### **Running Tests on Test Splits**

To evaluate a trained model on test splits, use:

```bash
python scripts/test.py \
    --data-path datasets/alphafoldDB_ftp_CANC.pt \
    --root-path weights \
    --model-name PeTriBERT \
    --test-name lmreport \
    --test-collator Collatorforlm \
    --test-model LMreport 
```

| Argument | Description |
|----------|-------------|
| `--data-path` | Path to the test dataset |
| `--root-path` | Folder where trained models are stored |
| `--model-name` | Name of the trained model to be used for testing |
| `--test-name` | Name of the folder/model containing test results |
| `--test-collator-cls` | Specifies the collator class used for testing |
| `--test-model-cls` | Specifies the model class to use during testing |

Additional options for evaluation can be found using:
```bash
python scripts/test.py --help
```

### Running Training and Tests Using Shell Scripts

For convenience, shell scripts are provided:

- `run_pretrain_*.sh` → Full training scripts (require adaptation for supercomputing clusters like Jean-Zay).
- `run_pretrain_*_dev.sh` → Debugging versions with smaller datasets and reduced hyperparameters.
- `run_test_*.sh` → Runs evaluation on test splits.

These scripts automate the execution of the Python commands above and can be modified to fit specific computational setups.



---


## **Advanced : finetune with your own models**

- Go to `<repos>/PeTriBOX/model/models.py` and add new head for a new task, for example :
```python 

class MyHeadForMytask(BaseModule):
    def __init__(self, 
        d_model
    ):
        super().__init__()
        self.linear = nn.Linear(d_model, 8) 
    
    def forward(self, x):
        x = x[:,0]  # pooled output 
        x = self.linear(x) 
        return x
```

- Optionnally, add a modified transformer encoder :

```python

class MyEncoder(BaseModule):
    def __init__(
        self, # etc
    ):
        # add code 

    def forward(
        self, 
        seqs,
        coords # etc
    )
```

You can freely copy any existing encoder and add or remove submodules from it


- register head and encoder in `MODEL_MAPPING` and `TASK_MAPPING` from `load` in  `<repos>/PeTriBOX/model/models.py`:
```python
def load(kwargs=None, 
        checkpoint_path=None, 
        checkpoint_part_to_load = ["base", "head"],
        strict=True,
        device_type = 'cpu', 
        device_id=0, 
        distributed=False, 
    ):
    """
        ...
    """
    # register models
    MODEL_MAPPING = {
        'PeTriBERT': PeTriBERT,
        'PeTriPOV': PeTriPOV,
        'PeTriMPOV': PeTriMPOV,
        'PeTriPPI': PeTriPPI,
        'MyEncoder' : MyEncoder #   -----> REGISTER encoder  HERE  
    }
    # register task
    TASK_MAPPING = {
        'MLM': MLMHead,
        'ppi': PPIHead, 
        'MyTask': MyHeadForMytask #  ----> REGISTER head HERE
    }
```


- Add your dataset logic in `<repos>/PeTriBOX/data/data.py` data.py and collator in `<repos>/PeTriBOX/data/collate.py`. Collator will need the following format:

```python

class CollatorForPPI(object):
    def __init__(self, # etc
    ):
        # Your code
        

    def __call__(self, batch):
        
       # Your code

        return {
            'inputs': {
                'seqs': sequences,
                'coords' : coordinates, # etc
            }, 
            'labels': labels 
        }
```

where the keys in `'inputs'` should match the name of the arguments in the forward method of the encoder and `'labels'` contains the labels required for training

- In `<repos>/PeTriBOX/scripts/train.py`, ensure dataset and collator are loaded. Add a loss in `<repos>/PeTriBOX/opt/opt/criterions.py` if necessary. (**NOTE : This will be automated in a future realease based on registered tasks**)

- run `<repos>/PeTriBOX/model/models.py` with your own hyperparameters as args and providing both your encoder and task/head (`--model-cls-name MyEncoder --task MyTask`) and providing path of pretrained weights (`--from-pretrain <pretrain_model_path>`). Unused args/kwargs from argparse will be filtered out.

---

## **Other**

### **PeTriPPI training**

We do not provide the dataset for finetuning PeTriPPI because of the size of the dataset (1.7 TB)


### **Weights format**

Models weights consists in folders containing the following files:
- train.json : specific hyperparameters for training.
- model.json : specific hyperparameters necessary to instanciate a model.
- model.pt : model weights saved as pytorch state dict
- resume.json : most of the data required to resume training.
- resume.pt : state dicts of optimizer and scheduler required to resume training.

Note that only model.pt and model.json are necessary to perform inference.

### **Tensorboard logging**

During training a `runs` folder is created at the root path of model folders  (specified by `--root-path`). This can be used to track the training metrics through a navigator after  `tensorboard --logdir=runs` is run in a terminal .


### **Predictors, samplers and indices iterators** 

- `<repos>/PeTriBOX/predict/predictors.py` folders contains predictors that can wraps models to perform inference. 
- `<repos>/PeTriBOX/predict/samplers.py` contains python objects which purposes is to define model inference sampling such as **temperature sampling** or **nucleous sampling**
- `<repos>/PeTriBOX/predict/indices_iterators.py` contains iterators that can be used to define the sampling order (sequentially, randomly, etc.) 

---

## **Cite this work**

For PeTriBERT
```
@article{dumortier2022petribert,
  title={Petribert: Augmenting bert with tridimensional encoding for inverse protein folding and design},
  author={Dumortier, Baldwin and Liutkus, Antoine and Carr{\'e}, Cl{\'e}ment and Krouk, Gabriel},
  journal={BioRxiv},
  pages={2022--08},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

For PeTriPOV, PeTriMPOV and PeTrIPPI

```
@article{connesson2025hapi,
  title={FILL},
  author={FILL},
  journal={BioRxiv},
  pages={FILL},
  year={FILL},
  publisher={Cold Spring Harbor Laboratory}
}
```


## **License**

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENCE) file for details.


## **Contributing**

If you’d like to contribute, please submit an issue or a pull request. Any improvements are welcome!


## **Contributors**

- **Baldwin Dumortier** - Main developement
- **Lena Connesson** - PeTriPPI developement 
- **Gabriel Krouk**, **Antoine Liutkus**, **Clément Carre** - Ideas and other contributions

