import os
# General Libraries
import pandas as pd

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar 
from pytorch_lightning.loggers import TensorBoardLogger
import torch
# Scikit-learn
from sklearn.model_selection import train_test_split


#our code 
from kfold_loop import KFoldLoop
from USPPM_model import USPPPM_model
from USPPM_dataset import set_max_len
from USPPM_kfold_datamodule import USPPPM_kf_datamodule

from pynvml import *
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from pytorch_lightning.callbacks.callback import Callback


batch_sizes = {
        "distilbert-base-uncased":256,
        "bert-base-uncased":64,
        #"AI-Growth-Lab/PatentSBERTa":128,
        #"anferico/bert-for-patents":128,
        "Yanhao/simcse-bert-for-patent":64,
        "ahotrod/electra_large_discriminator_squad2_512":32,
        "microsoft/deberta-v3-large":8
        }

features = [
           'CPCdescription_same_anchor_context_similar_targets',
           'anchor_target_CPCdescription',
            # do not add this #'same_anchor_similar_targets',
           'same_anchor_context_targets',
           'same_anchor_context_similar_targets'
                                    ]


model_name = list(batch_sizes.keys())[0]
batch_size = batch_sizes[model_name]
gpu_id = '1'

print("Model: " + model_name)
print("GPU: " + gpu_id)
print("Batch size: " + str(batch_size))


# Defining a search space!
config_dict = {
    "debug_samples": 1500,
    "DEBUG": False,
    "target_size" : 1,
    "num_workers" : 8,
    
    # Training parameters
    "batch_size" : batch_size,
    "epochs" : 5,
    "n_fold" : 4,
    "warmup_steps" : 0, 
    "min_lr" : 1e-6,
    "encoder_lr" : 2e-5,
    "decoder_lr" : 2e-5,
    "eps" : 1e-6,
    "betas" : (0.9, 0.999),
    "weight_decay" : 0.01,
    "fc_dropout" : 0.2,
    "seed" : 42,
    "train_test_split": 0.9,
    "loss": "pearson",
    "stratify_on" : 'stratification_index',
    "features" : 'anchor_target_CPCdescription',
    "model" : model_name
    }

INPUT_DIR = '../dataset/us-patent-phrase-to-phrase-matching/'

visible_devices = gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=visible_devices
num_gpus = len(visible_devices.split(","))

dataframe = pd.read_csv(f"./train_dataframe_with_features.csv")


done_pre_evaluation = False

# this try catch is needed to properly terminate the run

pl.seed_everything(config_dict["seed"])

steps_per_epoch = len(dataframe) * config_dict['train_test_split'] // config_dict['batch_size']
config_dict['training_steps'] = steps_per_epoch * config_dict['epochs']
config_dict['warmup_steps'] = int(config_dict['training_steps'] * config_dict['warmup_steps'])
print          
set_max_len(config_dict, dataframe)  

datamodule = USPPPM_kf_datamodule(config_dict, dataframe)
model = USPPPM_model(config_dict)

for name, para in model.named_parameters():
    print('{}: {}'.format(name, para.shape))

