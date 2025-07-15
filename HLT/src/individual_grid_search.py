import os
# General Libraries
import pandas as pd

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar 
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# Ray[Tune]
import ray
from ray import air
from ray import tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.runtime_env import RuntimeEnv
from ray.air.config import RunConfig, CheckpointConfig

#our code 
from model.kfold_loop import KFoldLoop
from model.USPPM_model import USPPPM_model
from model.USPPM_model_no_attention import USPPPM_model_no_attention
from model.USPPM_model_lstm import USPPM_model_lstm
from model.USPPM_dataset import set_max_len
from model.USPPM_kfold_datamodule import USPPPM_kf_datamodule

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
        "ahotrod/electra_large_discriminator_squad2_512":16,
        "microsoft/deberta-v3-large":4
        }

features = [
            # 'anchor_target_CPCdescription',
            # 'same_anchor_context_similar_targets',
            # 'CPCdescription_same_anchor_context_similar_targets',
            #'same_anchor_context_targets',
            #'cpc_code',
            None
                                    ]


parser = argparse.ArgumentParser(description='Run individual grid-searches')
parser.add_argument('model_id', type=int, choices=range(0, len(batch_sizes)), help='transformer ID')
parser.add_argument('gpu_id', type=str, help='GPU ID')
parser.add_argument('--initial_bs', type=int, help='initial batch size, gets halved if the trial runs out of memory.')
parser.add_argument('--readout', type=str, help='initial batch size, gets halved if the trial runs out of memory.', choices=['linear','lstm','attention'], default='attention')
parser.add_argument('--out_dir', type=str, help='output directory prefix', default="AUTO_BATCH_SIZE")
parser.add_argument('--debug', action="store_true", help='enable debug')
parser.add_argument('--initial_valid_only', action="store_true", help='performs just the initial validation, without fitting')
parser.add_argument('--lr', type=float, help='learning rate', default=2e-5)
parser.add_argument('--features', type=str, help='features', default=tune.grid_search(features))
args = parser.parse_args()

model_name = list(batch_sizes.keys())[args.model_id]

if args.initial_bs is None:
    batch_size = batch_sizes[model_name]
else:
    batch_size = args.initial_bs

gpu_id = args.gpu_id

print("Model: " + model_name)
print("GPU: " + gpu_id)
print("Batch size: " + str(batch_size))


# Defining a search space!
config_dict = {
    "debug_samples": 1500,
    "DEBUG": args.debug,
    "target_size" : 1,
    "num_workers" : 8,
    
    # Training parameters
    "batch_size" : batch_size,
    "epochs" : 8,
    "n_fold" : 4,
    "warmup_steps" : tune.grid_search([ 
                    0, 
					#0.01,
					 #0.1,
                     #0.5,
                    #1
			]),
    "save_configs": False,
    "min_lr" : 1e-6,
    "encoder_lr" : args.lr,#tune.grid_search([1e-4,1e-6]),
    "decoder_lr" : 2e-5,
    "eps" : 1e-6,
    "betas" : (0.9, 0.999),
    "weight_decay" : 0.01,
    "fc_dropout" : 0.2,
    "seed" : 42,
    "train_test_split": 0.9,
    "loss": "bce",
    "readout": "attention",#tune.grid_search(['linear','attention','lstm']),
    "stratify_on" : tune.grid_search([
                                    #'score_map', 
                                    'stratification_index',
                                     ]),

    "features" : args.features,
    # Transformers
    "model" : tune.choice([model_name])
    }

INPUT_DIR = '../dataset/us-patent-phrase-to-phrase-matching/'

visible_devices = gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=visible_devices
num_gpus = len(visible_devices.split(","))

#runtime_env = RuntimeEnv("CUDA_VISIBLE_DEVICES": visible_devices})
ray.init()#, runtime_env=runtime_env)

dataframe = pd.read_csv("./data/train_dataframe_with_features.csv")
if config_dict["DEBUG"]:
    dataframe = dataframe.iloc[:config_dict["debug_samples"],:]

dataframe_ref = ray.put(dataframe)

def trainable(config_dict):  # Pass a "config" dictionary into your trainable.
    
    metrics = {"val_score": "val_score", "train_loss" : "train_loss", "val_loss" : "val_loss", "batch_size":"batch_size","fold":"fold", "epoch":"epoch"}

    # for gid in ray.get_gpu_ids():
    #     # Previous trial may not have freed its memory yet, so wait to avoid OOM
    #     ray.tune.utils.wait_for_gpu(gid)
    
    dataframe_copy = ray.get(dataframe_ref)

    #trial_id = ray.air.session.get_trial_id()
    OUTPUT_DIR = './'
    logging_dir = f"USPPPM"
    
    export_path = f'ensemble_checkpoints/'
    
    for d in [export_path, OUTPUT_DIR, "lightning_logs/"+logging_dir]:
        try:
            os.makedirs(d)
        except FileExistsError:
            pass
    
    #session.report({"val_score": "val_score"}, checkpoint="best_checkpoint")

    logger = TensorBoardLogger("lightning_logs", name=logging_dir)
    
    done_pre_evaluation = False
    
    # this try catch is needed to properly terminate the run
    while True:
        try:
            pl.seed_everything(config_dict["seed"])
            
            steps_per_epoch = len(dataframe_copy) * config_dict['train_test_split'] // config_dict['batch_size']
            config_dict['training_steps'] = steps_per_epoch * config_dict['epochs']
            config_dict['warmup_steps'] = int(config_dict['training_steps'] * config_dict['warmup_steps'])
                                        
            set_max_len(config_dict, dataframe_copy)  
            

            callbacks = [
                        TuneReportCallback(metrics, on="validation_end"),
                         
                        ModelCheckpoint(
                            dirpath=f"checkpoints/",
                            filename="best_checkpoint",
                            save_top_k=1,
                            verbose=True,
                            monitor='val_score',
                            mode='max'
                        ), 
                        #EarlyStopping(monitor='val_score', patience=1,mode='max'), 
                        TQDMProgressBar(refresh_rate=100)
                        ]
            
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            datamodule = USPPPM_kf_datamodule(config_dict, dataframe_copy)

            if config_dict["readout"] == "attention":
                model = USPPPM_model(config_dict)
            elif config_dict["readout"] == "linear":
                model = USPPPM_model_no_attention(config_dict)
            elif config_dict["readout"] == "lstm":
                model = USPPM_model_lstm(config_dict)
            trainer = pl.Trainer(
                    logger=logger,
                    num_sanity_val_steps=0,
                    check_val_every_n_epoch=1,
                    callbacks=callbacks,
                    max_epochs=config_dict['epochs'],
                    min_epochs=2,
                    devices=[0], # lightning sees only the gpu that is being assigned to this instance of trainable, so it will be always 0 even if it's using gpu 1,2 or 3
                    accelerator="gpu"
                    )

            datamodule.setup()
            datamodule.setup_folds(config_dict['n_fold'])
            if not done_pre_evaluation:
                for fold in range(0,config_dict['n_fold']):
                    datamodule.setup_fold_index(fold)
                    model.current_fold = fold
                    model.epoch=-1
                    trainer.validate(model, datamodule)
                done_pre_evaluation = True
            
            
            if not args.initial_valid_only:
                internal_fit_loop = trainer.fit_loop
                trainer.fit_loop = KFoldLoop(config_dict['n_fold'], config_dict, export_path=export_path)
                trainer.fit_loop.connect(internal_fit_loop)        
                trainer.fit(model, datamodule)

            break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                config_dict['batch_size'] = config_dict['batch_size'] // 2
                print(f"Out of memory, reducing batch size to {config_dict['batch_size']}")
                torch.cuda.empty_cache()

                del internal_fit_loop, model, datamodule, trainer, callbacks
                continue
            else:
                raise e
            
    del dataframe_copy, logger

    

    
    
tuner = tune.Tuner(tune.with_resources(trainable, 
                                       {"cpu":config_dict["num_workers"],"gpu":1},
                                       ),
                    param_space = config_dict,
                    #tune_config = tune.TuneConfig(max_concurrent_trials=0),
                    run_config = RunConfig(name=args.out_dir+model_name, verbose=1)
                                   )

results = tuner.fit()

ray.shutdown()

# Get a dataframe for the last reported results of all of the trials 
df = results.get_dataframe()

df.to_csv("Pearson_loss/" + model_name.replace("/","-") + '_grid_search_results.csv')
