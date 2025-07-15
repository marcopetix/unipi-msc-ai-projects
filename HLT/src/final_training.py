import os
# General Libraries
import pandas as pd
import numpy as np
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar 
from pytorch_lightning.loggers import TensorBoardLogger
import torch
# Scikit-learn
from sklearn.model_selection import train_test_split

# Ray[Tune]
import ray
from ray import air
from ray import tune
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.runtime_env import RuntimeEnv
from ray.air.config import RunConfig, CheckpointConfig

#our code 
from model.USPPM_dataset import set_max_len
from model.USPPM_datamodule import USPPPM_datamodule
from model.USPPM_model import USPPPM_model
from model.USPPM_model_no_attention import USPPPM_model_no_attention
from model.USPPM_model_lstm import USPPM_model_lstm
from pynvml import *
import argparse

import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning) 
from pytorch_lightning.callbacks.callback import Callback


batch_sizes = {
        "distilbert-base-uncased":54,
        "bert-base-uncased":64,
        "Yanhao/simcse-bert-for-patent":64,
        "ahotrod/electra_large_discriminator_squad2_512":32,
        "microsoft/deberta-v3-large":16
        }

features = [
           'CPCdescription_same_anchor_context_similar_targets',
           'anchor_target_CPCdescription',
            # do not add this #'same_anchor_similar_targets',
           'same_anchor_context_targets',
           'same_anchor_context_similar_targets'
                                    ]


parser = argparse.ArgumentParser(description='Run individual grid-searches')
parser.add_argument('model_id', type=int, choices=range(0, len(batch_sizes)), help='transformer ID')
parser.add_argument('gpu_id', type=str, help='GPU ID')
parser.add_argument('feature', type=int, help='feature')
parser.add_argument('warmup_steps', type=float, help='warmup_steps')
parser.add_argument('epochs', type=int, help='number of epochs')
parser.add_argument('--initial_bs', type=int, help='initial batch size, gets halved if the trial runs out of memory.')
parser.add_argument('--out_dir', type=str, help='output directory prefix')
parser.add_argument('--resume_from', type=str, help='checkpoint path to load the model')
parser.add_argument('--debug', action="store_true", help='enable debug')
parser.add_argument('--initial_valid_only', action="store_true", help='performs just the initial validation, without fitting')

parser.add_argument('--save_configs', action="store_true", help='performs just the initial validation, without fitting')
args = parser.parse_args()

model_name = list(batch_sizes.keys())[args.model_id]

if args.initial_bs is None:
    batch_size = batch_sizes[model_name]
else:
    batch_size = args.initial_bs

if args.out_dir is None:
    out_dir_prefix = "final_train_"
else:
    out_dir_prefix = args.out_dir
gpu_id = args.gpu_id

print("Model: " + model_name)
print("GPU: " + gpu_id)
print("Batch size: " + str(batch_size))


# Defining a search space!
config_dict = {
    "debug_samples": 1500,
    "save_configs": args.save_configs,
    "DEBUG": args.debug,
    "target_size" : 1,
    "num_workers" : 8,
    # Training parameters
    "batch_size" : batch_size,
    "epochs" : args.epochs,
    "warmup_steps" : args.warmup_steps,
    "min_lr" : 1e-6,
    "readout": 'attention',
    "encoder_lr" : 2e-5,
    "decoder_lr" : 2e-5,
    "eps" : 1e-6,
    "betas" : (0.9, 0.999),
    "weight_decay" : 0.01,
    "fc_dropout" : 0.2,
    "seed" : 42,
    "train_test_split": 1,
    "loss": "bce",
    "stratify_on" : 'stratification_index',
    "features" : features[args.feature],
    "model" : model_name,
    "starting_checkpoint": args.resume_from,
    }


visible_devices = gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=visible_devices
num_gpus = len(visible_devices.split(","))

ray.init()

train_df = pd.read_csv("./data/train_dataframe_with_features.csv")
test_df = pd.read_csv("./data/test_dataframe_with_features.csv")
if config_dict["DEBUG"]:
    train_df = train_df.iloc[:config_dict["debug_samples"],:]

train_ref = ray.put(train_df)
test_ref = ray.put(test_df)

def trainable(config_dict):  # Pass a "config" dictionary into your trainable.
    
    metrics = {"train_loss" : "train_loss", "val_loss":"val_loss", "val_score":"val_score","train_score":"train_score", "batch_size":"batch_size","fold":"fold", "epoch":"epoch"}

    train_copy = ray.get(train_ref)
    test_copy = ray.get(test_ref)

    #trial_id = ray.air.session.get_trial_id()
    OUTPUT_DIR = './'
    logging_dir = f"USPPPM"
    
    for d in [OUTPUT_DIR, "lightning_logs/"+logging_dir]:
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
            
            steps_per_epoch = len(train_copy) * config_dict['train_test_split'] // config_dict['batch_size']
            config_dict['training_steps'] = steps_per_epoch * config_dict['epochs']
            config_dict['warmup_steps'] = int(config_dict['training_steps'] * config_dict['warmup_steps'])
                                        
            set_max_len(config_dict, train_copy)  
            

            callbacks = [
                        TuneReportCallback(metrics, on="epoch_end"),
                        ModelCheckpoint(
                            dirpath=f"checkpoints/",
                            filename="best_checkpoint",
                            save_top_k=4,
                            verbose=True,
                            monitor='train_score',
                            mode='max'
                        ), 
                        EarlyStopping(monitor='train_score', patience=2,mode='max'), 
                        TQDMProgressBar(refresh_rate=100)
                        ]
            
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            datamodule = USPPPM_datamodule(config_dict, train_copy, test_copy, train_val_split=0)

            trainer = pl.Trainer(
                    logger=logger,
                    num_sanity_val_steps=0,
                    check_val_every_n_epoch=1,
                    callbacks=callbacks,
                    max_epochs=config_dict['epochs'],
                    min_epochs=2,
                    devices=[0], # lightning sees only the gpu that is being assigned to this instance of trainable, so it will be always 0 even if it's using gpu 1,2 or 3
                    accelerator="gpu",
                    limit_val_batches = 0.0 # needed to skip validation
                    )
                    

            datamodule.setup()

            if config_dict["starting_checkpoint"] is None:
                if config_dict["readout"] == "attention":
                    model = USPPPM_model(config_dict)
                elif config_dict["readout"] == "lstm":
                    model = USPPM_model_lstm(config_dict)
                elif config_dict["readout"] == "linear":
                    model = USPPPM_model_no_attention(config_dict)
                    
                if not done_pre_evaluation:
                    model.epoch=-1
                    trainer.validate(model, datamodule)
                    done_pre_evaluation = True
            else:
                model = USPPPM_model.load_from_checkpoint(config_dict["starting_checkpoint"], config_dict=config_dict)
            
            if not args.initial_valid_only:   
                trainer.fit(model, datamodule)

            # predictions = trainer.predict(model, datamodule.test_dataloader(), return_predictions=True)
            # print(type(predictions), type(predictions[0]), type(predictions[0][0]))
            # print(predictions[0][1].numpy())
            # test_copy['score'] = predictions[0][1].numpy()
            # test_copy[['id','score']].to_csv("test_predictions.csv", index=None)
            break
        except RuntimeError as e:
            if 'out of memory' in str(e):
                config_dict['batch_size'] = config_dict['batch_size'] // 2
                print(f"Out of memory, reducing batch size to {config_dict['batch_size']}")
                torch.cuda.empty_cache()

                continue
            else:
                raise e
            
    del train_copy, test_copy, logger

    

    
    
tuner = tune.Tuner(tune.with_resources(trainable, 
                                       {"cpu":config_dict["num_workers"],"gpu":1},
                                       ),
                    param_space = config_dict,
                    #tune_config = tune.TuneConfig(max_concurrent_trials=0),
                    run_config = RunConfig(name=out_dir_prefix+model_name, verbose=1)
                                   )

results = tuner.fit()

ray.shutdown()

# Get a dataframe for the last reported results of all of the trials 
df = results.get_dataframe()
grid_out_dir = "grid_results"
try:
    os.mkdir(grid_out_dir)
except:
    pass
df.to_csv(os.path.join(grid_out_dir, model_name.replace("/","-") + '_grid_search_results.csv'))
