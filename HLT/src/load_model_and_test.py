import os

import pandas as pd
from USPPM_datamodule import USPPPM_datamodule
from USPPM_dataset import set_max_len
from USPPM_model import USPPPM_model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar 


batch_sizes = {
        "distilbert-base-uncased":128,
        "bert-base-uncased":64,
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

out_dir_prefix = "final_train_"
gpu_id = "1"

print("Model: " + model_name)
print("GPU: " + gpu_id)
print("Batch size: " + str(batch_size))
feature = 2

# Defining a search space!
config_dict = {
    "debug_samples": 1500,
    "DEBUG": True,
    "target_size" : 1,
    "num_workers" : 8,
    # Training parameters
    "batch_size" : batch_size,
    "epochs" : 8,
    "warmup_steps" : 0,
    "min_lr" : 1e-6,
    "encoder_lr" : 2e-5,
    "decoder_lr" : 2e-5,
    "eps" : 1e-6,
    "betas" : (0.9, 0.999),
    "weight_decay" : 0.01,
    "fc_dropout" : 0.2,
    "seed" : 42,
    "train_test_split": 1,
    "loss": "pearson",
    "stratify_on" : 'stratification_index',
    "features" : features[feature],
    "model" : model_name,
    }

INPUT_DIR = '../dataset/us-patent-phrase-to-phrase-matching/'

visible_devices = gpu_id
os.environ["CUDA_VISIBLE_DEVICES"]=visible_devices
num_gpus = len(visible_devices.split(","))


train_df = pd.read_csv("./train_dataframe_with_features.csv")
test_df = pd.read_csv("./test_dataframe_with_features.csv")
if config_dict["DEBUG"]:
    train_df = train_df.iloc[:config_dict["debug_samples"],:]

metrics = {"train_loss" : "train_loss", "val_loss":"val_loss", "val_score":"val_score","train_score":"train_score", "batch_size":"batch_size","fold":"fold", "epoch":"epoch"}

done_pre_evaluation = False

# this try catch is needed to properly terminate the run

pl.seed_everything(config_dict["seed"])

steps_per_epoch = len(train_df) * config_dict['train_test_split'] // config_dict['batch_size']
config_dict['training_steps'] = steps_per_epoch * config_dict['epochs']
config_dict['warmup_steps'] = int(config_dict['training_steps'] * config_dict['warmup_steps'])
                            
set_max_len(config_dict, train_df)  

callbacks = [
            #TuneReportCallback(metrics, on="epoch_end"),
            ModelCheckpoint(
                dirpath=f"checkpoints/",
                filename="best_checkpoint",
                save_top_k=1,
                verbose=True,
                monitor='train_loss',
                mode='min'
            ), 
            EarlyStopping(monitor='train_score', patience=2,mode='max'), 
            TQDMProgressBar(refresh_rate=100)
            ]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
datamodule = USPPPM_datamodule(config_dict, 0.9, train_df, test_df)

trainer = pl.Trainer(
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        max_epochs=config_dict['epochs'],
        min_epochs=2,
        devices=[1], # lightning sees only the gpu that is being assigned to this instance of trainable, so it will be always 0 even if it's using gpu 1,2 or 3
        accelerator="gpu",
        limit_val_batches = 0.0 # needed to skip validation
        )

datamodule.setup()

model = USPPPM_model.load_from_checkpoint("~/ray_results/final_train_distilbert-base-uncased/trainable_e42a4_00000_0_2023-04-16_19-48-27/checkpoints/best_checkpoint.ckpt", config_dict=config_dict)


predictions = trainer.predict(model, datamodule.test_dataloader(), return_predictions=True)

test_df['score'] = predictions[0][1].numpy()
test_df[['id','score']].to_csv(f"test_predictions/{model_name}_{features[feature]}.csv", index=None)

