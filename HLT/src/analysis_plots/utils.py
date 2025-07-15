import os
import pandas as pd
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

feature_labels = {'anchor_target_CPCdescription':'CPC', 
                  'same_anchor_context_targets':'SACT',
                  'same_anchor_context_similar_targets':'SACST', 
                  'CPCdescription_same_anchor_context_similar_targets':'CPC+SACST',
                  None: 'No context',
                  'cpc_code': 'CPC code',}

model_labels = {"distilbert-base-uncased":'DistilBERT', 
                "bert-base-uncased": 'BERT',
                "Yanhao/simcse-bert-for-patent":'BERT-for-patent',
                "ahotrod/electra_large_discriminator_squad2_512":'Electra-l', 
                "microsoft/deberta-v3-large":'DeBERTa-v3-l'
                }

model_sizes = {'distilbert-base-uncased': 66362880,
                'bert-base-uncased': 109482240,
                'Yanhao/simcse-bert-for-patent': 355359744,
                'ahotrod/electra_large_discriminator_squad2_512': 334092288,
                'microsoft/deberta-v3-large': 434012160,
                'gpt2': 124439808}


def compute_model_sizes():
    model_sizes = {}
    for model_path in model_labels.keys():
        config = AutoConfig.from_pretrained(model_path, output_hidden_states = True,truncation = True)
        model = AutoModel.from_pretrained(model_path, config = config)
        model_sizes[model_path] = model.num_parameters()

    model_sizes

def compute_model_size(model_path):
    config = AutoConfig.from_pretrained(model_path, output_hidden_states = True,truncation = True)
    model = AutoModel.from_pretrained(model_path, config = config)
    return model.num_parameters()

def convert_epoch(x):
    if type(x) == str:
        if 'e' in x:
            x=x[0]
        x=float(x)
    return x

def load_results(source_dirs = ["AUTO_BATCH_SIZE","linear_"]):
    model_names = [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "Yanhao/simcse-bert-for-patent",
            "ahotrod/electra_large_discriminator_squad2_512",
            "microsoft/deberta-v3-large"
            ]

    run_paths = []    
    for source_dir in source_dirs:
        for model in model_names:
            dir = f"../../../ray_results/{source_dir}{model}/"
            # base_path = f"trainable_*stratify_on={strat_field}*_"
            try:
                run_paths.extend([dir+p for p in os.listdir(dir) if "trainable_" in p])
                run_paths.sort()
            except:
                continue
            # for x in run_paths:
            #     print(x)

    df_list = []

    for p in run_paths:
        try:
            df = pd.read_csv(os.path.join(p, "progress.csv"))
        except:
            continue
        with open(os.path.join(p,"params.json")) as f:
            params = json.load(f)
        features = params["features"]
        stratify_on = params["stratify_on"]
        encoder_lr = params["encoder_lr"]

        try:
            n_fold = params["n_fold"]
        except KeyError: # for runs with no cross validation
            n_fold = 1

        epochs = params["epochs"]
        model = params["model"]
        warmup = params["warmup_steps"]
        try:
            readout = params["readout"]
        except KeyError:
            readout = "attention"
        if encoder_lr == 0.000001:
            print(model)
        if ((model!="distilbert-base-uncased" and len(df) < n_fold*epochs+n_fold) \
            and (n_fold != 1)) and (encoder_lr != 0.000001):
            if encoder_lr == 0.000001:
                print("skip")
            continue
        
        df = df.iloc[-(epochs+1)*n_fold:].reset_index()
        
        
        df["features"] = feature_labels[features]
        df["loss"] = params["loss"]
        df["stratify_on"] = stratify_on
        df["encoder_lr"] = encoder_lr

        if not np.any(df.columns.values == "epoch"): # skips dataframes that already have the epoch column
            df["epoch"] = -2
            df[:4]["epoch"] = -1

            fold_label = list(range(0,n_fold)) if len(df) == n_fold*epochs+n_fold else []
            epoch_label = [-1]*n_fold if len(df) == n_fold*epochs+n_fold else []

            for f in range(n_fold):
                fold_label.extend([f] * epochs)
                epoch_label.extend(list(range(epochs)))
            try:
                df["fold"] = fold_label
                df["epoch"] = epoch_label
            except:
                try:
                    df["fold"] = fold_label[:len(df)]
                    df["epoch"] = epoch_label[:len(df)]
                except:
                    continue

        df["epoch"] = df["epoch"].apply(convert_epoch)

        # time statistics
        required_epochs=0
        total_time=0
        dates = pd.to_datetime(df.date, format='%Y-%m-%d_%H-%M-%S')
        mean_epoch_time = dates.diff().mean().total_seconds()
        total_time = (dates.iloc[-1] - dates.iloc[0]).total_seconds()
        
        try:
            required_epochs = df.groupby("epoch").agg("mean").val_score.idxmax() + 1 
        except:
            try:
                required_epochs = df.groupby("epoch").agg("mean").train_score.idxmax() + 1 
            except:
                continue
        

        df["model"] = model_labels[model]
        df["warmup_steps"] = warmup
        df['required_epochs'] = required_epochs
        df['total_time'] = total_time
        df['mean_epoch_time'] = mean_epoch_time
        df['readout'] = readout
        df_list.append(df)

    return pd.concat(df_list)