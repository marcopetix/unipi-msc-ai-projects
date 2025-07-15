from transformers import AutoTokenizer, AutoModel, AutoConfig
# PyTorch
import torch

from torch.utils.data import Dataset, DataLoader, Subset
from pytorch_lightning import LightningModule
from tqdm.auto import tqdm

def set_tokenizer(config_dict):
    tokenizer = AutoTokenizer.from_pretrained(config_dict['model'], truncation=True)
    if config_dict["save_configs"]:
        tokenizer.save_pretrained(f"./models/{config_dict['model']}/tokenizer/")
    
    return tokenizer
      
def set_max_len(config_dict, train_df):
    tokenizer = AutoTokenizer.from_pretrained(config_dict['model'], truncation=True)
    
    lengths = []
    if config_dict["features"] == None:
        tk0 = tqdm((train_df.anchor + "[SEP]" + train_df.target).values, total=len(train_df))
    elif config_dict["features"] == "cpc_code":
        tk0 = tqdm((train_df.anchor + "[SEP]" + train_df.target + "[SEP]" + train_df['context']).values, total=len(train_df))
    else:
        tk0 = tqdm(train_df[config_dict["features"]].fillna("").values, total=len(train_df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
            
    config_dict['max_len'] = min(512, max(lengths))
    
    
def prepare_input(config_dict, text, tokenizer):
    inputs = tokenizer(text,
                       add_special_tokens = True,
                       padding = "max_length",
                       truncation = True,
                       max_length = config_dict['max_len'],
                       return_offsets_mapping = False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class USPPM_dataset(Dataset):
    def __init__(self, config_dict, train_df, train=True):
        self.config_dict = config_dict
        if self.config_dict["features"] == None:
            self.texts = (train_df.anchor + "[SEP]" + train_df.target).values
        elif config_dict["features"] == "cpc_code":
            self.texts = (train_df.anchor + "[SEP]" + train_df.target + "[SEP]" + train_df['context']).values
        else:
            self.texts = train_df[config_dict["features"]].values
        self.cpc_codes =train_df['context'].values
        self.anchors = train_df['anchor'].values
        self.targets = train_df['target'].values
        self.train = train
        self.tokenizer = set_tokenizer(config_dict)
        try:
            self.stratify_on = train_df[config_dict["stratify_on"]]
        except:
            pass
        if train:
            self.labels = train_df['score'].values
            self.score_map = train_df['score_map'].values
        
        self.anchor = train_df['anchor'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.config_dict, self.texts[item], self.tokenizer)
        if self.train:
            labels = torch.tensor(self.labels[item], dtype=torch.float)
            return dict(
                  inputs = inputs,
                  labels = labels,
                  cpc_codes = self.cpc_codes[item],
                  anchors = self.anchors[item],
                  targets = self.targets[item] 
            )
        else:
            return dict(inputs=inputs, cpc_codes = self.cpc_codes[item], anchors = self.anchors[item], targets = self.targets[item])