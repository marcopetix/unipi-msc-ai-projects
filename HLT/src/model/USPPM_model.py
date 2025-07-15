from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import  AutoModel, AutoConfig
import torch
import torch.nn as nn
import numpy as np
from model.loss import *

def pearson_correlation_score(predictions, labels):
    mean_pred = torch.mean(predictions)
    mean_labels = torch.mean(labels)

    covariance = torch.sum((predictions - mean_pred) * (labels - mean_labels))
    std_pred = torch.sqrt(torch.sum((predictions - mean_pred) ** 2))
    std_labels = torch.sqrt(torch.sum((labels - mean_labels) ** 2))

    pearson_correlation = covariance / (std_pred * std_labels)

    return pearson_correlation

def pearson_correlation_loss(predictions, labels):
    return 1 - pearson_correlation_score(predictions, labels)



class USPPPM_model(LightningModule):
    def __init__(self, config_dict, config_path=None, pretrained=True):
        super().__init__()
        self.save_hyperparameters('config_dict', 'pretrained')
        self.epoch=0
        # print("\n\I'm initialiting the model...")
        
        if config_path is None:
            self.config = AutoConfig.from_pretrained(config_dict['model'], output_hidden_states = True,truncation = True)
            if config_dict["save_configs"]:
                self.config.save_pretrained(f"models/{config_dict['model']}/config/")
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.model = AutoModel.from_pretrained(config_dict['model'], config = self.config)
            if config_dict["save_configs"]:
                self.model.save_pretrained(f"models/{config_dict['model']}/model/")
        else:
            self.model = AutoModel.from_config(self.config)

        self.config_dict = config_dict
        self.n_warmup_steps = config_dict['warmup_steps']
        self.n_training_steps = config_dict['training_steps']

        if config_dict['loss'] == "pearson":
            self.criterion = pearson_correlation_loss
        elif config_dict['loss'] == "bce":
            self.criterion = nn.BCEWithLogitsLoss(reduction="mean")

        self.fc_dropout = nn.Dropout(config_dict['fc_dropout'])
        self.fc = nn.Linear(self.config.hidden_size, config_dict['target_size'])
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        
        self.batch_labels = []
        self._init_weights(self.attention)


    def _init_weights(self, module):
        
        # print("\nIN _init_weights()\n")
        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        del last_hidden_states, weights
        return feature

    def forward(self, inputs=None, labels=None):
        feature = self.feature(inputs)
        if self.config_dict['loss'] == "pearson":
            output = self.fc(self.fc_dropout(feature)).sigmoid()
        elif self.config_dict['loss'] == "bce":
            output = self.fc(self.fc_dropout(feature))        
        
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        
        del feature
        return loss, output
    
    def training_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
        self.log("train_loss", loss, prog_bar=True, logger=True)
        # session.report({"train_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # session.report({"val_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
    def predict_step(self, batch, batch_idx):
        inputs = batch["inputs"]
        labels = batch["labels"]
        cpc_codes = batch["cpc_codes"]
        anchors = batch["anchors"]
        targets = batch["targets"]
        loss, outputs = self(inputs)
        return {"inputs": inputs, "anchors": anchors, "targets": targets, "predictions": outputs.sigmoid(), "labels": labels, "cpc_codes": cpc_codes}
        # return outputs
    
    def test_step(self, batch, batch_idx):
        '''inputs = batch["inputs"]
        labels = batch["labels"]
        loss, outputs = self(inputs, labels.unsqueeze(1))
   
        self.log("test_loss", loss, prog_bar=True, logger=True)
        # session.report({"test_loss": loss})  # Send the score to Tune.
        return {"loss": loss, "predictions": outputs, "labels": labels}'''
        return {}
    

    def test_epoch_end(self, batch_results):
        '''outputs, labels, losses = [], [], []
        for batch in batch_results:
            outputs.append(batch['predictions'])
            labels.append(batch['labels'])
            losses.append(batch['loss'])

        with torch.no_grad():
            labels = torch.cat(labels).cpu().numpy()
            predictions = np.concatenate(torch.cat(outputs).sigmoid().to('cpu').numpy())
            
            score = get_score(labels, predictions)

        self.log("test_score", score, prog_bar=True, logger=True)
        del predictions
        del labels
        del outputs'''
        pass
        # tune.report({"test_score": score})  # Send the score to Tune.

    
    def validation_epoch_end(self, batch_results):
        outputs, labels, losses = [], [], []
        for batch in batch_results:
            outputs.append(batch['predictions'])
            labels.append(batch['labels'])
            losses.append(batch['loss'])

           
        labels = torch.cat(labels).cpu().numpy()
        predictions = np.concatenate(torch.cat(outputs).sigmoid().to('cpu').numpy())

        nans = np.count_nonzero(np.isnan(predictions))
        infs = np.count_nonzero(np.isinf(predictions))

        print(f"Number of NaNs: {nans}")
        print(f"Number of Infs: {infs}")
        score = get_score(labels, np.nan_to_num(predictions))
        self.log("val_score", score, prog_bar=True, logger=True)
        try: self.log("fold", self.current_fold) 
        except: pass
        #self.log("val_score", 1-torch.mean(torch.stack(losses)), prog_bar=True, logger=True)
        # tune.report({"val_score": score})  # Send the score to Tune..
    
    def training_epoch_end(self, batch_results) -> None:
        outputs, labels, losses = [], [], []
        for batch in batch_results:
            outputs.append(batch['predictions'])
            labels.append(batch['labels'])
            losses.append(batch['loss'])

           
        labels = torch.cat(labels).cpu().numpy()
        with torch.no_grad():
            predictions = np.concatenate(torch.cat(outputs).sigmoid().to('cpu').numpy())

        nans = np.count_nonzero(np.isnan(predictions))
        infs = np.count_nonzero(np.isinf(predictions))

        print(f"Number of NaNs: {nans}")
        print(f"Number of Infs: {infs}")
        score = get_score(labels, np.nan_to_num(predictions))
        self.log("train_score", score, prog_bar=True, logger=True)
        self.log("batch_size", self.config_dict['batch_size'])
        self.log("epoch", self.epoch)
        self.epoch+=1

        return super().training_epoch_end(batch_results)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config_dict['encoder_lr'])
       
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
          optimizer=optimizer,
          lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
          )
        )

