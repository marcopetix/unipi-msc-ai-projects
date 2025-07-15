from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import  AutoModel, AutoConfig
import torch
import torch.nn as nn
from torch.nn import LSTM
import numpy as np

from model.loss import *

class USPPM_model_lstm(LightningModule):
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

        self.dropout = nn.Dropout(config_dict['fc_dropout'])
        self.fc = nn.Linear(self.config.hidden_size, config_dict['target_size'])
        self._init_weights(self.fc)
        self.lstm = nn.LSTM(input_size=self.config.hidden_size,hidden_size=self.config.hidden_size)
                
        self.batch_labels = []
        self._init_weights(self.lstm)


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
        last_hidden_state_cls = outputs[0][:,0]
        lstm_out, _ = self.lstm(last_hidden_state_cls.view(len(last_hidden_state_cls), 1, -1))
        out = self.dropout(lstm_out)
        
        del last_hidden_state_cls
        return out

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        if self.config_dict['loss'] == "pearson":
            output = self.fc(feature).sigmoid()
        elif self.config_dict['loss'] == "bce":
            output = self.fc(feature)        
        
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
        # print("*"*100)
        # print("CONFIGURE_OPTIMIZERS()")
        # print(self.parameters)
        # print("*"*100)
        # optimizer = AdamW(self.parameters(), lr=2e-5)
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

'''        def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_parameters = [
                {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],'lr': encoder_lr, 'weight_decay': weight_decay},
                {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],'lr': encoder_lr, 'weight_decay': 0.0},
                {'params': [p for n, p in model.named_parameters() if "model" not in n],'lr': decoder_lr, 'weight_decay': 0.0}
            ]
            return optimizer_parameters

        optimizer_parameters = get_optimizer_params(self,
                                            encoder_lr=self.config_dict["encoder_lr"], 
                                            decoder_lr=self.config_dict["decoder_lr"],
                                            weight_decay=self.config_dict["weight_decay"])

        optimizer = AdamW(optimizer_parameters, 
                          lr=self.config_dict["encoder_lr"], 
                          eps=self.config_dict["eps"],
                          betas=self.config_dict["betas"])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )'''

