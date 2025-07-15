from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from model.USPPM_dataset import USPPM_dataset

    
@dataclass
class USPPPM_datamodule(LightningDataModule):
    def __init__(self, config_dict, train_df, test_df, train_val_split):
        
        self.config_dict = config_dict
        self.train_val_split = train_val_split
        self.prepare_data_per_node = False
        self._log_hyperparams = False
        train_dataset: Optional[Dataset] = None
        test_dataset: Optional[Dataset] = None
        val_dataset: Optional[Dataset] = None
        self.train_df = train_df
        self.test_df = test_df

        
            
    def setup(self, stage: Optional[str] = None) -> None:
        # train_df, test_df = train_test_split(self.dataframe, test_size = 0.1, random_state = CFG.seed, stratify = self.dataframe.score_map)
        if self.train_val_split == 1:
            self.val_dataset = self.train_dataset = USPPM_dataset(self.config_dict, self.train_df)
        else: 
            train_df, val_df = train_test_split(self.train_df, test_size = (1-self.train_val_split), random_state = self.config_dict["seed"])
        
            self.train_dataset = USPPM_dataset(self.config_dict, train_df)
            self.val_dataset = USPPM_dataset(self.config_dict, val_df)
            
        self.test_dataset = USPPM_dataset(self.config_dict, self.test_df, train=False)

    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, 
                          num_workers = self.config_dict['num_workers'], 
                          batch_size = self.config_dict['batch_size'])

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, 
                          num_workers = self.config_dict['num_workers'], 
                          batch_size = self.config_dict['batch_size'])
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, 
                        num_workers = self.config_dict['num_workers'], 
                        batch_size = self.config_dict['batch_size'])

    
    def __post_init__(cls):
        super().__init__()