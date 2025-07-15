from pytorch_lightning import LightningModule
from typing import Any, Dict, List, Optional, Type
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from model.USPPM_kfold_datamodule import BaseKFoldDataModule
from copy import deepcopy
import os
import torch
from model.loss import get_score
import numpy as np

class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str], config_dict):
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        # print("*"*100)
        # print("CHECKPOINT_PATHS: ")
        # print(checkpoint_paths)
        # print("*"*100)
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p, config_dict=config_dict) for p in checkpoint_paths])
        self.get_score = get_score

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        inputs = batch["inputs"]
        labels = batch["labels"]
        
        losses = []
        scores = []
        
        for m in self.models : 
            with torch.no_grad():
                loss, outputs = m(inputs, labels.unsqueeze(1))
                predictions = outputs.squeeze().sigmoid().cpu().detach().numpy()
                
                score = self.get_score(labels.cpu().numpy(), predictions)
                scores.append(score)
                losses.append(loss.cpu().detach().numpy())

                del(m)
            
        avg_loss = np.mean(losses)
        avg_score = np.mean(scores)
        
        self.log("ensemble_avg_loss", avg_loss, prog_bar=True, logger=True)
        self.log("ensemble_avg_score", avg_score, prog_bar=True, logger=True)
    
    
class KFoldLoop(Loop):
    def __init__(self, num_folds: int, config_dict, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path
        self.config_dict = config_dict

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds(self.num_folds)
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

        # print("*"*100)
        # print(f"STATE_DICT DEEPCOPY FROM on_run_start()")
        # print(self.lightning_module_state_dict)
        # print("*"*100)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        self.trainer.model.current_fold = self.current_fold
        self.trainer.model.epoch = 0
        # print("*"*100)
        # print(f"STATE_DICT FROM on_advance_start() FOLD_{self.current_fold}")
        # print(self.trainer.lightning_module.state_dict())
        # print("*"*100)
        
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current fold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()
        

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
                
    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(os.path.join(self.export_path, f"model.{self.current_fold}.pt"))
        self.current_fold += 1  # increment fold tracking number.
        
        # print("*"*100)
        # print(f"STATE_DICT FROM on_advance_end() FOLD_{self.current_fold}")
        # print(self.trainer.lightning_module.state_dict())
        # print("*"*100)
        
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict) 
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [os.path.join(self.export_path, f"model.{f_idx}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths, self.config_dict)
        voting_model.trainer = self.trainer

        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        print("Starting testing...")
        self.trainer.test_loop.run()

        del voting_model

    def on_save_checkpoint(self) -> Dict[str, int]:
        # print("*"*100)
        # print("IN: on_save_checkpoint()")
        # print("*"*100)
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]
        # print("*"*100)
        # print("IN: on_load_checkpoint()")
        # print(state_dict["current_fold"])
        # print("*"*100)
        
    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)