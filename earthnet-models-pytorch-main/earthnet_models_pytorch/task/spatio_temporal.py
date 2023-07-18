
from typing import Optional, Union

import argparse
import ast
import copy
import json
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as clr
import xarray as xr

import torch
import torchvision

import pytorch_lightning as pl 

from torch import nn

from pathlib import Path
import shutil

from earthnet_models_pytorch.utils import str2bool, log_viz
from earthnet_models_pytorch.task import setup_loss, SHEDULERS
from earthnet_models_pytorch.setting import METRICS

class SpatioTemporalTask(pl.LightningModule):

    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()
        
        if hasattr(self, "save_hyperparameters"):  # SpatioTemporalTask inherits from the LightningModule
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)
        self.model = model

        if hparams.pred_dir is None:
            self.pred_dir = Path(self.logger.log_dir)/"predictions" if self.logger is not None else Path.cwd()/"experiments"/"predictions"  # logger: hyperparameter of LightningModule for the Trainer
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        #self.n_stochastic_preds = hparams.n_stochastic_preds 

        self.current_filepaths = []

        self.metric = METRICS[self.hparams.setting]()
        
        self.pred_mode = {"drought":"ndvi"}[self.hparams.setting]

        self.model_shedules = []
        for shedule in self.hparams.model_shedules:
            self.model_shedules.append((shedule["call_name"], SHEDULERS[shedule["name"]](**shedule["args"])))


    @staticmethod
    def add_task_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):  # Optional[X] is equivalent to Union[X, None].
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)  # parents - A list of ArgumentParser objects whose arguments should also be included

        parser.add_argument('--pred_dir', type = str, default = None)

        parser.add_argument('--loss', type = ast.literal_eval, default = '{"name": "masked", "args": {"distance_type": "L1"}}')

        parser.add_argument('--context_length', type = int, default = 3)
        parser.add_argument('--target_length', type = int, default = 2)

        #parser.add_argument('--n_stochastic_preds', type = int, default = 10)

        parser.add_argument('--n_log_batches', type = int, default = 2)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--optimization', type = ast.literal_eval, default = '{"optimizer": [{"name": "Adam", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], "lr_shedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}')

        parser.add_argument('--model_shedules', type = ast.literal_eval, default = '[]')

        parser.add_argument('--setting', type = str, default = "en21-std")

        parser.add_argument('--compute_metric_on_test', type = str2bool, default = False)
        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None, kwargs = {}):
        """
        data is a dict with tensors
        pred_start is the first index that shall be predicted, defaults to zero.
        n_preds is the length of the prediction, could also be None.
        kwargs are optional keyword arguments parsed to the model, right now these are model shedulers.
        """    
        x = data["context"][0]   
        return self.model(x.float(), **kwargs)

    def configure_optimizers(self):
        optimizers = [getattr(torch.optim,o["name"])(self.parameters(), **o["args"]) for o in self.hparams.optimization["optimizer"]] # This gets any (!) torch.optim optimizer
        # torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs. 
        shedulers = [getattr(torch.optim.lr_scheduler,s["name"])(optimizers[i], **s["args"]) for i, s in enumerate(self.hparams.optimization["lr_shedule"])] # This gets any(!) torch.optim.lr_scheduler - but only those with standard callback will work (i.e. not the Plateau one)
        return optimizers, shedulers

    
    def training_step(self, batch, batch_idx):
        '''compute and return the training loss and some additional metrics for e.g. the progress bar or logger'''
        kwargs = {}
        for (shedule_name, shedule) in self.model_shedules:
            kwargs[shedule_name] = shedule(self.global_step)
        
        # Predictions generation
        preds = self(batch)
        loss, logs = self.loss(preds, batch, current_step = self.global_step)

        # Logs
        for shedule_name in kwargs:
            if len(kwargs[shedule_name]) > 1:
                for i, shed_val in enumerate(kwargs[shedule_name]):
                    logs[f"{shedule_name}_i"] = shed_val
            else:
                logs[shedule_name] = kwargs[shedule_name]
        logs['batch_size_train'] = torch.tensor(self.hparams.train_batch_size, dtype=torch.float32)
        logs['loss_train'] = loss.clone().detach() #torch.tensor(loss, dtype=torch.float32) # already in logs normally
        self.log_dict(logs)  
        
        return loss

    def validation_step(self, batch, batch_idx):
        '''Operates on a single batch of data from the validation set. In this step you d might generate examples or calculate anything of interest like accuracy.'''
        
        preds = self(batch) #self(x.float()) #, pred_start = self.context_length, n_preds = self.target_length)  # output model
        loss, logs = self.loss(preds, batch)
        metric = self.metric(preds, batch["target"][0][:, :, self.hparams["loss"]["ndvi_pred_idx"],...]) # Check that shapes are ok
        batch_size = torch.tensor(self.hparams.val_batch_size, dtype=torch.int64)#float32)
        logs['val_batch_size'] = torch.tensor(self.hparams.val_batch_size, dtype=torch.float32)
        logs['loss_val'] = torch.tensor(loss.clone().detach(), dtype=torch.float32)
        self.log_dict(logs)

        
    def on_validation_epoch_end(self):

        current_scores = self.metric.compute()
        self.log_dict(current_scores, sync_dist=True) # is a dict with {"metric name": val}
        self.metric.reset()

        if self.logger is not None and type(self.logger.experiment).__name__ != "DummyExperiment" and self.trainer.is_global_zero:
            current_scores["epoch"] = self.current_epoch
            current_scores = {k: str(v.detach().cpu().item())  if isinstance(v, torch.Tensor) else str(v) for k,v in current_scores.items()}
            outpath = Path(self.logger.log_dir)/"validation_scores.json"
            if outpath.is_file():
                with open(outpath, "r") as fp:
                    past_scores = json.load(fp)
                scores = past_scores + [current_scores]
            else:
                scores = [current_scores]        

            with open(outpath, "w") as fp:
                json.dump(scores, fp)


    def test_step(self, batch, batch_idx):
        '''Operates on a single batch of data from the test set. In this step you'd normally generate examples or calculate anything of interest such as accuracy.'''
        scores = [] # For saving metric
        preds = self(batch) # shape batch size, time len, feature
        # Compute and log loss
        loss, logs = self.loss(preds, batch)
        self.log('loss_test', torch.tensor(loss, dtype=torch.float32).clone().detach(), batch_size=self.hparams.test_batch_size)

        # Loop through batch of results
        for j in range(preds.shape[0]):
            targ_path = Path(batch["filepath"][j])
            # Extract ndvi preds
            ndvi_preds = preds[j].detach().cpu().numpy()
            # Save predicitons
            pred_dir = self.pred_dir
            pred_path = pred_dir/targ_path.parent.stem/targ_path.name
            pred_path.parent.mkdir(parents = True, exist_ok = True)
            if not pred_path.is_file():
                np.savez(pred_path, preds=ndvi_preds)
            #scores.append((targ_path, self.metric(ndvi_preds, batch["target"][0][j, :, self.hparams["loss"]["ndvi_pred_idx"],...])))
            scores.append((str(targ_path.name), 0.5)) # TOCHANGE
        self.indiv_test_scores = scores

        if self.hparams.compute_metric_on_test:
            metric = self.metric(preds, batch["target"][0][:, :, self.hparams["loss"]["ndvi_pred_idx"],...])

        return
           

    def on_test_epoch_end(self):
        '''Called at the end of a test epoch with the output of all test steps.'''

        if self.hparams.compute_metric_on_test:
  
            # Save individual metric per test sample. Need to have them in the test_step and pass them here
            self.pred_dir.mkdir(parents = True, exist_ok = True)
            with open(self.pred_dir/f"individual_scores_{self.global_rank}.json", "w") as fp:
                json.dump({k: v for k,v in self.indiv_test_scores}, fp)
            
            # Save and log overall test metric
            current_scores = self.metric.compute()
            self.log_dict({l+"_test": current_scores[l] for l in current_scores}, sync_dist=True) # is a dict with {"metric name": val}
            self.metric.reset()

            if self.trainer.is_global_zero:
                with open(self.pred_dir/"total_score.json", "w") as fp:
                    json.dump({k: v if isinstance(v, str) else v.item() for k,v in current_scores.items()}, fp)
    
    def teardown(self, stage):
        if stage == "test" and self.hparams.compute_metric_on_test:
            if self.global_rank == 0:
                data = {}
                for path in self.pred_dir.glob("individual_scores_*.json"):
                    with open(path, "r") as fp:
                        data.update(json.load(fp)) # Gets the keys (i.e. the indiv file names)
                    path.unlink(missing_ok=True)
                if len(data) == 0:
                    return

                # Combine all to one file, write scroes only once
                dict_out = {}
                out_names = []
                out = []
                for d in data:
                    if d not in out_names:
                        dict_out[d]= data[d]
                        out_names.append(d)
                        out.append(data[d])
                with open(self.pred_dir/f"individual_scores.json", "w") as fp:
                    json.dump(dict_out, fp)

        return
   