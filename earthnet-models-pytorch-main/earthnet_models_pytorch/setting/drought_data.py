
from typing import Union, Optional

import argparse
import copy
import multiprocessing
import re

from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from earthnet_models_pytorch.utils import str2bool

class DroughtDataset(Dataset):

    def __init__(self, folder: Union[Path, str]):
        if not isinstance(folder, Path):
            folder = Path(folder)
        assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))

        self.filepaths = sorted(list(folder.glob("*.npz")))
        self.min_vals = list(folder.glob("*_min.npy"))
        self.max_vals = list(folder.glob("*_max.npy"))
        # Exclude min and max files from self.filepaths
        self.filepaths = [file for file in self.filepaths if file not in self.min_vals and file not in self.max_vals]

        # Compute global min/max for training set
        self.min_stats = np.min([np.load(i) for i in self.min_vals], axis=0) if self.min_vals else None
        self.max_stats = np.max([np.load(i) for i in self.max_vals], axis=0) if self.max_vals else None

        self.type = np.float32

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        npz = np.load(filepath)

        context = npz["context"].reshape((npz["context"].shape[1],npz["context"].shape[0], 1, 1)) # seq_len, input_size/num features, height, width
        target =  npz["target"].reshape((npz["target"].shape[1],npz["target"].shape[0], 1, 1))
        if self.min_stats is None or self.max_stats is None:
            self.min_stats = np.ones((context.shape[1],))
            self.max_stats = np.zeros((context.shape[1],))
            min_stats = self.min_stats[np.newaxis, :, np.newaxis, np.newaxis]
            max_stats = self.max_stats[np.newaxis, :, np.newaxis, np.newaxis]
            self.min_stats, self.max_stats = None, None
        else:
            min_stats = self.min_stats[np.newaxis, :, np.newaxis, np.newaxis]
            max_stats = self.max_stats[np.newaxis, :, np.newaxis, np.newaxis]

        # Normalise
        context_normalized = (context - min_stats)/(max_stats-min_stats)
        target_normalized = (target - min_stats)/(max_stats-min_stats)

        data = {
            "context": [
                torch.from_numpy(context_normalized)
            ],
            "target": [
                torch.from_numpy(target_normalized)
            ],
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath)
        }

        npz.close()

        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format startyear_startmonth_startday_endyear_endmonth_endday_lon_lat_xsize_ysize_shift.npz)
        """        

        pattern = r'\d{4}.*(?=\.npz)'
        match = re.search(pattern, path.name)  
        if match:
            cubename = match.group(0)
            return cubename
        else:
            return None 
        


class DroughtDataModule(pl.LightningDataModule):

    __TRACKS__ = {
        "iid": ("test/iid/","iid_test")
    }

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(copy.deepcopy(hparams))
        else:
            self.hparams = copy.deepcopy(hparams)
        self.base_dir = Path(hparams.base_dir)
        
    @staticmethod
    def add_data_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser,list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]
        
        parser = argparse.ArgumentParser(parents = parent_parser, add_help = False)

        parser.add_argument('--base_dir', type = str, default = "data/datasets/")
        parser.add_argument('--test_track', type = str, default = "iid")

        parser.add_argument('--val_pct', type = float, default = 0.05)
        parser.add_argument('--val_split_seed', type = float, default = 42)

        parser.add_argument('--train_batch_size', type = int, default = 1)
        parser.add_argument('--val_batch_size', type = int, default = 1)
        parser.add_argument('--test_batch_size', type = int, default = 1)

        parser.add_argument('--num_workers', type = int, default = multiprocessing.cpu_count())

        return parser
    
    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            data_corpus = DroughtDataset(self.base_dir/"train")
            
            val_size = int(self.hparams.val_pct * len(data_corpus))
            train_size = len(data_corpus) - val_size

            try: #PyTorch 1.5 safe....
                self.data_train, self.data_val = random_split(data_corpus, [train_size, val_size], generator=torch.Generator().manual_seed(int(self.hparams.val_split_seed)))
            except TypeError:
                self.data_train, self.data_val = random_split(data_corpus, [train_size, val_size])

        if stage == 'test' or stage is None:
            self.data_test = DroughtDataset(self.base_dir/self.__TRACKS__[self.hparams.test_track][0])
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=self.hparams.train_batch_size, num_workers = self.hparams.num_workers,pin_memory=True,drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.data_val, batch_size=self.hparams.val_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.data_test, batch_size=self.hparams.test_batch_size, num_workers = self.hparams.num_workers, pin_memory=True)