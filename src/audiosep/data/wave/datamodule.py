import lightning as L
from torch.utils.data import DataLoader
from .dataset import WaveDataset
import os
import torch
from typing import List, Tuple, Dict

class WaveDatamodule(L.LightningDataModule):
    def __init__(self, train_data_dir: str, test_data_dir: str = None, batch_size: int = 16, num_workers: int = 7, val_split: float = 0.2, seed: int = 42, max_len: int = 32000):
        super().__init__()
        # data directories
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir 
        
        # data loading params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.max_len = max_len

        # placeholders set in setup
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        # list example folders and split deterministically
        if stage == 'fit':
            all_folders_train = sorted(
                d for d in os.listdir(self.train_data_dir)
                if os.path.isdir(os.path.join(self.train_data_dir, d))
            )
            
            # create train dataset
            self.train_dataset = WaveDataset(root_dir=self.train_data_dir, max_len=self.max_len)
            self.train_dataset.example_dirs = all_folders_train
            
        if self.test_data_dir is not None:
            all_folders_test = sorted(
                d for d in os.listdir(self.test_data_dir)
                if os.path.isdir(os.path.join(self.test_data_dir, d))
            )
            self.test_dataset = WaveDataset(root_dir=self.test_data_dir, max_len=self.max_len)
            self.test_dataset.example_dirs = all_folders_test
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=self._pad_collate,
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self._pad_collate,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers)

    # def _pad_collate(self, batch):
    #     mixes = [b[0] for b in batch]
    #     voices = [b[1]['voice'] for b in batch]
    #     noises = [b[1]['noise'] for b in batch]
    #     max_t = max(m.shape[-1] for m in mixes)
    #     def pad_list(arrs):
    #         outs = []
    #         for a in arrs:
    #             pad = max_t - a.shape[-1]
    #             if pad > 0:
    #                 a = F.pad(a, (0, pad))
    #             outs.append(a)
    #         return torch.stack(outs, dim=0)

    #     mixes = pad_list(mixes)
    #     voices = pad_list(voices)
    #     noises = pad_list(noises)
    #     targets = {'voice': voices, 'noise': noises}
    #     return mixes, targets

