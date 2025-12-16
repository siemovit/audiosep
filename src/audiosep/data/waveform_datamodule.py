import lightning as L
from torch.utils.data import DataLoader
from audiosep.data import WaveFormVoiceNoiseDataset
import os
import torch
from typing import List, Tuple, Dict

class WaveFormVoiceNoiseDatamodule(L.LightningDataModule):
    def __init__(self, 
                 root_dir: str = None, 
                 batch_size: int = 1, 
                 num_workers: int = 0, 
                 seed: int = 42, 
            ):
        super().__init__()
        # data directories
        self.root_dir = root_dir 
        
        # data loading params
        self.batch_size = batch_size
        self.seed = int(seed)

        # placeholders set in setup
        self.test_dataset = WaveFormVoiceNoiseDataset(root_dir=self.root_dir)
                
    def setup(self, stage="test"):     
        self.root_dir.audio_sample_dirs = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1, # really important
            shuffle=False,
            num_workers=self.num_workers,
        )


