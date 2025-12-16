import os

import lightning as L
from torch.utils.data import DataLoader

from audiosep.data.spectrogram_orig.spectogram_dataset import OriginalVoiceNoiseDataset
from audiosep.data.waveform_dataset import WaveFormVoiceNoiseDataset


class OriginalVoiceNoiseDatamodule(L.LightningDataModule):
    """Simple datamodule for u-net on spectrograms"""

    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        batch_size: int = 16,
        num_workers: int = 7,
        val_split: float = 0.2,
        seed: int = 42,
    ):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.train_dataset = OriginalVoiceNoiseDataset(root_dir=self.train_data_dir)
        self.test_dataset = WaveFormVoiceNoiseDataset(root_dir=self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # really important
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # really important
            shuffle=False,
            num_workers=self.num_workers,
        )
