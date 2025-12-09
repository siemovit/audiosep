import os

import lightning as L
from torch.utils.data import DataLoader

from audiosep.data.spectrogram_orig.dataset import OriginalVoiceNoiseDataset


class OriginalVoiceNoiseDatamodule(L.LightningDataModule):
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
        # data directories
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        # data loading params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = float(val_split)
        self.seed = int(seed)
        self.train_dataset = OriginalVoiceNoiseDataset(root_dir=self.train_data_dir)
        self.test_dataset = OriginalVoiceNoiseDataset(root_dir=self.test_data_dir)

    def setup(self, stage=None):
        # list example folders and split deterministically
        if stage == "fit":
            all_folders_train = sorted(
                d
                for d in os.listdir(self.train_data_dir)
                if os.path.isdir(os.path.join(self.train_data_dir, d))
            )

            # create train dataset
            self.train_dataset.audio_sample_dirs = all_folders_train

        if self.test_data_dir is not None:
            all_folders_test = sorted(
                d
                for d in os.listdir(self.test_data_dir)
                if os.path.isdir(os.path.join(self.test_data_dir, d))
            )
            self.test_dataset.audio_sample_dirs = all_folders_test

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
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()
