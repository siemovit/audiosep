import os
from typing import Dict, List, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from audiosep.data.spectrogram.dataset import VoiceNoiseDataset
from audiosep.data.waveform_dataset import WaveFormVoiceNoiseDataset


class VoiceNoiseDatamodule(L.LightningDataModule):
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

        # placeholders set in setup

        all_folders_train = sorted(
            d
            for d in os.listdir(self.train_data_dir)
            if os.path.isdir(os.path.join(self.train_data_dir, d))
        )

        # create train dataset
        self.train_dataset = VoiceNoiseDataset(root_dir=self.train_data_dir)
        self.train_dataset.example_dirs = all_folders_train

        all_folders_test = sorted(
            d
            for d in os.listdir(self.test_data_dir)
            if os.path.isdir(os.path.join(self.test_data_dir, d))
        )
        self.val_dataset = VoiceNoiseDataset(root_dir=self.test_data_dir)
        self.val_dataset.example_dirs = all_folders_test
        self.test_dataset = WaveFormVoiceNoiseDataset(root_dir=self.test_data_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._pad_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._pad_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # really important
            shuffle=False,
            num_workers=self.num_workers,
        )

    def _pad_collate(self, batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
        """Pad a batch of (mix, targets) so the time dimension (last dim) matches.

        Expects each sample as (mix_mag, {"voice": voice_mag, "noise": noise_mag}).
        Pads with zeros on the right for time dimension.
        """
        mixes = [b[0] for b in batch]
        targets_voice = [b[1]["voice"] for b in batch]
        targets_noise = [b[1]["noise"] for b in batch]

        # find max time length
        max_t = max(m.shape[-1] for m in mixes)

        def pad_to(m: torch.Tensor, t: int) -> torch.Tensor:
            if m.shape[-1] == t:
                return m
            pad_w = t - m.shape[-1]
            # pad (left, right) on last dim -> F.pad expects (pad_left, pad_right, pad_top, pad_bottom)
            return torch.nn.functional.pad(m, (0, pad_w))

        mixes_p = torch.stack([pad_to(m, max_t) for m in mixes], dim=0)
        voice_p = torch.stack([pad_to(v, max_t) for v in targets_voice], dim=0)
        noise_p = torch.stack([pad_to(n, max_t) for n in targets_noise], dim=0)

        return mixes_p, {"voice": voice_p, "noise": noise_p}
