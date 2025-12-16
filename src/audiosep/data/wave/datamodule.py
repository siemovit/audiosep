import lightning as L
from torch.utils.data import DataLoader

from audiosep.data.wave.dataset import WaveDataset
from audiosep.data.waveform_dataset import WaveFormVoiceNoiseDataset


class WaveDatamodule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        batch_size: int = 16,
        num_workers: int = 7,
        seed: int = 42,
    ):
        super().__init__()
        # data directories
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        # data loading params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = int(seed)

        # placeholders set in setup
        self.train_dataset = WaveDataset(root_dir=self.train_data_dir)
        self.val_dataset = WaveFormVoiceNoiseDataset(root_dir=self.test_data_dir)
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
