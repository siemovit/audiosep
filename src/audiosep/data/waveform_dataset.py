import os
from typing import NamedTuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class WaveFormVoiceNoiseBatch(NamedTuple):
    mix: torch.Tensor
    voice: torch.Tensor
    noise: torch.Tensor
    mix_filename: str


class WaveFormVoiceNoiseDataset(Dataset):
    """Simple dataset"""

    def __init__(self, root_dir):
        """
        root_dir  : dossier avec les sous-dossiers 0001, 0002...
        cache_dir : dossier où stocker spectrogrammes pré-calculés (normalisés)
        """
        self.root_dir = root_dir

        self.audio_sample_dirs = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )

    def __len__(self):
        return len(self.audio_sample_dirs)

    def __getitem__(self, idx):
        return self._load_item(idx)

    def _load_item(self, idx):
        folder = self.audio_sample_dirs[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # filenames
        mix_file = [f for f in os.listdir(folder_path) if f.startswith("mix")][0]

        mix_path = os.path.join(folder_path, mix_file)
        voice_path = os.path.join(folder_path, "voice.wav")
        noise_path = os.path.join(folder_path, "noise.wav")
        wave_mix, _ = librosa.load(mix_path, sr=None, mono=True)
        wave_voice, _ = librosa.load(voice_path, sr=None, mono=True)
        wave_noise, _ = librosa.load(noise_path, sr=None, mono=True)
        return WaveFormVoiceNoiseBatch(
            torch.from_numpy(wave_mix.astype(np.float32)),
            torch.from_numpy(wave_voice.astype(np.float32)),
            torch.from_numpy(wave_noise.astype(np.float32)),
            mix_file,
        )
