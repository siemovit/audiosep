# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from audiosep.io import load_audio_tensor
RESAMPLE_RATE = 8000


class WaveDataset(Dataset):
    def __init__(self, root_dir, max_len=None):
        """
        root_dir  : dossier avec les sous-dossiers 0001, 0002...
        max_len   : maximum length of audio in samples. If longer, random crop is taken.
        """
        self.root_dir = root_dir
        self.max_len = max_len
        self.example_dirs = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )

    def __len__(self):
        return len(self.example_dirs)

    def __getitem__(self, idx):
        folder = self.example_dirs[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # Mix filename
        mix_file = [f for f in os.listdir(folder_path) if f.startswith("mix")][0]

        # Paths
        mix_path   = os.path.join(folder_path, mix_file)
        voice_path = os.path.join(folder_path, "voice.wav")
        noise_path = os.path.join(folder_path, "noise.wav")

        # Load audios
        if torch.mps.is_available():
            mix, sample_rate   = load_audio_tensor(mix_path, target_sample_rate=RESAMPLE_RATE)
            # print(f"Loaded mix from {mix_path} with sample rate {sample_rate}")
            # print(f"Mix shape: {mix.shape}")
            voice, _ = load_audio_tensor(voice_path, target_sample_rate=RESAMPLE_RATE)
            noise, _ = load_audio_tensor(noise_path, target_sample_rate=RESAMPLE_RATE)
        else:
            mix, sample_rate   = load_audio_tensor(mix_path)
            voice, _ = load_audio_tensor(voice_path)
            noise, _ = load_audio_tensor(noise_path)
        
        # Random crop if needed
        if self.max_len is not None and mix.shape[-1] > self.max_len:
            max_start = mix.shape[-1] - self.max_len
            start = torch.randint(0, max_start + 1, (1,)).item()
            end = start + self.max_len
            mix = mix[..., start:end]
            voice = voice[..., start:end]
            noise = noise[..., start:end]

        return mix, {"voice": voice, "noise": noise}