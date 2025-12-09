# src/dataset.py
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from audiosep.data.spectrogram_orig.spectrogram import wav_to_mag_phase


class OriginalVoiceNoiseDataset(Dataset):
    """Original Spectrogram Voice-Noise Dataset"""

    def __init__(self, root_dir, cache_dir="cache"):
        """
        root_dir  : dossier avec les sous-dossiers 0001, 0002...
        cache_dir : dossier où stocker spectrogrammes pré-calculés (normalisés)
        """
        self.root_dir = root_dir

        # use provided cache_dir but default to '<root_dir>_cache' when None or 'cache'
        if cache_dir is None or cache_dir == "cache":
            self.cache_dir = root_dir + "_cache"
        else:
            self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

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

        # cached filenames (déjà normalisés)
        mix_cache = os.path.join(self.cache_dir, f"{folder}_mix.pt")
        voice_cache = os.path.join(self.cache_dir, f"{folder}_voice.pt")

        load_from_cache = all(os.path.exists(p) for p in [mix_cache, voice_cache])

        if load_from_cache:
            mix = torch.load(mix_cache)
            voc = torch.load(voice_cache)

        else:
            # To tensor

            # --- calcul des spectrogrammes sans normalisation ---
            mix, mix_phase = wav_to_mag_phase(mix_path)
            voc, voc_phase = wav_to_mag_phase(voice_path)

            start = random.randint(0, mix.shape[-1] - 128 - 1)
            mix = mix[1:, start : start + 128, np.newaxis]
            voc = voc[1:, start : start + 128, np.newaxis]
            mix = np.asarray(mix, dtype=np.float32)
            voc = np.asarray(voc, dtype=np.float32)
            scale = mix.max()

            mix = mix / scale
            voc = voc / scale

            mix = torch.from_numpy(mix).permute(2, 0, 1)
            voc = torch.from_numpy(voc).permute(2, 0, 1)
            # --- normalisation commune basée sur le mix ---

            # sauvegarde en cache
            torch.save(mix, mix_cache)
            torch.save(voc, voice_cache)
            torch.save(mix_phase, mix_cache.replace("_mix.pt", "_mix_phase.pt"))
            torch.save(voc_phase, voice_cache.replace("_voice.pt", "_voice_phase.pt"))

        return mix, {"voice": voc}
