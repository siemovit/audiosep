# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from audiosep.data.spectrogram.spectrogram import wav_to_mag_phase


class VoiceNoiseDataset(Dataset):
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

        self.example_dirs = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )

    def __len__(self):
        return len(self.example_dirs)

    def __getitem__(self, idx):
        return self._load_item(idx)

    def _load_item(self, idx):
        folder = self.example_dirs[idx]
        folder_path = os.path.join(self.root_dir, folder)

        # filenames
        mix_file = [f for f in os.listdir(folder_path) if f.startswith("mix")][0]

        mix_path = os.path.join(folder_path, mix_file)
        voice_path = os.path.join(folder_path, "voice.wav")
        noise_path = os.path.join(folder_path, "noise.wav")

        # cached filenames (déjà normalisés)
        mix_cache = os.path.join(self.cache_dir, f"{folder}_mix.pt")
        voice_cache = os.path.join(self.cache_dir, f"{folder}_voice.pt")
        noise_cache = os.path.join(self.cache_dir, f"{folder}_noise.pt")

        load_from_cache = all(
            os.path.exists(p) for p in [mix_cache, voice_cache, noise_cache]
        )

        if load_from_cache:
            mix_mag = torch.load(mix_cache)
            voice_mag = torch.load(voice_cache)
            noise_mag = torch.load(noise_cache)

        else:
            # --- calcul des spectrogrammes sans normalisation ---
            mix_mag, _ = wav_to_mag_phase(mix_path)
            voice_mag, _ = wav_to_mag_phase(voice_path)
            noise_mag, _ = wav_to_mag_phase(noise_path)

            # --- normalisation commune basée sur le mix ---
            scale = mix_mag.max() + 1e-8  # protège contre silence total

            mix_mag = mix_mag / scale
            voice_mag = voice_mag / scale
            noise_mag = noise_mag / scale

            # ajout dimension canal (1, F, T)
            mix_mag = mix_mag.unsqueeze(0)
            voice_mag = voice_mag.unsqueeze(0)
            noise_mag = noise_mag.unsqueeze(0)

            # sauvegarde en cache
            torch.save(mix_mag, mix_cache)
            torch.save(voice_mag, voice_cache)
            torch.save(noise_mag, noise_cache)

        return mix_mag, {"voice": voice_mag, "noise": noise_mag}
