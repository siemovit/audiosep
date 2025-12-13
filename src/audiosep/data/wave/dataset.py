# src/dataset.py
import os
from typing import NamedTuple
import torch
from torch.utils.data import Dataset
from audiosep.io import load_audio_tensor
from audiosep.utils import center_crop

class WaveDataset(Dataset):
    def __init__(self, root_dir, out_len: int = 16384, context: int = 4096, self_thr: float = 1e-4):
        """
        root_dir  : dossier avec les sous-dossiers 0001, 0002...
        out_len   : supervised length (center region)
        context   : extra context on each side for the model input
        """
        self.root_dir = root_dir
        self.out_len = int(out_len)
        self.context = int(context)
        self.in_len = self.out_len + 2 * self.context
        self.thr = self_thr # speech-aware threshold (to tune) [range 5e-5, 2e-4 ?]
        self.speech_aware = True
        self.tries = 10

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
        mix, _   = load_audio_tensor(mix_path)
        voice, _ = load_audio_tensor(voice_path) # shape (1, T)
        noise, _ = load_audio_tensor(noise_path) # shape (1, T)
        
        T = mix.shape[-1] 

        # normalize by mix peak (preserves mix = voice + noise)
        gain = 1.0 / (mix.abs().max() + 1e-8)
        mix   = mix * gain
        voice = voice * gain
        noise = noise * gain
        
        # If too short, pad (important for short clips)
        if T < self.in_len:
            pad = self.in_len - T
            mix = torch.nn.functional.pad(mix, (0, pad))
            voice = torch.nn.functional.pad(voice, (0, pad))
            noise = torch.nn.functional.pad(noise, (0, pad))
            T = self.in_len

        # Random window of length in_len
        max_start = T - self.in_len
        
        # speech-aware sampling (to ensure voice activity in segment)
        start = torch.randint(0, max_start + 1, (1,)).item() # initial random start
        if self.speech_aware:
            best = start # initial best start
            for _ in range(self.tries):
                s = torch.randint(0, max_start + 1, (1,)).item()
                v_seg = voice[..., s + self.context : s + self.context + self.out_len] # where voice is evaluated in model output
                if (v_seg ** 2).mean().item() > self.thr: # has "enough speech"
                    best = s
                    break
                best = s # keep last if none found
            start = best

        # input segment with context
        end = start + self.in_len
        mix_in = mix[..., start:end]

        # targets aligned with context explicitly
        tgt0 = start + self.context
        voice_out = voice[..., tgt0:tgt0 + self.out_len]
        noise_out = noise[..., tgt0:tgt0 + self.out_len]

        return mix_in, {"voice": voice_out, "noise": noise_out}
