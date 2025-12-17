"""Create mix files at specified SNRs for each sample folder under data/test_pline.
IMPORTANT: requires that noise.wav files are already present in each folder (did it by hand here).
"""
from pathlib import Path
import soundfile as sf # not put in project requirements.txt on purpose 
import numpy as np
import torch

ROOT = Path("data/test_pline")
SNR_LIST = [-4, 2, -3, -3, -4, 0, 4, -2, 3, -1, -4, -4] #same as 000 - 011 in test set

def main():
    folders = sorted([d for d in ROOT.iterdir() if d.is_dir()])
    for i, folder in enumerate(folders):
        voice_path = folder / "voice.wav"
        noise_path = folder / "noise.wav"
        
        voice, sr = sf.read(str(voice_path))
        noise, sr2 = sf.read(str(noise_path))
        if sr != sr2:
            raise ValueError(f"Sample rates differ in {folder}: {sr} vs {sr2}")

        # ensure same length
        L = max(len(voice), len(noise))
        if len(voice) < L:
            voice = np.pad(voice, (0, L - len(voice)))
        if len(noise) < L:
            noise = np.pad(noise, (0, L - len(noise)))

        # NumPy arrays to Torch tensors
        voice_t = torch.from_numpy(voice.astype(np.float32))
        noise_t = torch.from_numpy(noise.astype(np.float32))

        # choose one SNR per folder based on index (folder 000 -> SNR_LIST[0], etc.)
        if i < len(SNR_LIST):
            snr = SNR_LIST[i]
        else:
            # wrap around if more folders than provided SNRs
            snr = SNR_LIST[i % len(SNR_LIST)]

        # LOGIC HERE: alpha so that mixed signal has desired SNR (voice power / noise power)
        alpha = 10 ** (-snr / 20) * torch.norm(voice_t) / (torch.norm(noise_t) + 1e-12)
        mixed = voice_t + alpha * noise_t
        # normalize mix to avoid clipping (optional): scale so max abs <= 0.99
        mx = mixed.abs().max().item()
        if mx > 0.999:
            mixed = mixed / (mx / 0.99)

        out_name = folder / f"mix_snr_{snr}.wav"
        sf.write(str(out_name), mixed.numpy().astype(np.float32), sr)
        print(f"Wrote {out_name} (snr={snr})")


if __name__ == "__main__":
    main()
