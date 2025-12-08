import librosa
import numpy as np
import torch

SR = 8000
N_FFT = 1024
HOP_LENGTH = 768

def wav_to_mag_phase(path, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
    y, _ = librosa.load(path, sr=sr, mono=True)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    mag = np.abs(S)          # pas de normalisation
    phase = np.angle(S)
    mag_t = torch.from_numpy(mag).float()  # (F, T)
    return mag_t, phase