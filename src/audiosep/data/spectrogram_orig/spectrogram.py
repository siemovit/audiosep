import librosa
import numpy as np
import torch

SR = 8000
N_FFT = 1024
HOP_LENGTH = 768


def wav_to_mag_phase(path, n_fft=N_FFT, hop_length=HOP_LENGTH):
    audio, _ = librosa.load(path, sr=None, mono=True)
    signal = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spectrum, phase = librosa.magphase(signal)
    spectrogram = np.abs(spectrum).astype(np.float32)
    return torch.from_numpy(spectrogram).float(), phase
