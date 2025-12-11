import librosa
import numpy as np
import torch

N_FFT = 1024
HOP_LENGTH = 768
FS = 8000


def wav_to_mag_phase(waveform: np.ndarray, n_fft=N_FFT, hop_length=HOP_LENGTH):
    signal = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    spectrum, phase = librosa.magphase(signal)
    spectrogram = np.abs(spectrum).astype(np.float32)
    return torch.from_numpy(spectrogram).float(), phase
