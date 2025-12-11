import numpy as np
import torch
import librosa



def load_audio_tensor(filepath, target_sample_rate=None):
    """Load audio using librosa (or soundfile as fallback) and return (tensor, sr).

    Returns a torch.FloatTensor of shape (channels, samples).
    """

    # prefer librosa for resampling convenience
    if librosa is not None:
        y, sr = librosa.load(filepath, sr=target_sample_rate, mono=False)
        # librosa returns (n,) or (n_channels, n)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return torch.from_numpy(y), sr