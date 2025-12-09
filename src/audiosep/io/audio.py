import torchaudio

def load_audio_tensor(filepath, target_sample_rate=None, trim_length=None, to_numpy=False):
    signal, sample_rate = torchaudio.load(filepath)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        signal = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )(signal)
        sample_rate = target_sample_rate
    if trim_length is not None:
        signal = signal[:, :trim_length]
    if to_numpy:
        signal = signal.numpy()
    return signal, sample_rate

def save_audio_tensor(filepath, signal, sample_rate):
    torchaudio.save(
        filepath, 
        signal.detach().cpu(), 
        sample_rate,
        )
    