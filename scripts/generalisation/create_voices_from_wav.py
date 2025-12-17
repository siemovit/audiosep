"""Create dataset chunks from a long wav file.

Loads `audio_samples/histoire_naturelle_pline.wav`, resamples to 8 kHz,
splits into 10 second non-overlapping chunks and writes them to
`data/test_pline/000/voice.wav`, `data/test_pline/001/voice.wav`, ...

"""
import os
from pathlib import Path
import soundfile as sf
import librosa
import numpy as np


SRC = Path("audio_samples/histoire_naturelle_pline.wav")
OUT_DIR = Path("data/test_pline")
SR = 8000
CHUNK_SECONDS = 10


def main():
    y, orig_sr = librosa.load(str(SRC), sr=SR, mono=True)

    total_samples = y.shape[0]
    chunk_size = CHUNK_SECONDS * SR
    n_chunks = int(np.ceil(total_samples / chunk_size))

    print(f"Total duration: {total_samples/orig_sr:.1f}s ({total_samples} samples) -> {n_chunks} chunks of {CHUNK_SECONDS}s")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_samples)
        chunk = y[start:end]
        # if last chunk shorter than chunk_size, pad with zeros
        if chunk.shape[0] < chunk_size:
            pad = chunk_size - chunk.shape[0]
            chunk = np.pad(chunk, (0, pad), mode="constant")

        subdir = OUT_DIR / f"{i:03d}"
        subdir.mkdir(parents=True, exist_ok=True)
        out_path = subdir / "voice.wav"
        sf.write(str(out_path), chunk.astype(np.float32), SR)
        print(f"Wrote {out_path} ({len(chunk)} samples)")


if __name__ == "__main__":
    main()
