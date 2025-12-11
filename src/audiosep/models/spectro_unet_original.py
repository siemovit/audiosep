"""6 -level U-Net for spectrogram masking (voice only)."""

from typing import Mapping, Union, cast, Any, override

import librosa
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers.wandb import WandbLogger
from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import wandb

from audiosep.data.spectrogram_orig.spectogram_dataset import (
    OriginalVoiceNoiseDatasetBatch,
)
from audiosep.data.spectrogram_orig.spectrogram import (
    FS,
    HOP_LENGTH,
    N_FFT,
    wav_to_mag_phase,
)
from audiosep.data.waveform_dataset import WaveFormVoiceNoiseBatch


class SpectroUNetOriginal(L.LightningModule):
    """6 -level U-Net for spectrogram masking (voice only)."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        self.metric_sisdr = ScaleInvariantSignalDistortionRatio()
        self.metric_sdr = SignalDistortionRatio()
        self.metric_snr = SignalNoiseRatio()
        self.metric_pesq = PerceptualEvaluationSpeechQuality(fs=FS, mode="nb")
        self.metric_stoi = ShortTimeObjectiveIntelligibility(fs=FS)

        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True),
        )
        self.deconv1 = nn.ConvTranspose2d(
            512, 256, kernel_size=(5, 5), stride=(2, 2), padding=2
        )
        self.deconv1_bad = nn.Sequential(
            nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.ConvTranspose2d(
            512, 128, kernel_size=(5, 5), stride=(2, 2), padding=2
        )
        self.deconv2_bad = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.ConvTranspose2d(
            256, 64, kernel_size=(5, 5), stride=(2, 2), padding=2
        )
        self.deconv3_bad = nn.Sequential(
            nn.BatchNorm2d(64), nn.ReLU(True), nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.ConvTranspose2d(
            128, 32, kernel_size=(5, 5), stride=(2, 2), padding=2
        )
        self.deconv4_bad = nn.Sequential(
            nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.ConvTranspose2d(
            64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2
        )
        self.deconv5_bad = nn.Sequential(
            nn.BatchNorm2d(16), nn.ReLU(True), nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(
            32, 1, kernel_size=(5, 5), stride=(2, 2), padding=2
        )

        # Define the criterion and optimizer
        self.crit = nn.L1Loss()

    @override
    def forward(self, mix):
        """
        Generate the mask for the given mixture audio spectrogram

        Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
        Ret:    The soft mask which size is (B, 1, 512, 128)
        """
        conv1_out = self.conv1(mix)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size=conv5_out.size())
        deconv1_out = self.deconv1_bad(deconv1_out)
        deconv2_out = self.deconv2(
            torch.cat([deconv1_out, conv5_out], 1), output_size=conv4_out.size()
        )
        deconv2_out = self.deconv2_bad(deconv2_out)
        deconv3_out = self.deconv3(
            torch.cat([deconv2_out, conv4_out], 1), output_size=conv3_out.size()
        )
        deconv3_out = self.deconv3_bad(deconv3_out)
        deconv4_out = self.deconv4(
            torch.cat([deconv3_out, conv3_out], 1), output_size=conv2_out.size()
        )
        deconv4_out = self.deconv4_bad(deconv4_out)
        deconv5_out = self.deconv5(
            torch.cat([deconv4_out, conv2_out], 1), output_size=conv1_out.size()
        )
        deconv5_out = self.deconv5_bad(deconv5_out)
        deconv6_out = self.deconv6(
            torch.cat([deconv5_out, conv1_out], 1), output_size=mix.size()
        )
        out = torch.sigmoid(deconv6_out)
        return out

    def _shared_step(self, batch: OriginalVoiceNoiseDatasetBatch, stage: str):
        mix, voc = batch
        msk = self(mix)
        loss = self.crit(msk * mix, voc)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def test_step(self, batch: WaveFormVoiceNoiseBatch, batch_idx: int):
        if batch.mix.size(0) != 1:
            raise ValueError("Batch size must be 1 for test step.")
        mix_waveform = batch.mix.squeeze()
        mix_spec_mag, phase = wav_to_mag_phase(mix_waveform.cpu().numpy())
        voice_waveform = batch.voice.squeeze()

        spec_sum: Union[np.ndarray, None] = None
        window = 128
        T = mix_spec_mag.shape[-1]
        if T == 0:
            return {}
            # 1. Normalize input (Critical: model expects 0-1 range)
        norm_factor = mix_spec_mag.max()
        if norm_factor > 0:
            mix_spec_mag = mix_spec_mag / norm_factor

        # Pad time dimension so we always have full 128-frame segments
        pad_width = 0
        if T < window:
            pad_width = window - T
        elif T % window != 0:
            pad_width = window - (T % window)

        if pad_width > 0:
            # mix_spec shape: (freq, time), pad only the time dim
            mix_spec_mag = F.pad(mix_spec_mag, (0, pad_width))
        for i in range(mix_spec_mag.shape[-1] // 128):
            # Get the fixed size of segment
            seg = mix_spec_mag[1:, i * 128 : i * 128 + 128, np.newaxis]
            seg = np.asarray(seg, dtype=np.float32)
            seg = torch.from_numpy(seg).permute(2, 0, 1)
            seg = torch.unsqueeze(seg, 0)
            seg = seg.to(self.device)

            # generate mask
            msk = self.forward(seg)

            vocal_ = seg * msk

            # accumulate the segment until the whole song is finished
            vocal_ = vocal_.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, 0]
            vocal_ = np.vstack((np.zeros((128)), vocal_))
            spec_sum = (
                vocal_ if spec_sum is None else np.concatenate((spec_sum, vocal_), -1)
            )
        if spec_sum is None:
            return {}
        length = min(phase.shape[-1], spec_sum.shape[-1])
        mag = spec_sum[:, :length]
        phase = phase[:, :length]
        spectrogram = mag * phase
        spectrogram *= norm_factor.item()
        predicted_voice = librosa.istft(
            spectrogram, win_length=N_FFT, hop_length=HOP_LENGTH
        )
        pred_voice = torch.from_numpy(predicted_voice).float().to(self.device)

        # Match lengths
        min_len = min(pred_voice.shape[-1], voice_waveform.shape[-1])
        pred_voice = pred_voice[..., :min_len]
        target = voice_waveform[..., :min_len]
        control_mix = mix_waveform[..., :min_len]
        # ----- Metrics -----
        sisdr = self.metric_sisdr(pred_voice, target)
        sdr = self.metric_sdr(pred_voice, target)
        snr = self.metric_snr(pred_voice, target)
        pesq = self.metric_pesq(pred_voice, target)
        stoi = self.metric_stoi(pred_voice, target)
        snr_control = self.metric_snr(control_mix, target)
        # Log metrics
        self.log(
            "test/si_sdr",
            sisdr,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            "test/sdr",
            sdr,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log("test/snr", snr, on_step=True, on_epoch=True, batch_size=1)
        self.log("test/pesq", pesq, on_step=True, on_epoch=True, batch_size=1)
        self.log("test/stoi", stoi, on_step=True, on_epoch=True, batch_size=1)
        self.log("test/control_snr", snr_control, on_step=True, batch_size=1)

        return {
            "pred": pred_voice,
            "mix": mix_waveform,
            "voice": target,
            "noise": batch.noise.squeeze(),
            "idx": batch_idx,
            "sisdr": sisdr,
            "snr": snr,
            "control_snr": snr_control,
            "mix_filename": batch.mix_filename,
        }

    def on_test_start(self):
        self.table_data: list[list[Any]] = []

    @override
    def on_test_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if not isinstance(outputs, dict):
            return

        pred_np = outputs["pred"].cpu().numpy().astype("float32")
        voice_np = outputs["voice"].cpu().numpy().astype("float32")
        noise_np = outputs["noise"].cpu().numpy().astype("float32")
        mix_np = outputs["mix"].cpu().numpy().astype("float32")
        self.table_data.append(
            [
                batch_idx,
                wandb.Audio(pred_np, sample_rate=FS),
                wandb.Audio(voice_np, sample_rate=FS),
                wandb.Audio(noise_np, sample_rate=FS),
                wandb.Audio(mix_np, sample_rate=FS),
                outputs["sisdr"].item(),
                outputs["snr"].item(),
                outputs["control_snr"].item(),
                outputs["mix_filename"][0],
            ]
        )

    def on_test_end(self):
        logger = cast(WandbLogger, self.logger)
        logger.log_table(
            "test_table",
            columns=[
                "idx",
                "predicted_voice",
                "target_voice",
                "noise",
                "mix",
                "sisdr",
                "snr",
                "control_snr",
                "mix_filename",
            ],
            data=self.table_data,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
