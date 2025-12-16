"""Tunable -level U-Net for spectrogram masking (voice only)."""

from typing import Union, cast, Any, override

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
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio

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


class SpectroUNetOriginalTunable(L.LightningModule):
    """Tunable -level U-Net for spectrogram masking (voice only)."""

    def __init__(
        self,
        dropout: float = 0.5,
        depth: int = 6,
        start_channels: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.metric_sisdr = ScaleInvariantSignalDistortionRatio()
        self.metric_sdr = SignalDistortionRatio()
        self.metric_snr = SignalNoiseRatio()
        self.metric_pesq = PerceptualEvaluationSpeechQuality(fs=FS, mode="nb")
        self.metric_stoi = ShortTimeObjectiveIntelligibility(fs=FS)
        self.metric_sisnr = ScaleInvariantSignalNoiseRatio()

        self.depth = depth

        self.depth = depth

        # --- ENCODER ---
        self.encoder_layers = nn.ModuleList()
        in_c = 1
        out_c = start_channels

        # Store expected channel sizes to help build the decoder mirror
        encoder_channels = []

        for _ in range(depth):
            block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(5, 5), stride=(2, 2), padding=2),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(True),
            )
            self.encoder_layers.append(block)
            encoder_channels.append(out_c)

            in_c = out_c
            out_c *= 2

        # --- DECODER ---
        self.decoder_trans = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()

        # Reverse channels for decoder construction: e.g., [512, 256, 128, 64, 32, 16]
        rev_channels = list(reversed(encoder_channels))

        for i in range(depth):
            # Determine Input Channels
            if i == 0:
                # Top of the U-Net (bottleneck): Input is just the last encoder output
                decoder_in_c = rev_channels[i]
            else:
                # Subsequent layers: Input is (Previous Decoder Out) + (Skip Connection)
                # In standard U-Net, these usually have the same channel count, so we multiply by 2.
                # Logic: We are concatenating the result of the previous layer (which has `rev_channels[i]` channels)
                # with the skip connection from the encoder (which also has `rev_channels[i]` channels).
                decoder_in_c = rev_channels[i] * 2

            # Determine Output Channels
            if i == depth - 1:
                # Final layer outputs 1 channel (the mask)
                decoder_out_c = 1
            else:
                # Outputs the channel count of the next layer down
                decoder_out_c = rev_channels[i + 1]

            # 1. Transpose Convolution
            self.decoder_trans.append(
                nn.ConvTranspose2d(
                    decoder_in_c,
                    decoder_out_c,
                    kernel_size=(5, 5),
                    stride=(2, 2),
                    padding=2,
                )
            )

            # 2. BN/ReLU/Dropout Block (The "bad" block in original code)
            # The last layer (i == depth - 1) does NOT have BN/ReLU/Dropout
            if i < depth - 1:
                self.decoder_bn.append(
                    nn.Sequential(
                        nn.BatchNorm2d(decoder_out_c),
                        nn.ReLU(True),
                        nn.Dropout2d(dropout),
                    )
                )
            else:
                self.decoder_bn.append(nn.Identity())

        self.crit = nn.L1Loss()

    @override
    def forward(self, mix):
        """
        Generate the mask for the given mixture audio spectrogram

        Arg:    mix     (torch.Tensor)  - The mixture spectrogram which size is (B, 1, 512, 128)
        Ret:    The soft mask which size is (B, 1, 512, 128)
        """
        x = mix

        # We need to store outputs for skip connections
        # encoder_outputs[0] -> result of layer 1 (channels=16)
        # encoder_outputs[-1] -> result of bottleneck (channels=512)
        encoder_outputs = []
        x = mix
        for layer in self.encoder_layers:
            x = layer(x)
            encoder_outputs.append(x)

        # --- DECODER PASS ---
        # x currently holds the bottleneck output (equivalent to conv6_out)

        for i in range(self.depth):
            # 1. Prepare Input
            if i == 0:
                # First decoder step just takes the bottleneck
                inp = x
                # The target size for this unpooling is the size of the layer *before* the bottleneck
                # i.e., encoder_outputs[-2]
                target_output_size = encoder_outputs[-(i + 2)].size()
            else:
                # Subsequent steps take (Previous Output concatenated with Skip Connection)
                # Skip connection index:
                # If i=1 (second decoder layer), we need skip from encoder_outputs[-2] (conv5)
                # Wait, let's trace:
                # i=0 uses encoder_outputs[-1] (conv6) as input.
                # i=1 uses cat(decoder_prev, encoder_outputs[-2]).

                skip_connection = encoder_outputs[-(i + 1)]
                inp = torch.cat([x, skip_connection], 1)

                # Determine output_size for Transpose Conv
                if i == self.depth - 1:
                    # Last layer targets the original mix size
                    target_output_size = mix.size()
                else:
                    # Others target the specific encoder layer size below them
                    target_output_size = encoder_outputs[-(i + 2)].size()

            # 2. Transpose Conv
            # Note: We must index the ModuleList manually
            x = self.decoder_trans[i](inp, output_size=target_output_size)

            # 3. Activation/Dropout (BN -> ReLU -> Drop)
            # This is an Identity() for the last layer
            x = self.decoder_bn[i](x)

        # Final Sigmoid activation
        out = torch.sigmoid(x)
        return out

    def _shared_step(self, batch: OriginalVoiceNoiseDatasetBatch, stage: str):
        mix, voc = batch
        msk = self(mix)
        loss = self.crit(msk * mix, voc)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    @override
    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    @override
    def validation_step(self, batch, batch_idx: int):
        return self.shared_test_step(batch, batch_idx, stage="val")

    @override
    def test_step(self, batch: WaveFormVoiceNoiseBatch, batch_idx: int):
        return self.shared_test_step(batch, batch_idx, stage="test")

    def shared_test_step(
        self, batch: WaveFormVoiceNoiseBatch, batch_idx: int, stage: str = "test"
    ):
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
        sisnr = self.metric_sisnr(pred_voice, target)
        si_snr_control = self.metric_sisnr(control_mix, target)

        # Log metrics
        self.log(
            f"{stage}/si_sdr",
            sisdr,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{stage}/sdr",
            sdr,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(f"{stage}/snr", snr, on_step=True, on_epoch=True, batch_size=1)
        self.log(f"{stage}/pesq", pesq, on_step=True, on_epoch=True, batch_size=1)
        self.log(f"{stage}/stoi", stoi, on_step=True, on_epoch=True, batch_size=1)
        self.log(f"{stage}/control_snr", snr_control, on_step=True, batch_size=1)
        self.log(f"{stage}/si_snr", sisnr, on_step=True, batch_size=1)
        self.log(f"{stage}/control_si_snr", si_snr_control, on_step=True, batch_size=1)
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
            "mix_original_snr": int(batch.mix_filename[0].split(".")[0].split("_")[-1]),
        }

    @override
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
        mix_original_snr = outputs.get("mix_original_snr")
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
                mix_original_snr,
            ]
        )

    @override
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
                "mix_original_snr",
            ],
            data=self.table_data,
        )

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
