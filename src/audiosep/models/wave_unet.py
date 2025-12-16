import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Any, List, cast

from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from torchmetrics.audio import (
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
    SignalNoiseRatio,
)
from lightning.pytorch.loggers.wandb import WandbLogger
import wandb
from audiosep.data.waveform_dataset import WaveFormVoiceNoiseBatch

import numpy as np
import os
import soundfile as sf
from lightning.pytorch.utilities.rank_zero import rank_zero_warn



FS = 8000  # sampling frequency for metrics

from audiosep.utils import center_crop

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 15, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class Encoder(nn.Module):
    """Single-level encoder: conv block followed by decimation (downsample by 2)."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 15, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        # downsample by factor 2 (decimation)
        x_down = x[:, :, ::2]
        return x, x_down

class Decoder(nn.Module):
    """Single-level decoder: upsample, concatenate skip, apply conv block."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, kernel_size=kernel_size, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # upsample by factor 2
        x_up = F.interpolate(x, scale_factor=2, mode="linear", align_corners=True)
        x_cat = torch.cat([x_up, skip], dim=1)
        return self.conv(x_cat)
class WaveUNet(L.LightningModule):
    """Wave-U-Net with 2 channels output (voice + noise)

    Parameters:
        - in_channels: number of input channels
        - out_channels: number of output channels
        - depth: number of down/up sampling layers
        - base_filters: channel step per layer
    """

    def __init__(self, 
                 depth: int = 5, 
                 base_filters: int = 24, 
                 lr: float = 1e-3, 
                 lambda_mix: float = 0.1, 
                 lambda_noise: float = 0.3, 
                 mse_loss: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.depth = int(depth)
        self.base_filters = int(base_filters)
        self.in_channels = 1 # mono input
        self.out_channels = 2 # voice + noise
        self.lambda_mix = lambda_mix  # weight for mix loss
        self.lr = lr
        self.lambda_noise = lambda_noise  # weight for noise loss, voice quality is prioritized
        self.mse_loss = mse_loss  # use MSE loss instead of SI-SNR

        # metrics
        self.metric_sisnr = ScaleInvariantSignalNoiseRatio()
        self.metric_sisdr = ScaleInvariantSignalDistortionRatio()
        self.metric_sdr = SignalDistortionRatio()
        self.metric_snr = SignalNoiseRatio()
        self.metric_pesq = PerceptualEvaluationSpeechQuality(fs=FS, mode="nb")
        self.metric_stoi = ShortTimeObjectiveIntelligibility(fs=FS, extended=False)

        # ----------- Encoder -----------
        # encoder/decoder config
        encoder_in_channels_list = [self.in_channels] + [i * self.base_filters for i in range(1, self.depth)]
        encoder_out_channels_list = [i * self.base_filters for i in range(1, self.depth + 1)]

        # encoder blocks
        self.encoder = nn.ModuleList()
        for i in range(self.depth):
            self.encoder.append(
                Encoder(encoder_in_channels_list[i], encoder_out_channels_list[i], kernel_size=15)
            )

        # bottleneck
        self.bottleneck = ConvBlock(self.depth * self.base_filters, self.depth * self.base_filters, kernel_size=15)
        
        # ----------- Decoder -----------
        # decoder in/out channels
        decoder_in_channels_list = [(2 * i + 1) * self.base_filters for i in range(1, self.depth)] + [2 * self.depth * self.base_filters]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        # decoder blocks
        self.decoder = nn.ModuleList()
        for i in range(self.depth):
            self.decoder.append(
                Decoder(decoder_in_channels_list[i], decoder_out_channels_list[i], kernel_size=5)
            )

        # final conv: concat with input (base_filters + in_channels) -> out_channels
        self.out = nn.Conv1d(self.base_filters + self.in_channels, self.out_channels, kernel_size=1, stride=1)
        
    def __repr__(self):
        return f"WaveUNet_d{self.depth}_f{self.base_filters}"
    
    def forward(self, x: torch.Tensor):
        skips = []
        o = x

        # 1) Encoder
        for i in range(self.depth):
            skip, o = self.encoder[i](o)
            skips.append(skip)

        # 2) Bottleneck
        o = self.bottleneck(o)

        # 3) Decoder
        for i in range(self.depth):
            o = self.decoder[i](o, skips[self.depth - i - 1])

        # 4) Final conv + concat with input
        if o.shape[-1] != x.shape[-1]:
            print("Warning: output length != input length:", o.shape[-1], x.shape[-1])
        o = torch.cat([o, x], dim=1)
        o = self.out(o)

        # split outputs (assume out_channels >= 2)
        est_voice = o[:, 0:1, :] # shape (B, 1, T)
        est_noise = o[:, 1:2, :] # shape (B, 1, T)
        
        return est_voice, est_noise

    def _shared_step(self, batch, stage: str):
        # Get data from batch
        x, y = batch
        v_ref, n_ref  = y["voice"], y["noise"]
        
        # Model forward (B, 1, T_in)
        v_hat, n_hat = self(x)
        
        # Crop to match target length
        out_len = v_ref.shape[-1]
        v_hat_c = center_crop(v_hat, out_len)
        n_hat_c = center_crop(n_hat, out_len)
        mix_c = center_crop(x, out_len)
        
        # ---- Losses ----
        # ---- Baseline (MSE) ---
        if self.mse_loss:
            loss_v = F.mse_loss(v_hat_c, v_ref)
            loss_n = F.mse_loss(n_hat_c, n_ref)
            loss = loss_v + loss_n
        
        else:
            # ---- Voice + Noise SI-SNR with silence gating ----
            eps = 1e-8
            v_energy = (v_ref.squeeze(1) ** 2).mean(dim=-1)      # (B,)
            v_mask   = (v_energy > 1e-4).float()
            si_v = self.metric_sisnr(v_hat_c.squeeze(1), v_ref.squeeze(1))  # (B,)
            loss_v = - (si_v * v_mask).sum() / (v_mask.sum() + eps)
            # Noise SI-SNR (usually no need to gate since noise is present) ----
            si_n = self.metric_sisnr(n_hat_c.squeeze(1), n_ref.squeeze(1))  # (B,)
            loss_n = - si_n.mean()
            # Total SI-SNR loss (with weighting)
            loss_si = loss_v + self.lambda_noise * loss_n
            # Mixture consistency ----
            loss_mix = F.mse_loss(v_hat_c + n_hat_c, mix_c)
            
            # Total loss
            loss = loss_si + self.lambda_mix * loss_mix     
            
            # log individual losses
            self.log(f"{stage}_loss_si", loss_si, on_step=True)
            self.log(f"{stage}_loss_mix", loss_mix, on_step=True) 
        
        # ---- Voice SI-SNR with silence gating + PIT ----
        # TODO: (re)-implement PIT version
        
        
        # global loss logging
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
                
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    @torch.no_grad()
    def infer_full_simple(self, mix, out_len, context):
        """
        mix: (1, T)
        returns: voice_hat (1, T)
        """
                
        if mix.dim() == 2:
            mix = mix.unsqueeze(0)  # (1,1,T)

        _, _, T = mix.shape
        # in_len = out_len + 2 * context

        voice_chunks = []
        noise_chunks = []

        t = 0
        while t < T:
            # input window
            left = t - context
            right = t + out_len + context

            pad_l = max(0, -left)
            pad_r = max(0, right - T)

            left = max(0, left)
            right = min(T, right)

            x = mix[..., left:right]
            if pad_l or pad_r:
                x = F.pad(x, (pad_l, pad_r))
             
            # Inference
            # print("x shape before inference", x.shape)
            v_hat, n_hat = self(x)  # (1,1,in_len-ish)

            # center crop
            v_c = center_crop(v_hat, out_len)
            n_c = center_crop(n_hat, out_len)
            # print("v_c shape after center crop", v_c.shape)
            # start = (v_hat.shape[-1] - out_len) // 2
            # v_c = v_hat[..., start:start + out_len]

            # trim last chunk
            v_c = v_c[..., : min(out_len, T - t)]
            n_c = n_c[..., : min(out_len, T - t)]

            voice_chunks.append(v_c.squeeze(0))
            noise_chunks.append(n_c.squeeze(0))
            t += out_len
            
        # Concatenate chunks
        return torch.cat(voice_chunks, dim=-1)[:, :T], torch.cat(noise_chunks, dim=-1)[:, :T]
    
    def test_step(self, batch:WaveFormVoiceNoiseBatch , batch_idx):
        mix, v_ref, n_ref, mix_filename = batch  # full signals

        # same as dataset windowing
        context = 4096
        out_len = 16384
                
        v_hat, _ = self.infer_full_simple(
            mix,
            out_len=out_len,
            context=context,
        )

        # match lengths (optional)
        L = min(v_hat.shape[-1], v_ref.shape[-1])
        v_hat = v_hat[..., :L]
        # n_hat = n_hat[..., :L]
        v_ref = v_ref[..., :L]
        # n_ref = n_ref[..., :L]
        mix   = mix[..., :L]

        # metrics (voice only, as in paper)
        sisnr = self.metric_sisnr(v_hat, v_ref)
        si_snr_control = self.metric_sisnr(mix, v_ref)
        sisdr = self.metric_sisdr(v_hat, v_ref)
        # The SDR computation is problematic when the true source is silent or near-silent. 
        # In case of silence, the SDR is undefined (log(0)), which happens often for vocal tracks
        sdr = self.metric_sdr(v_hat, v_ref)
        snr   = self.metric_snr(v_hat, v_ref)
        pesq  = self.metric_pesq(v_hat, v_ref)
        stoi  = self.metric_stoi(v_hat, v_ref)
        snr_control = self.metric_snr(mix, v_ref)

        # logging
        self.log("test/si_snr", sisnr, on_step=True, batch_size=1)  
        self.log("test/si_snr_control", si_snr_control, on_step=True, batch_size=1)
        self.log("test/si_sdr", sisdr, on_step=True, batch_size=1)
        self.log("test/sdr", sdr, on_step=True, batch_size=1)
        self.log("test/snr", snr, on_step=True, batch_size=1)
        self.log("test/pesq", pesq, on_step=True, batch_size=1)
        self.log("test/stoi", stoi, on_step=True, batch_size=1)
        self.log("control_snr", snr_control, on_step=True, batch_size=1)
        
        return {
            "pred": v_hat,
            "mix": mix,
            "voice": v_ref,
            "noise": n_ref,
            "idx": batch_idx,
            "sisdr": sisdr, 
            "snr": snr,
            "control_snr": snr_control, 
            "si_snr": sisnr,
            "control_si_snr": si_snr_control,
            "mix_filename": mix_filename,
        }
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_test_start(self):
        self.table_data: list[list[Any]] = []
        
    def to_wandb_audio(self, x: torch.Tensor, sr: int, batch_idx: int = 0, mix_filename=None, debug=False) -> wandb.Audio:
        # Return a sanitized 1D numpy array from a tensor
        # x can be (1,T) or (1,1,T) or (T,)
        x = x.detach().cpu()
        if x.dim() == 3:   # (B,1,T)
            x = x[0, 0]
        elif x.dim() == 2: # (1,T)
            x = x[0]
        # now (T,)
        x = x.to(torch.float32)
        # safety: replace NaNs/Infs
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # convert to numpy
        x = x.numpy().astype(np.float32)
        
        if debug:
            debug_dir = os.path.abspath("./test_debug_audio")
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"{batch_idx}_{mix_filename}")
            capt = mix_filename.split(".")[0]
            sf.write(out_path, x, sr)
        
        capt = mix_filename.split(".")[0]
        return wandb.Audio(x, sample_rate=sr, caption=f"Test #{batch_idx} - {capt}")

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        mix_fname = outputs.get("mix_filename")[0]
        
        # log a few audio examples to wandb for demo
        if batch_idx == 0:
            run = self.logger.experiment  # wandb.Run
            run.log({"noise": self.to_wandb_audio(outputs["noise"], FS, batch_idx, mix_filename=mix_fname)})
            run.log({"mix": self.to_wandb_audio(outputs["mix"], FS, batch_idx, mix_filename=mix_fname)})
            run.log({"pred": self.to_wandb_audio(outputs["pred"], FS, batch_idx, mix_filename=mix_fname, debug=True)})
            run.log({"voice": self.to_wandb_audio(outputs["voice"], FS, batch_idx, mix_filename=mix_fname)})
          
        if batch_idx == 6:
            run = self.logger.experiment  # wandb.Run
            run.log({"noise": self.to_wandb_audio(outputs["noise"], FS, batch_idx, mix_filename=mix_fname)})
            run.log({"mix": self.to_wandb_audio(outputs["mix"], FS, batch_idx, mix_filename=mix_fname)})
            run.log({"pred": self.to_wandb_audio(outputs["pred"], FS, batch_idx, mix_filename=mix_fname, debug=True)})
            run.log({"voice": self.to_wandb_audio(outputs["voice"], FS, batch_idx, mix_filename=mix_fname)})
        
        pred_arr = self.to_wandb_audio(outputs["pred"], FS, batch_idx, mix_filename=mix_fname, debug=True)
        voice_arr = self.to_wandb_audio(outputs["voice"], FS, batch_idx, mix_filename=mix_fname)
        noise_arr = self.to_wandb_audio(outputs["noise"], FS, batch_idx, mix_filename=mix_fname)
        mix_arr = self.to_wandb_audio(outputs["mix"], FS, batch_idx, mix_filename=mix_fname)
        sisdr = outputs.get("sisdr").item()
        snr = outputs.get("snr").item()
        control_snr = outputs.get("control_snr").item()    
        si_snr = outputs.get("si_snr").item()
        control_si_snr = outputs.get("control_si_snr").item()    

        self.table_data.append(
            [
                batch_idx,
                pred_arr,
                voice_arr,
                noise_arr,
                mix_arr,
                sisdr,
                snr,
                control_snr,
                si_snr,
                control_si_snr,
                mix_fname,
            ]
        )
        
    def on_test_end(self):
        logger = cast(WandbLogger, self.logger)
        logger.log_table(
            "test_waveunet_table",
            columns=[
                "idx",
                "predicted_voice",
                "target_voice",
                "noise",
                "mix",
                "sisdr",
                "snr",
                "control_snr",
                "si_snr",
                "control_si_snr",
                "mix_filename",
            ],
            data=self.table_data,
        )
if __name__ == "__main__":
    model = WaveUNet()
    x = torch.randn(2, 1, 8000)
    print(model.__repr__())
    est_voice, est_noise = model(x)
    print(est_voice.shape, est_noise.shape)