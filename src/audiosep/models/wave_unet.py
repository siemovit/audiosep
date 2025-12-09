import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import List

from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import os

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=15, padding=None):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class WaveUNet(L.LightningModule):
    """Compact Wave-U-Net implementation for source separation.

    This follows the architecture idea from "Wave-U-Net: A Multi-Scale Neural
    Network for End-to-End Audio Source Separation" (Stoller et al.).
    """

    def __init__(self, in_channels=1, out_channels=2, depth=6, base_filters=24, max_filters=128, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.depth = int(depth)
        self.base_filters = int(base_filters)
        self.max_filters = int(max_filters)

        # metrics
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        # torchmetrics PESQ
        self.pesq = PerceptualEvaluationSpeechQuality(fs=8000, mode='nb')

        # build encoder channel sizes with cap
        enc_chs = [self.base_filters]
        for i in range(1, self.depth):
            next_ch = min(enc_chs[-1] * 2, self.max_filters)
            enc_chs.append(next_ch)

        # store modules
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i, ch in enumerate(enc_chs):
            in_ch = in_channels if i == 0 else enc_chs[i - 1]
            self.encs.append(ConvBlock(in_ch, ch))
            if i < len(enc_chs) - 1:
                # Paper uses decimation (discard every other sample)
                # We can implement this as a simple lambda or module in forward
                # But to keep ModuleList structure, we can use a placeholder or just handle in forward
                pass

        # bottleneck channels
        bottleneck_ch = enc_chs[-1]
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch)

        # decoder: mirror of encoder (use skip connections)
        self.decs = nn.ModuleList()
        dec_ch = bottleneck_ch
        # iterate reversed encoder channels excluding last (bottleneck)
        for skip_ch in reversed(enc_chs[:-1]):
            in_ch = dec_ch + skip_ch
            out_ch = skip_ch
            self.decs.append(ConvBlock(in_ch, out_ch))
            dec_ch = out_ch

        # final conv maps back to out_channels
        self.final = nn.Conv1d(dec_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        skips = []
        out = x
        # encoder
        for i, enc in enumerate(self.encs):
            out = enc(out)
            skips.append(out)
            if i < len(self.encs) - 1:
                # Downsample by decimation (keep even indices)
                out = out[:, :, ::2]

        out = self.bottleneck(out)

        # decoder (upsample by factor 2, concat skip, conv)
        for i, dec in enumerate(self.decs):
            # Upsample via linear interpolation
            out = F.interpolate(out, scale_factor=2, mode='linear', align_corners=False)
            
            # match skip shape
            skip = skips[-(i + 2)]  # because last skip corresponds to last encoder output
            
            # Handle potential size mismatch due to odd input lengths in downsampling
            if out.shape[-1] != skip.shape[-1]:
                min_t = min(out.shape[-1], skip.shape[-1])
                out = out[..., :min_t]
                skip = skip[..., :min_t]
                
            out = torch.cat([out, skip], dim=1)
            out = dec(out)

        # final upsample if needed (to match original input length)
        if out.shape[-1] != x.shape[-1]:
            out = F.interpolate(out, size=x.shape[-1], mode='linear', align_corners=False)

        logits = self.final(out)
        # if output is multi-channel masks, use softmax across channel dim
        if logits.shape[1] > 1:
            masks = F.softmax(logits, dim=1)
            estimates = masks * x
        else:
            estimates = logits
            masks = None

        # split estimates
        est_voice = estimates[:, 0:1, :]
        est_noise = estimates[:, 1:2, :]

        return est_voice, est_noise, masks
    
    def _shared_step(self, batch, stage: str):
        x, y = batch
        # x: (B,1,T) ; y['voice'] (B,1,T)
        est_voice, est_noise, _ = self.forward(x)
        # reshape to (B, T)
        est = est_voice.squeeze(1)
        tgt = y['voice'].squeeze(1)
        si = self.si_snr(est, tgt)
        
        loss = -si.mean()
        
        try:
            pesq = self.pesq(est.unsqueeze(1), tgt.unsqueeze(1))
            self.log(f'{stage}_pesq', pesq.mean(), prog_bar=True)
        except Exception:
            # PESQ can fail if the reference signal is silent (NoUtterancesError)
            # This happens when a random crop falls on a silence period.
            pass

        # self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_loss', loss.mean(), prog_bar=True)

        return loss 
    
    def training_step(self, batch, batch_idx):
        loss =  self._shared_step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
