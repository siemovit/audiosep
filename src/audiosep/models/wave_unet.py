import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import List

from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torch

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
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        # upsample by factor 2
        x_up = self.upsample(x)
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

    def __init__(self, in_channels: int = 1, out_channels: int = 2, depth: int = 6, base_filters: int = 24, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.depth = int(depth)
        self.base_filters = int(base_filters)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.lambda_mix = 0.1  # weight for mix loss
        self.lr = lr

        # metrics
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=8000, mode="nb")
        self.stoi = ShortTimeObjectiveIntelligibility(fs=8000, extended=False)

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
        self.out = nn.Sequential(
            nn.Conv1d(self.base_filters + self.in_channels, self.out_channels, kernel_size=1, stride=1),
            nn.Tanh(),
        )
        
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
        o = torch.cat([o, x], dim=1)
        o = self.out(o)

        # split outputs (assume out_channels >= 2)
        est_voice = o[:, 0:1, :]
        est_noise = o[:, 1:2, :]
        return est_voice, est_noise

    def _shared_step(self, batch, stage: str):
        x, y = batch
        
        # Preds
        v_hat, n_hat = self.forward(x)
        
        # Targets
        v_ref = y["voice"]
        n_ref = y["noise"]
        
        # Losses
        si_voice = self.si_snr(v_hat, v_ref).mean()
        # si_noise = self.si_snr(n_hat, n_ref).mean()
        
        loss_si = -si_voice  # -si_snr to minimize
        
        mix_rec = v_hat + n_hat
        loss_mix = F.mse_loss(mix_rec, x)
        
        loss = loss_si + self.lambda_mix * loss_mix

        try:
            pesq = self.pesq(v_hat.unsqueeze(1), v_ref.unsqueeze(1))
            stoi = self.stoi(v_hat.unsqueeze(1), v_ref.unsqueeze(1))
            self.log(f"{stage}_pesq", pesq.mean(), on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_stoi", stoi.mean(), on_step=False, on_epoch=True, prog_bar=True)
            
        except Exception:
            # PESQ can fail on silent segments
            pass

        self.log(f"{stage}_loss", loss, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    model = WaveUNet()
    x = torch.randn(2, 1, 8000)
    print(model.__repr__())
    est_voice, est_noise = model(x)
    print(est_voice.shape, est_noise.shape)