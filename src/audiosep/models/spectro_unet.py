import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class SpectroUNet2D(L.LightningModule):
    """Simpler U-Net: 4 encoder layers

    - retains _match_spatial and loss utilities from previous implementation
    """

    def __init__(self, in_channels=1, out_channels=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Encoder (downsampling by 2 each block)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        
        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.dec4 = nn.ConvTranspose2d(
            16, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x):
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # decoder
        d1 = self.dec1(e4)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        logits = self.dec4(d3)
        
        logits = self._match_spatial(logits, x.shape[2:])

        masks = F.softmax(logits, dim=1)
        mask_voice = masks[:, 0:1, :, :]
        mask_noise = masks[:, 1:2, :, :]

        est_voice = mask_voice * x
        est_noise = mask_noise * x
        
        return est_voice, est_noise, masks

    # -------- losses / utils --------
    def _shared_step(self, batch, stage: str):
        x, y = batch
        est_voice, est_noise, masks = self(x)
        loss_voice = self.spectrogram_db_loss(est_voice, y["voice"])
        loss_noise = self.spectrogram_db_loss(est_noise, y["noise"])
        loss = loss_voice + loss_noise
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def l1_loss(self, pred, target):
        return F.l1_loss(pred, target)

    def spectrogram_db_loss(self, pred, target, eps=1e-8, freq_weights=None):
        est_power = pred ** 2
        tgt_power = target ** 2
        est_db = 10 * torch.log10(est_power + eps)
        tgt_db = 10 * torch.log10(tgt_power + eps)
        diff = torch.abs(est_db - tgt_db)
        if freq_weights is not None:
            while freq_weights.dim() < diff.dim():
                freq_weights = freq_weights.unsqueeze(0)
            diff = diff * freq_weights
        return diff.mean()

    def _match_spatial(self, tensor, target_hw):
        _, _, h, w = tensor.shape
        th, tw = target_hw
        if h > th or w > tw:
            tensor = tensor[:, :, :th, :tw]
        pad_h = max(0, th - tensor.shape[2])
        pad_w = max(0, tw - tensor.shape[3])
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)