import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class SpectroUNetSkip2D(L.LightningModule):
    """4-level U-Net for spectrogram masking (voice + noise).

    Input:  x  (B, 1, F, T)  magnitude spectrogram of the mixture
    Output: est_voice, est_noise, masks
            where est_* have shape (B, 1, F, T)
            and masks has shape (B, 2, F, T)
    """

    def __init__(self, in_channels=1, out_channels=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # ---------- Encoder ----------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> (B,16,F/2,T/2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> (B,32,F/4,T/4)

        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> (B,64,F/8,T/8)

        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )  # -> (B,128,F/16,T/16)

        # ---------- Decoder ----------
        def deconv(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(
                    in_c,
                    out_c,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,  # pour inverser stride=2,pad=2
                ),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.insert(1, nn.Dropout2d(0.3))
            return nn.Sequential(*layers)

        # Decoder layers with skip connections
        self.dec1 = deconv(128, 64, dropout=True)   # prend e4, concat avec e3 (64) → 128
        self.dec2 = deconv(128, 32, dropout=True)   # prend d1_cat, concat avec e2 (32) → 64
        self.dec3 = deconv(64, 16, dropout=False)   # prend d2_cat, concat avec e1 (16) → 32
        self.dec4 = nn.ConvTranspose2d(32, out_channels,
                                       kernel_size=5, stride=2,
                                       padding=2, output_padding=1)

    # ---------- Forward ----------
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # (B,16,*,*)
        e2 = self.enc2(e1)  # (B,32,*,*)
        e3 = self.enc3(e2)  # (B,64,*,*)
        e4 = self.enc4(e3)  # (B,128,*,*)

        # Decoder + skip connections
        d1 = self.dec1(e4)                             # (B,64,*,*)
        e3_resized = self._match_spatial(e3, d1.shape[2:])
        d1_cat = torch.cat([d1, e3_resized], dim=1)    # (B,64+64=128,*,*)

        d2 = self.dec2(d1_cat)                         # (B,32,*,*)
        e2_resized = self._match_spatial(e2, d2.shape[2:])
        d2_cat = torch.cat([d2, e2_resized], dim=1)    # (B,32+32=64,*,*)

        d3 = self.dec3(d2_cat)                         # (B,16,*,*)
        e1_resized = self._match_spatial(e1, d3.shape[2:])
        d3_cat = torch.cat([d3, e1_resized], dim=1)    # (B,16+16=32,*,*)

        logits = self.dec4(d3_cat)                     # (B,2,≈F,≈T)
        
        # match final spatial dims exactly with input
        logits = self._match_spatial(logits, x.shape[2:])

        # softmax over {voice, noise} so masks are complementary
        masks = F.softmax(logits, dim=1)
        mask_voice = masks[:, 0:1, :, :]
        mask_noise = masks[:, 1:2, :, :]

        est_voice = mask_voice * x    # (B,1,F,T)
        est_noise = mask_noise * x    # (B,1,F,T)

        return est_voice, est_noise, masks

    # -------- losses / utils --------
    def _shared_step(self, batch, stage: str):
        x, y = batch
        est_voice, est_noise, _ = self(x)
        # loss_voice = self.spectrogram_db_loss(est_voice, y["voice"])
        # loss_noise = self.spectrogram_db_loss(est_noise, y["noise"])
        loss_voice = self.l1_loss(est_voice, y["voice"])   
        loss_noise = self.l1_loss(est_noise, y["noise"])
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
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)