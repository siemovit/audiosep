"""6 -level U-Net for spectrogram masking (voice only)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class SpectroUNetOriginal(L.LightningModule):
    """6 -level U-Net for spectrogram masking (voice only)."""

    def __init__(self):
        super().__init__()

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
        out = F.sigmoid(deconv6_out)
        return out

    def _shared_step(self, batch):
        mix, voc = batch
        msk = self(mix)
        loss = self.crit(msk * mix, voc)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch)

    def validation_step(self, batch, _):
        self._shared_step(batch)

    def test_step(self, batch, _):
        self._shared_step(batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)
