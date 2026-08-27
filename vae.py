"""
vae.py
The V (Vision) component: a convolutional VAE that compresses 64x64x3
frames into a 32-dim latent vector z. Architecture follows the original
paper — 4 conv layers down, 4 deconv layers back up.

This is trained on its own, on the rollout frames from collect_rollouts.py,
completely independent of the RNN (M) or controller (C) that come later.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 64x64x3 -> 32 -> 16 -> 8 -> 4 (spatial), channels grow
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2)

        # After 4 conv layers on 64x64 input, spatial dims collapse to 2x2
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)

        # Decoder: mirror the encoder
        self.fc_decode = nn.Linear(latent_dim, 1024)
        self.dec_conv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """The 'variational' trick: sample z in a way that's still differentiable."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 1024, 1, 1)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        return torch.sigmoid(self.dec_conv4(h))  # pixels in [0, 1]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
    """
    Reconstruction loss (how well did we rebuild the frame) plus KL
    divergence (how close is our latent distribution to a standard
    normal — this is what keeps the latent space smooth/interpolatable,
    not just a lookup table of memorized frames).
    """
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + kld_weight * kld, recon_loss, kld