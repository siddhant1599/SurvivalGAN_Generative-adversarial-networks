"""
Generator architecture for the vanilla GAN baseline.

Kept in its own file so both train_gan.py and any inference
code can construct the same network and
load weights into it.
"""

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, out_dim),
        )

    def forward(self, z):
        return self.net(z)
