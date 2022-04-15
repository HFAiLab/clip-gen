import torch
import torch.nn as nn

from .codec import Encoder, Decoder
from .quantizer import VectorQuantizer
from .discriminator import Discriminator


class VQGAN(nn.Module):

    def __init__(self, codebook_size, n_embed):
        super().__init__()

        self.encoder = Encoder(z_channels=n_embed)
        self.decoder = Decoder(z_channels=n_embed)
        self.quantizer = VectorQuantizer(codebook_size, n_embed)
        self.discriminator = Discriminator()

    def encode(self, x):
        z = self.encoder(x)  # map to latent space, [N, C, H, W]
        z_q, loss_q, indices = self.quantizer(z)  # quantize
        return z_q, loss_q, indices

    def decode(self, z):
        # reconstruct images
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, stage: int):

        if stage == 0:
            # Stage 0: training E + G + Q
            z, loss_q, _ = self.encode(x)
            x_recon = self.decode(z)
            logits_fake = self.discriminator(x_recon)

            return x_recon, loss_q, logits_fake

        elif stage == 1:
            # Stage 1: training D
            with torch.no_grad():
                z, loss_q, _ = self.encode(x)
                x_recon = self.decode(z)

            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())
            return logits_real, logits_fake

        else:
            raise ValueError(f"Invalid stage: {stage}")
