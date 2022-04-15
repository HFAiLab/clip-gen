import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(self, codebook_size, embed_dim, beta=0.2):
        super().__init__()

        self.beta = beta
        self.K = codebook_size
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, x):
        indices = self.encode(x)
        x_q = self.decode(indices)  # [B, H, W, C]
        x_q = x_q.permute(0, 3, 1, 2)  # [B, C, H, W]

        # quantization loss
        loss = F.mse_loss(x, x_q.detach()) + self.beta * F.mse_loss(x.detach(), x_q)

        # pass gradient to x
        x_q = x + (x_q - x).detach()

        return x_q, loss, indices

    def encode(self, x):
        B, C, H, W = x.shape
        vectors = x.permute(0, 2, 3, 1).reshape(-1, C)      # [B*H*W, C]
        dist = torch.cdist(vectors, self.embedding.weight)  # [B*H*W, K]
        indices = dist.argmin(1).view(B, H, W)

        return indices

    def decode(self, indices):
        embeddings = self.embedding(indices)  # [B, H, W, C]
        return embeddings
