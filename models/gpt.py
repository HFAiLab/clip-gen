import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, block_size) -> None:
        super().__init__()

        self.encoder = nn.TransformerEncoderLayer(
            n_embd,
            n_head,
            n_embd * 4,
            dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        mask = torch.ones(block_size, block_size, dtype=torch.bool)
        mask = ~torch.tril(mask)  # top-left is False, up-right is True
        self.register_buffer("mask", mask)

    def forward(self, x):
        L = x.size(1)
        assert L <= self.mask.size(0)
        mask = self.mask[:L, :L]

        return self.encoder(x, mask)


class GPT(nn.Module):

    def __init__(self, vocab_size, n_layer, n_embed, n_head, block_size=256, n_cond_embed=512, dropout=0.1):
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embed))
        self.cond_proj = nn.Linear(n_cond_embed, n_embed)
        self.drop = nn.Dropout(dropout)

        blocks = [Block(n_embed, n_head, dropout, block_size) for _ in range(n_layer)]
        self.blocks = nn.Sequential(*blocks)

        self.norm = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size)
        self.block_size = block_size

    def forward(self, idx, embed):
        x = self.tok_emb(idx)
        embed = self.cond_proj(embed).unsqueeze(1)  # (N, 1, n_embed)
        x = torch.cat([embed, x], dim=1)  # (N, L+1, n_embed)

        assert x.size(1) <= self.block_size
        x = x + self.pos_emb[:, :x.size(1)]  # (N, L+1, n_embed)
        x = self.drop(x)

        x = self.blocks(x)
        logits = self.head(self.norm(x))

        return logits

    @torch.no_grad()
    def sample(self, embed, steps, temperature=1.0, top_k=None, top_p=1.0):
        N = embed.size(0)
        indices = torch.zeros(N, 0).long().to(embed.device)

        for _ in range(steps):
            logits = self(indices, embed)         # [B, L, vocab_size]
            logits = logits[:, -1] / temperature  # [B, vocab_size]
            logits = self.top_k_top_p(logits, top_k, top_p)

            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            indices = torch.cat((indices, idx), dim=1)

        return indices

    @staticmethod
    def top_k_top_p(logits, top_k=None, top_p=1.0):
        if top_k is not None:
            assert 1 <= top_k <= logits.size(-1)
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[..., [-1]]] = -torch.inf

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            mask = cum_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = 0

            # scatter sorted tensors to original indexing
            mask = mask.scatter(1, sorted_indices, mask)
            logits[mask] = -torch.inf

        return logits


def gpt2_medium(vocab_size, block_size=256, n_cond_embed=512, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=24,
        n_embed=1024,
        n_head=16,
        block_size=block_size,
        n_cond_embed=n_cond_embed,
        dropout=dropout,
    )


def gpt2_large(vocab_size, block_size=256, n_cond_embed=512, dropout=0.1):
    return GPT(
        vocab_size=vocab_size,
        n_layer=36,
        n_embed=1280,
        n_head=20,
        block_size=block_size,
        n_cond_embed=n_cond_embed,
        dropout=dropout,
    )
