import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, value=0)
        x = self.conv(x)

        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        )

        self.has_shortcut = (in_c != out_c)
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.block(x)
        if self.has_shortcut:
            x = self.shortcut(x)

        return x + h


class AttnBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_c)
        self.attn = nn.MultiheadAttention(in_c, num_heads=1, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]

        out, _ = self.attn(h, h, h, need_weights=False)  # [B, H*W, C]
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)   # [B, C, H, W]
        out = x + out

        return out


class Encoder(nn.Module):
    def __init__(
        self,
        in_c=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(in_c, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)

        blocks = []
        for level in range(len(ch_mult)):
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in))

            if level != len(ch_mult) - 1:
                blocks.append(Downsample(block_in))
                curr_res = curr_res // 2

        self.down = nn.Sequential(*blocks)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(z_channels, z_channels, kernel_size=1),
        )

    def forward(self, x):
        h = self.conv_in(x)
        h = self.down(h)
        h = self.mid(h)
        h = self.final(h)

        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=3,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[16],
        dropout=0.0,
        resolution=256,
        z_channels=256,
    ):
        super().__init__()

        # number of channels at lowest res
        block_in = ch * ch_mult[len(ch_mult) - 1]

        # z to block_in
        self.quant_conv_in = nn.Conv2d(z_channels, z_channels, kernel_size=1)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Sequential(
            ResnetBlock(block_in, block_in, dropout=dropout),
            AttnBlock(block_in),
            ResnetBlock(block_in, block_in, dropout=dropout),
        )

        # upsampling
        blocks = []
        curr_res = resolution // 2 ** (len(ch_mult) - 1)
        for level in reversed(range(len(ch_mult))):
            block_out = ch * ch_mult[level]

            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(block_in, block_out, dropout=dropout))
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_out))
                block_in = block_out

            if level != 0:
                blocks.append(Upsample(block_out))
                curr_res = curr_res * 2

        self.up = nn.Sequential(*blocks)

        # end
        self.final = nn.Sequential(
            nn.GroupNorm(32, block_in),
            nn.SiLU(),
            nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z):
        h = self.quant_conv_in(z)
        h = self.conv_in(h)
        h = self.mid(h)
        h = self.up(h)
        h = self.final(h)

        return h
