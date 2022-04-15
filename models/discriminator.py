import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_c=3, ch=64, n_layer=3):
        super().__init__()

        modules = [nn.Conv2d(in_c, ch, kernel_size=4, stride=2, padding=1)]
        modules += [nn.LeakyReLU(0.2, True)]

        chs = [ch * min(2 ** i, 8) for i in range(n_layer + 1)]

        # increase channels
        for i in range(1, n_layer + 1):
            stride = 2 if i != n_layer else 1
            modules += [
                nn.Conv2d(chs[i - 1], chs[i], kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(chs[i]),
                nn.LeakyReLU(0.2, True)
            ]

        self.features = nn.Sequential(*modules)
        self.head = nn.Conv2d(chs[-1], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.features(x)
        out = self.head(x)
        return out
