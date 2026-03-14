import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_attn = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x1 = self.act(x1)
        attn = self.pool(x1)
        attn = torch.sigmoid(self.conv_attn(attn))
        x1 = x1 * attn
        x1 = self.conv2(x1)
        return x + x1


class FullResolutionBlock(nn.Module):
    def __init__(self, channels: int, num_fca: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[FeatureChannelAttention(channels) for _ in range(int(num_fca))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.blocks(x)


class ImageGuidanceNetwork(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_val = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_att = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv_img = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, fd: torch.Tensor, lr_full_up: torch.Tensor):
        val = self.conv_val(fd)
        att = torch.sigmoid(self.conv_att(fd))
        fr = fd + val * att
        r2 = self.conv_img(fr) + lr_full_up
        return fr, r2


class GuideSRGuidanceBranch(nn.Module):
    def __init__(self, channels: int = 64, num_frb: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.stem = nn.Conv2d(3, self.channels, 3, padding=1)
        self.frbs = nn.Sequential(*[FullResolutionBlock(self.channels, num_fca=2) for _ in range(int(num_frb))])
        self.ign = ImageGuidanceNetwork(self.channels)

    def forward(self, lr_full_up: torch.Tensor):
        f0 = self.stem(lr_full_up)
        fd = self.frbs(f0)
        fr, r2 = self.ign(fd, lr_full_up)

        g64 = F.pixel_unshuffle(fr, 8)
        g32 = F.pixel_unshuffle(fr, 16)

        return {
            "r2": r2,
            "g64": g64,
            "g32": g32,
        }
