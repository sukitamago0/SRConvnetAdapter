import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFEPack(nn.Module):
    """Pure-PyTorch CARAFE-style upsampling fallback.

    API-compatible with mmcv.ops.CARAFEPack for this project usage.
    """

    def __init__(
        self,
        channels,
        scale_factor=2,
        up_kernel=5,
        up_group=1,
        encoder_kernel=3,
        encoder_dilation=1,
        compressed_channels=64,
    ):
        super().__init__()
        self.channels = int(channels)
        self.scale_factor = int(scale_factor)
        self.up_kernel = int(up_kernel)
        self.up_group = int(up_group)
        self.compressed_channels = int(compressed_channels)

        mask_channels = (self.scale_factor ** 2) * (self.up_kernel ** 2) * self.up_group
        padding = (encoder_kernel + (encoder_kernel - 1) * (encoder_dilation - 1) - 1) // 2

        self.channel_compressor = nn.Conv2d(self.channels, self.compressed_channels, 1)
        self.mask_encoder = nn.Conv2d(
            self.compressed_channels,
            mask_channels,
            kernel_size=encoder_kernel,
            stride=1,
            padding=padding,
            dilation=encoder_dilation,
        )

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale_factor
        k = self.up_kernel

        feat = self.channel_compressor(x)
        mask = self.mask_encoder(feat)
        mask = F.pixel_shuffle(mask, s)  # [B, k*k*group, H*s, W*s]
        mask = mask.view(b, self.up_group, k * k, h * s, w * s)
        mask = torch.softmax(mask, dim=2)

        x_up = F.interpolate(x, scale_factor=s, mode="nearest")
        patches = F.unfold(x_up, kernel_size=k, padding=k // 2)
        patches = patches.view(b, self.up_group, c // self.up_group, k * k, h * s, w * s)
        out = (patches * mask.unsqueeze(2)).sum(dim=3)
        out = out.view(b, c, h * s, w * s)
        return out
