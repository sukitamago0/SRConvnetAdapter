import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock


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
        fg = fd + val * att
        r2 = lr_full_up + self.conv_img(fg)
        return fg, r2


class GuideSRFullResBranch(nn.Module):
    def __init__(self, guide_channels: int = 64):
        super().__init__()
        self.guide_channels = int(guide_channels)
        self.stem = nn.Conv2d(3, self.guide_channels, 3, padding=1)
        self.frbs = nn.Sequential(*[FullResolutionBlock(self.guide_channels, num_fca=2) for _ in range(4)])
        self.ign = ImageGuidanceNetwork(self.guide_channels)

        # Build a real multi-scale guidance stream using chained pixel-unshuffle.
        # 512 -> 256 -> 128 -> 64 (all via pixel-unshuffle), while keeping channel width controlled.
        self.scale_reduce_1 = nn.Conv2d(self.guide_channels * 4, self.guide_channels, 1)
        self.scale_reduce_2 = nn.Conv2d(self.guide_channels * 4, self.guide_channels, 1)
        self.scale_reduce_3 = nn.Conv2d(self.guide_channels * 4, self.guide_channels, 1)

    @staticmethod
    def _pixel_unshuffle_to(feat: torch.Tensor, target_hw) -> torch.Tensor:
        th, tw = int(target_hw[0]), int(target_hw[1])
        h, w = int(feat.shape[-2]), int(feat.shape[-1])
        if h == th and w == tw:
            return feat
        if h % th != 0 or w % tw != 0:
            raise ValueError(f"Cannot pixel-unshuffle from {(h, w)} to {(th, tw)}")
        sh = h // th
        sw = w // tw
        if sh != sw:
            raise ValueError(f"Non-uniform pixel-unshuffle scale is not supported: sh={sh}, sw={sw}")
        return F.pixel_unshuffle(feat, sh)

    def forward(self, lr_full_up: torch.Tensor, target_shapes):
        f0 = self.stem(lr_full_up)
        fd = self.frbs(f0)
        fg_full, r2 = self.ign(fd, lr_full_up)

        # Real pyramid features (different source scales), all downsampled by pixel-unshuffle.
        s1 = self.scale_reduce_1(F.pixel_unshuffle(fg_full, 2))  # 1/2
        s2 = self.scale_reduce_2(F.pixel_unshuffle(s1, 2))       # 1/4
        s3 = self.scale_reduce_3(F.pixel_unshuffle(s2, 2))       # 1/8

        # Map to adapter targets from different scales to avoid degenerate identical guidance tensors.
        g2 = self._pixel_unshuffle_to(s3, target_shapes[0])
        g3 = self._pixel_unshuffle_to(s2, target_shapes[1])
        g4 = self._pixel_unshuffle_to(s1, target_shapes[2])
        return fg_full, r2, [g2, g3, g4]


class SRConvNetLSAAdapter(nn.Module):
    def __init__(self, hidden_size: int = 1152):
        super().__init__()
        self.hidden_size = int(hidden_size)

        self.stem = nn.Conv2d(3, 64, 3, padding=1)
        self.stage1 = nn.Sequential(SRConvNetBlock(64), SRConvNetBlock(64))
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(SRConvNetBlock(128), SRConvNetBlock(128))
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256))
        self.stage4 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256))

        self.guide_channels = 64
        self.guide_branch = GuideSRFullResBranch(guide_channels=self.guide_channels)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * (64 + 128 + 256 + 256)),
        )

        self.proj2 = nn.Conv2d(128, 256, 1)
        self.proj3 = nn.Conv2d(256, 256, 1)
        self.proj4 = nn.Conv2d(256, 256, 1)

        # In current adapter geometry: g2/g3/g4 become 256/1024/4096 channels at 32x32.
        self.guide_to_c2 = nn.Conv2d(self.guide_channels * 4, 256, 1)
        self.guide_to_c3 = nn.Conv2d(self.guide_channels * 16, 256, 1)
        self.guide_to_c4 = nn.Conv2d(self.guide_channels * 64, 256, 1)
        self.out_proj = nn.Conv2d(768, self.hidden_size, 1)

        for m in [self.proj2, self.proj3, self.proj4, self.out_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)
        for m in [self.guide_to_c2, self.guide_to_c3, self.guide_to_c4]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

    def forward(self, lr_small_struct: torch.Tensor, lr_full_up: torch.Tensor = None, t_embed: torch.Tensor = None):
        f1 = self.stage1(self.stem(lr_small_struct))
        f2 = self.stage2(self.down1(f1))
        f3 = self.stage3(self.down2(f2))
        f4 = self.stage4(f3)

        if t_embed is not None:
            tb = self.time_mlp(t_embed)
            splits = [64, 128, 256, 256, 64, 128, 256, 256]
            g1, g2, g3, g4, b1, b2, b3, b4 = tb.split(splits, dim=-1)
            f1 = self._film(f1, g1, b1)
            f2 = self._film(f2, g2, b2)
            f3 = self._film(f3, g3, b3)
            f4 = self._film(f4, g4, b4)

        f2_32 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.proj2(f2_32)
        c3 = self.proj3(f3)
        c4 = self.proj4(f4)

        r2 = None
        guide_stats = {}
        if lr_full_up is not None:
            _, r2, (g2, g3, g4) = self.guide_branch(lr_full_up, [c2.shape[-2:], c3.shape[-2:], c4.shape[-2:]])
            c2 = c2 + self.guide_to_c2(g2)
            c3 = c3 + self.guide_to_c3(g3)
            c4 = c4 + self.guide_to_c4(g4)
            guide_stats = {
                "guide_feat_norm": float((g2.detach().float().norm().item() + g3.detach().float().norm().item() + g4.detach().float().norm().item()) / 3.0),
                "guide_to_c2_norm": float(self.guide_to_c2.weight.detach().float().norm().item()),
                "guide_to_c3_norm": float(self.guide_to_c3.weight.detach().float().norm().item()),
                "guide_to_c4_norm": float(self.guide_to_c4.weight.detach().float().norm().item()),
            }

        cond_map = self.out_proj(torch.cat([c2, c3, c4], dim=1))
        cond_tokens = cond_map.flatten(2).transpose(1, 2)

        return {
            "cond_map": cond_map,
            "cond_tokens": cond_tokens,
            "guide_residual": r2,
            "guide_stats": guide_stats,
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
