import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock


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

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * (64 + 128 + 256 + 256)),
        )

        self.proj2 = nn.Conv2d(128, 256, 1)
        self.proj3 = nn.Conv2d(256, 256, 1)
        self.proj4 = nn.Conv2d(256, 256, 1)
        self.out_proj = nn.Conv2d(768, self.hidden_size, 1)

        for m in [self.proj2, self.proj3, self.proj4, self.out_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

    def forward(self, lr_small: torch.Tensor, t_embed: torch.Tensor = None):
        f1 = self.stage1(self.stem(lr_small))
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
        cond_map = self.out_proj(torch.cat([c2, c3, c4], dim=1))
        cond_tokens = cond_map.flatten(2).transpose(1, 2)

        return {
            "cond_map": cond_map,
            "cond_tokens": cond_tokens,
        }


class DegradationCleanStem(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()
        self.out = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.act(self.pw(self.dw(x))))


class SRConvNetLSAAdapterV8(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_size: int = 1152, ref_token_hw: int = 32):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.ref_token_hw = int(ref_token_hw)

        self.stem = nn.Conv2d(self.in_channels, 64, 3, padding=1)
        self.stage1 = nn.Sequential(SRConvNetBlock(64), SRConvNetBlock(64))
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(SRConvNetBlock(128), SRConvNetBlock(128))
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256), SRConvNetBlock(256))
        self.stage4 = nn.Sequential(SRConvNetBlock(256), SRConvNetBlock(256))

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * (64 + 128 + 256 + 256)),
        )

        self.proj2 = nn.Conv2d(128, 256, 1)
        self.proj3 = nn.Conv2d(256, 256, 1)
        self.proj4 = nn.Conv2d(256, 256, 1)
        self.out_proj = nn.Conv2d(768, self.hidden_size, 1)

        self.ref_low_stem = DegradationCleanStem(128)
        self.ref_mid_stem = DegradationCleanStem(256)
        self.ref_high_stem = DegradationCleanStem(256)
        self.ref_low_proj = nn.Conv2d(128, self.hidden_size, 1)
        self.ref_mid_proj = nn.Conv2d(256, self.hidden_size, 1)
        self.ref_high_proj = nn.Conv2d(256, self.hidden_size, 1)

        self.lr_low_proj = nn.Conv2d(128, self.hidden_size, 1)
        self.lr_mid_proj = nn.Conv2d(256, self.hidden_size, 1)
        self.lr_high_proj = nn.Conv2d(256, self.hidden_size, 1)

        for m in [
            self.proj2, self.proj3, self.proj4, self.out_proj,
            self.ref_low_proj, self.ref_mid_proj, self.ref_high_proj,
            self.lr_low_proj, self.lr_mid_proj, self.lr_high_proj,
        ]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

    def _to_ref_token_hw(self, feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat, output_size=(self.ref_token_hw, self.ref_token_hw))

    def forward(self, lr_small: torch.Tensor, t_embed: torch.Tensor = None):
        f1 = self.stage1(self.stem(lr_small))
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
        fused_cond_map = self.out_proj(torch.cat([c2, c3, c4], dim=1))
        cond_tokens = fused_cond_map.flatten(2).transpose(1, 2)

        ref_low = self.ref_low_proj(self.ref_low_stem(self._to_ref_token_hw(f2)))
        ref_mid = self.ref_mid_proj(self.ref_mid_stem(self._to_ref_token_hw(f3)))
        ref_high = self.ref_high_proj(self.ref_high_stem(self._to_ref_token_hw(f4)))

        lr_low = self.lr_low_proj(self._to_ref_token_hw(f2))
        lr_mid = self.lr_mid_proj(self._to_ref_token_hw(f3))
        lr_high = self.lr_high_proj(self._to_ref_token_hw(f4))

        return {
            "cond_map": fused_cond_map,
            "cond_tokens": cond_tokens,
            "ref_low": ref_low,
            "ref_mid": ref_mid,
            "ref_high": ref_high,
            "lr_low": lr_low,
            "lr_mid": lr_mid,
            "lr_high": lr_high,
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None, ref_token_hw=32):
    del injection_layers_map
    return SRConvNetLSAAdapterV8(in_channels=in_channels, hidden_size=hidden_size, ref_token_hw=ref_token_hw)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
