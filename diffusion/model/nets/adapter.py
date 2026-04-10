import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock
from diffusion.model.nets.smfanet_blocks_official import FMB

try:
    from mmcv.ops import CARAFEPack
except Exception:
    from diffusion.model.nets.carafe import CARAFEPack


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


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=1, keepdim=True, unbiased=False)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = channels * dw_expand
        ffn_channel = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, stride=1, padding=1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, stride=1, padding=0),
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, kernel_size=1, stride=1, padding=0)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channel, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(ffn_channel // 2, channels, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class SRConvNetLSAAdapterV8(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_size: int = 1152, ref_token_hw: int = 32, structure_only: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.ref_token_hw = int(ref_token_hw)
        self.structure_only = bool(structure_only)

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

        # NOTE:
        # - 当前 ref_* path 使用 NAFBlock 作为成熟恢复模块，替代原自创 clean stem。
        # - 目的仅是提高 reference feature refinement 的先验可靠性。
        # - 不是完整的 CasSR stage-1 reproduction。
        self.ref_low_refiner = nn.Sequential(NAFBlock(128))
        self.ref_mid_refiner = nn.Sequential(NAFBlock(256))
        self.ref_high_refiner = nn.Sequential(NAFBlock(256))
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
        # NOTE: cond_tokens are currently not consumed by the CasSR-like mainline.
        # They are kept only for interface compatibility with older experiments.
        cond_tokens = fused_cond_map.flatten(2).transpose(1, 2)

        if self.structure_only:
            return {
                "cond_map": fused_cond_map,
                "cond_tokens": cond_tokens,
            }

        ref_low = self.ref_low_proj(self.ref_low_refiner(self._to_ref_token_hw(f2)))
        ref_mid = self.ref_mid_proj(self.ref_mid_refiner(self._to_ref_token_hw(f3)))
        ref_high = self.ref_high_proj(self.ref_high_refiner(self._to_ref_token_hw(f4)))

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


def build_adapter_v8(in_channels=3, hidden_size=1152, ref_token_hw=32, structure_only=True):
    return SRConvNetLSAAdapterV8(
        in_channels=in_channels,
        hidden_size=hidden_size,
        ref_token_hw=ref_token_hw,
        structure_only=structure_only,
    )


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)




class SRConvNetLSAAdapterV12(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_size: int = 1152):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)

        self.stem = nn.Conv2d(self.in_channels, 64, 3, 1, 1)
        self.stage1 = nn.Sequential(
            FMB(64, ffn_scale=2.0),
            FMB(64, ffn_scale=2.0),
        )
        self.down1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.stage2 = nn.Sequential(
            FMB(128, ffn_scale=2.0),
            FMB(128, ffn_scale=2.0),
        )
        self.down2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.stage3 = nn.Sequential(
            FMB(256, ffn_scale=2.0),
            FMB(256, ffn_scale=2.0),
            FMB(256, ffn_scale=2.0),
            FMB(256, ffn_scale=2.0),
        )
        self.stage4 = nn.Sequential(
            FMB(256, ffn_scale=2.0),
            FMB(256, ffn_scale=2.0),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * (64 + 128 + 256 + 256)),
        )

        self.up3 = CARAFEPack(
            channels=256,
            scale_factor=2,
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64,
        )
        self.up4 = CARAFEPack(
            channels=256,
            scale_factor=2,
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64,
        )

        self.proj2_hi = nn.Conv2d(128, 256, 1)
        self.proj3_hi = nn.Conv2d(256, 256, 1)
        self.proj4_hi = nn.Conv2d(256, 256, 1)
        self.fuse64 = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.GELU(),
            nn.Conv2d(512, 288, 3, 1, 1),
        )
        self.to32 = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(1152, self.hidden_size, 1),
        )

        for m in [
            self.proj2_hi,
            self.proj3_hi,
            self.proj4_hi,
            self.fuse64[0],
            self.fuse64[2],
            self.to32[1],
        ]:
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

        f3_up = self.up3(f3)
        f4_up = self.up4(f4)

        c2_64 = self.proj2_hi(f2)
        c3_64 = self.proj3_hi(f3_up)
        c4_64 = self.proj4_hi(f4_up)

        fused_64 = torch.cat([c2_64, c3_64, c4_64], dim=1)
        fused_64 = self.fuse64(fused_64)
        cond_map = self.to32(fused_64)
        cond_tokens = cond_map.flatten(2).transpose(1, 2).contiguous()

        return {
            "cond_map": cond_map,
            "cond_tokens": cond_tokens,
        }


def build_adapter_v12(in_channels=3, hidden_size=1152):
    return SRConvNetLSAAdapterV12(in_channels=in_channels, hidden_size=hidden_size)
