import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.nets.srconvnet_blocks import SRConvNetBlock


class ComponentBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.mask_head = nn.Conv2d(channels, 1, 1)
        self.feat_head = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        h = self.block(x)
        mask_logits = self.mask_head(h)
        mask = torch.sigmoid(mask_logits)
        comp_feat = self.feat_head(h)
        return mask, comp_feat * mask


class AuxSRHead(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_ch, 3, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor, out_hw):
        x = self.body(feat)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


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

        self.trunk_to_comp = nn.Conv2d(768, 256, 1)
        self.flat_branch = ComponentBranch(256)
        self.edge_branch = ComponentBranch(256)
        self.corner_branch = ComponentBranch(256)
        self.comp_fuse = nn.Sequential(
            nn.Conv2d(256 * 3, hidden_size, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
        )

        self.comp_token_proj_flat = nn.Conv2d(256, hidden_size, 1)
        self.comp_token_proj_edge = nn.Conv2d(256, hidden_size, 1)
        self.comp_token_proj_corner = nn.Conv2d(256, hidden_size, 1)
        self.comp_token_fuse = nn.Linear(hidden_size * 3, hidden_size)

        self.aux_head_flat = AuxSRHead(256)
        self.aux_head_edge = AuxSRHead(256)
        self.aux_head_corner = AuxSRHead(256)

        for m in [self.proj2, self.proj3, self.proj4, self.trunk_to_comp]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _film(feat: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return (1.0 + gamma[:, :, None, None]) * feat + beta[:, :, None, None]

    def _to_tokens(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        t = proj(feat).flatten(2).transpose(1, 2)
        return t

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

        shared_f = self.trunk_to_comp(torch.cat([c2, c3, c4], dim=1))

        mask_flat, feat_flat = self.flat_branch(shared_f)
        mask_edge, feat_edge = self.edge_branch(shared_f)
        mask_corner, feat_corner = self.corner_branch(shared_f)

        comp_cat = torch.cat([feat_flat, feat_edge, feat_corner], dim=1)
        cond_map = self.comp_fuse(comp_cat)

        tok_flat = self._to_tokens(feat_flat, self.comp_token_proj_flat)
        tok_edge = self._to_tokens(feat_edge, self.comp_token_proj_edge)
        tok_corner = self._to_tokens(feat_corner, self.comp_token_proj_corner)
        cond_tokens = self.comp_token_fuse(torch.cat([tok_flat, tok_edge, tok_corner], dim=-1))

        out_hw = (lr_small.shape[-2] * 4, lr_small.shape[-1] * 4)
        aux_flat = self.aux_head_flat(feat_flat, out_hw)
        aux_edge = self.aux_head_edge(feat_edge, out_hw)
        aux_corner = self.aux_head_corner(feat_corner, out_hw)

        return {
            "cond_map": cond_map,
            "cond_tokens": cond_tokens,
            "comp_masks": {
                "flat": mask_flat,
                "edge": mask_edge,
                "corner": mask_corner,
            },
            "comp_aux": {
                "flat": aux_flat,
                "edge": aux_edge,
                "corner": aux_corner,
            }
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
