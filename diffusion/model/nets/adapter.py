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

        self.comp_mask_head = nn.Sequential(
            nn.Conv2d(768, 128, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 3, 1),
        )

        self.comp_cond_flat = nn.Conv2d(768, self.hidden_size, 1)
        self.comp_cond_edge = nn.Conv2d(768, self.hidden_size, 1)
        self.comp_cond_corner = nn.Conv2d(768, self.hidden_size, 1)

        self.comp_latent_flat = nn.Conv2d(768, 4, 1)
        self.comp_latent_edge = nn.Conv2d(768, 4, 1)
        self.comp_latent_corner = nn.Conv2d(768, 4, 1)

        self.comp_res_gain = nn.Parameter(torch.tensor(0.0))

        for m in [self.proj2, self.proj3, self.proj4, self.out_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

        nn.init.kaiming_normal_(self.comp_mask_head[0].weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.zeros_(self.comp_mask_head[0].bias)
        nn.init.zeros_(self.comp_mask_head[2].weight)
        nn.init.zeros_(self.comp_mask_head[2].bias)

        for m in [
            self.comp_cond_flat, self.comp_cond_edge, self.comp_cond_corner,
            self.comp_latent_flat, self.comp_latent_edge, self.comp_latent_corner,
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

        f2_32 = F.interpolate(f2, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.proj2(f2_32)
        c3 = self.proj3(f3)
        c4 = self.proj4(f4)

        base = torch.cat([c2, c3, c4], dim=1)
        base_cond = self.out_proj(base)

        comp_logits = self.comp_mask_head(base)
        comp_prob = torch.softmax(comp_logits, dim=1)

        m_flat = comp_prob[:, 0:1]
        m_edge = comp_prob[:, 1:2]
        m_corner = comp_prob[:, 2:3]

        base_flat = base * m_flat
        base_edge = base * m_edge
        base_corner = base * m_corner

        comp_cond_res = (
            self.comp_cond_flat(base_flat)
            + self.comp_cond_edge(base_edge)
            + self.comp_cond_corner(base_corner)
        )

        cond_map = base_cond + self.comp_res_gain.to(dtype=base_cond.dtype) * comp_cond_res
        cond_tokens = cond_map.flatten(2).transpose(1, 2)

        comp_latent_preds = {
            "flat": self.comp_latent_flat(base_flat),
            "edge": self.comp_latent_edge(base_edge),
            "corner": self.comp_latent_corner(base_corner),
        }

        return {
            "cond_map": cond_map,
            "cond_tokens": cond_tokens,
            "comp_logits": comp_logits,
            "comp_prob": comp_prob,
            "comp_latent_preds": comp_latent_preds,
        }


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
