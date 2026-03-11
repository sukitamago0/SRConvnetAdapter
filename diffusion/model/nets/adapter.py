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

        c2, c3, c4 = self._infer_feature_channels()
        self.proj2 = nn.Conv2d(c2, 256, 1)
        self.proj3 = nn.Conv2d(c3, 256, 1)
        self.proj4 = nn.Conv2d(c4, 256, 1)
        self.out_proj = nn.Conv2d(768, self.hidden_size, 1)

        self.mem_dim = 512
        self.mem_proj_f2 = nn.Conv2d(c2, self.mem_dim, 1)
        self.mem_proj_f3 = nn.Conv2d(c3, self.mem_dim, 1)
        self.mem_proj_f4 = nn.Conv2d(c4, self.mem_dim, 1)

        self.scale_embed_f2 = nn.Parameter(torch.zeros(1, 1, self.mem_dim))
        self.scale_embed_f3 = nn.Parameter(torch.zeros(1, 1, self.mem_dim))
        self.scale_embed_f4 = nn.Parameter(torch.zeros(1, 1, self.mem_dim))
        nn.init.normal_(self.scale_embed_f2, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embed_f3, mean=0.0, std=0.02)
        nn.init.normal_(self.scale_embed_f4, mean=0.0, std=0.02)

        self.resampler_f2 = MultiScaleResampler(
            latent_tokens=64,
            dim=self.mem_dim,
            heads=8,
            depth=2,
            ffn_expansion=4,
        )
        self.resampler_f3 = MultiScaleResampler(
            latent_tokens=32,
            dim=self.mem_dim,
            heads=8,
            depth=2,
            ffn_expansion=4,
        )
        self.resampler_f4 = MultiScaleResampler(
            latent_tokens=16,
            dim=self.mem_dim,
            heads=8,
            depth=2,
            ffn_expansion=4,
        )

        self.memory_out_proj = nn.Linear(self.mem_dim, self.hidden_size)
        self.memory_ln = nn.LayerNorm(self.hidden_size)
        self._printed_shape_debug = False

        for m in [self.proj2, self.proj3, self.proj4, self.out_proj]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)

        print(f"[AdapterChannelCheck] c2={c2}, c3={c3}, c4={c4}, mem_dim={self.mem_dim}")

    def _infer_feature_channels(self):
        with torch.no_grad():
            p = next(self.stem.parameters())
            dummy = torch.zeros(1, 3, 64, 64, device=p.device, dtype=p.dtype)
            f1 = self.stage1(self.stem(dummy))
            f2 = self.stage2(self.down1(f1))
            f3 = self.stage3(self.down2(f2))
            f4 = self.stage4(f3)
        return int(f2.shape[1]), int(f3.shape[1]), int(f4.shape[1])

    @staticmethod
    def _build_2d_sincos(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if dim % 4 != 0:
            raise ValueError(f"Expected dim%4==0 for 2D sin-cos encoding, got dim={dim}")
        gy = torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32)
        gx = torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(gy, gx, indexing="ij")
        half = dim // 2
        base = torch.arange(half // 2, device=device, dtype=torch.float32)
        inv = 1.0 / (10000 ** (base / max(1.0, float((half // 2) - 1))))
        py = yy.reshape(-1, 1) * inv.reshape(1, -1)
        px = xx.reshape(-1, 1) * inv.reshape(1, -1)
        pos = torch.cat([torch.sin(py), torch.cos(py), torch.sin(px), torch.cos(px)], dim=-1)
        return pos.to(dtype=dtype)

    def _tokenize(self, feat: torch.Tensor, scale_embed: torch.Tensor) -> torch.Tensor:
        _, c, h, w = feat.shape
        tok = feat.flatten(2).transpose(1, 2)
        pos = self._build_2d_sincos(h, w, c, feat.device, feat.dtype)
        tok = tok + pos.unsqueeze(0) + scale_embed.to(dtype=tok.dtype)
        return tok

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

        tf2 = self._tokenize(self.mem_proj_f2(f2), self.scale_embed_f2)
        tf3 = self._tokenize(self.mem_proj_f3(f3), self.scale_embed_f3)
        tf4 = self._tokenize(self.mem_proj_f4(f4), self.scale_embed_f4)

        m2 = self.resampler_f2(tf2)
        m3 = self.resampler_f3(tf3)
        m4 = self.resampler_f4(tf4)

        memory_tokens = torch.cat([m2, m3, m4], dim=1)
        memory_tokens = self.memory_ln(self.memory_out_proj(memory_tokens))

        if not self._printed_shape_debug:
            print(
                f"[AdapterShapeCheck] f2={tuple(f2.shape)} f3={tuple(f3.shape)} f4={tuple(f4.shape)} | "
                f"tf2={tuple(tf2.shape)} tf3={tuple(tf3.shape)} tf4={tuple(tf4.shape)} | "
                f"memory={tuple(memory_tokens.shape)}"
            )
            self._printed_shape_debug = True

        return {
            "memory_tokens": memory_tokens,
            "memory_meta": {"n_f2": 64, "n_f3": 32, "n_f4": 16},
        }


class PerScaleResamplerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ffn_expansion: int = 4):
        super().__init__()
        self.ln_latent = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ln_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    def forward(self, latent: torch.Tensor, src_tokens: torch.Tensor) -> torch.Tensor:
        q = self.ln_latent(latent)
        kv = self.ln_kv(src_tokens)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        latent = latent + attn_out
        latent = latent + self.ffn(self.ln_ffn(latent))
        return latent


class MultiScaleResampler(nn.Module):
    def __init__(self, latent_tokens: int, dim: int = 512, heads: int = 8, depth: int = 2, ffn_expansion: int = 4):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, latent_tokens, dim) * 0.02)
        self.blocks = nn.ModuleList([PerScaleResamplerBlock(dim=dim, heads=heads, ffn_expansion=ffn_expansion) for _ in range(depth)])
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, src_tokens: torch.Tensor) -> torch.Tensor:
        latent = self.latent.expand(src_tokens.shape[0], -1, -1)
        for blk in self.blocks:
            latent = blk(latent, src_tokens)
        return self.out_ln(latent)


def build_adapter_v8(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)


def build_adapter_v7(in_channels=3, hidden_size=1152, injection_layers_map=None):
    del in_channels, injection_layers_map
    return SRConvNetLSAAdapter(hidden_size=hidden_size)
