import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


def zero_module(module: nn.Module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class SFTLayer(nn.Module):
    def __init__(self, cond_nc=64, feat_nc=1152):
        super().__init__()
        self.scale_conv0 = nn.Conv2d(cond_nc, cond_nc, 1)
        self.scale_conv1 = nn.Conv2d(cond_nc, feat_nc, 1)
        self.shift_conv0 = nn.Conv2d(cond_nc, cond_nc, 1)
        self.shift_conv1 = nn.Conv2d(cond_nc, feat_nc, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, feat, cond, strength=1.0):
        scale = self.scale_conv1(self.act(self.scale_conv0(cond)))
        shift = self.shift_conv1(self.act(self.shift_conv0(cond)))
        gain = self.w * float(strength)
        return feat * (1 + gain * scale) + gain * shift, scale, shift


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(
        self,
        force_null_caption: bool = True,
        anchor_layers=None,
        semantic_layers=None,
        **kwargs
    ):
        kwargs.setdefault("model_max_length", 300)
        kwargs.setdefault("pred_sigma", False)
        kwargs.setdefault("learn_sigma", False)
        out_channels = kwargs.pop("out_channels", None)
        super().__init__(**kwargs)

        if out_channels is not None and int(out_channels) != int(self.out_channels):
            self.out_channels = int(out_channels)
            head_hidden = self.x_embedder.proj.out_channels
            self.final_layer = T2IFinalLayer(head_hidden, self.patch_size, self.out_channels)
            nn.init.trunc_normal_(self.final_layer.linear.weight, std=0.02)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        self.depth = len(self.blocks)
        self.hidden_size = self.x_embedder.proj.out_channels
        self.force_null_caption = bool(force_null_caption)
        self.aug_embedder = TimestepEmbedder(self.hidden_size)

        self.lr_token_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.lr_token_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.lr_token_dropout = nn.Dropout(0.0)
        nn.init.normal_(self.lr_token_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lr_token_proj.bias)

        self.local_entry_proj = nn.Conv2d(self.hidden_size, self.hidden_size, 1, 1, 0, bias=True)
        self.local_entry_gate = nn.Parameter(torch.tensor(-4.0))
        nn.init.zeros_(self.local_entry_proj.weight)
        nn.init.zeros_(self.local_entry_proj.bias)

        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        self.local_sft_start_layer = 16
        self.local_sft_end_layer = self.depth

        self._last_sft_stats = None
        self._last_image_cond_stats = None

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        sft_strength = float(kwargs.pop("sft_strength", 1.0))
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size)
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed
        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        cond_map = None
        cond_tokens = None
        if adapter_cond is not None:
            cond_map = adapter_cond.get("cond_map", None)
            cond_tokens = adapter_cond.get("cond_tokens", None)

        lr_cond_tokens = None
        if cond_tokens is not None:
            lr_cond_tokens = self.lr_token_proj(self.lr_token_norm(cond_tokens.to(dtype=x.dtype)))
            lr_cond_tokens = self.lr_token_dropout(lr_cond_tokens)

        entry_local_tokens = None
        cond_red = None
        if cond_map is not None:
            cond_map_x = cond_map.to(dtype=x.dtype)
            if cond_map_x.shape[-2:] != (self.h, self.w):
                cond_map_x = F.interpolate(cond_map_x, size=(self.h, self.w), mode="bilinear", align_corners=False)
            entry_local_tokens = self.local_entry_proj(cond_map_x).flatten(2).transpose(1, 2).contiguous()
            cond_red = self.sft_cond_reduce(cond_map_x)

        if entry_local_tokens is not None:
            x = x + torch.sigmoid(self.local_entry_gate).to(x.dtype).view(1, 1, 1) * entry_local_tokens

        t0 = self.t_block(t)
        if force_drop_ids is None and self.force_null_caption:
            force_drop_ids = torch.ones(bs, device=y.device, dtype=torch.long)
        y = self.y_embedder(y, self.training, force_drop_ids=force_drop_ids)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        sft_delta_stds = []
        image_stats = []
        for i, block in enumerate(self.blocks):
            if cond_red is not None and (self.local_sft_start_layer <= i < self.local_sft_end_layer):
                b, n, c = x.shape
                if n != self.h * self.w:
                    raise RuntimeError(f"Token grid mismatch: N={n}, expected H*W={self.h * self.w}")
                x_map = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                x_map_pre = x_map
                x_map, _, _ = self.sft_layers[i](x_map, cond_red, strength=sft_strength)
                sft_delta_stds.append(float((x_map - x_map_pre).detach().float().std().item()))
                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()

            block_kwargs = dict(
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                image_cond=lr_cond_tokens,
                image_gate=None,
                **kwargs,
            )
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, **block_kwargs)
            blk_stats = getattr(block, "_last_image_stats", None)
            if blk_stats is not None:
                image_stats.append(blk_stats)

        self._last_sft_stats = {
            "sft_strength": float(sft_strength),
            "sft_delta_std": float(sum(sft_delta_stds) / max(1, len(sft_delta_stds))),
            "late_sft_active_start": int(self.local_sft_start_layer),
            "local_entry_gate": float(torch.sigmoid(self.local_entry_gate.detach()).item()),
        }

        if len(image_stats) > 0:
            self._last_image_cond_stats = {
                "avg_image_alpha": float(sum(float(s.get("image_alpha_value", 0.0)) for s in image_stats) / len(image_stats)),
                "lr_cross_text_ctx_std": float(sum(float(s.get("text_ctx_std", 0.0)) for s in image_stats) / len(image_stats)),
                "lr_cross_img_delta_std": float(sum(float(s.get("img_delta_std", 0.0)) for s in image_stats) / len(image_stats)),
                "cross_out_std": float(sum(float(s.get("cross_out_std", 0.0)) for s in image_stats) / len(image_stats)),
            }
        else:
            self._last_image_cond_stats = None

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
