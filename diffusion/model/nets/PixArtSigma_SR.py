import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer, DecoupledImageTextCrossAttention, MultiHeadCrossAttention
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


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
    def __init__(self, force_null_caption: bool = True, **kwargs):
        pixel_layers = kwargs.pop("pixel_layers", range(0, 16))
        lr_conv_layers = kwargs.pop("lr_conv_layers", range(0, 8))
        semantic_layers = kwargs.pop("semantic_layers", [20, 21, 22, 23, 24, 25, 26, 27])
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
        self.pixel_layers = tuple(int(i) for i in pixel_layers)
        self.lr_conv_layers = tuple(int(i) for i in lr_conv_layers)
        self.semantic_layers = tuple(int(i) for i in semantic_layers)
        self.anchor_layers = self.pixel_layers
        self.aug_embedder = TimestepEmbedder(self.hidden_size)

        self.lr_ip_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.lr_ip_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        nn.init.normal_(self.lr_ip_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.lr_ip_proj.bias)
        self.sem_ip_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.sem_ip_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        nn.init.normal_(self.sem_ip_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sem_ip_proj.bias)

        self.local_entry_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.local_entry_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.local_entry_gate = nn.Parameter(torch.full((1,), -4.0))
        nn.init.zeros_(self.local_entry_proj.weight)
        nn.init.zeros_(self.local_entry_proj.bias)

        self._last_sft_stats = None
        self._last_image_stats = None
        self._last_image_cond_stats = None

    def enable_sr_conditioning_layers(
        self,
        pixel_layers=None,
        lr_conv_layers=None,
        semantic_layers=None,
        init_gate: float = -4.0,
    ):
        pixel_layers = self.pixel_layers if pixel_layers is None else pixel_layers
        lr_conv_layers = self.lr_conv_layers if lr_conv_layers is None else lr_conv_layers
        semantic_layers = self.semantic_layers if semantic_layers is None else semantic_layers
        pixel_set = set(int(i) for i in pixel_layers)
        lr_conv_set = set(int(i) for i in lr_conv_layers)
        semantic_set = set(int(i) for i in semantic_layers)

        self.pixel_layers = tuple(sorted(pixel_set))
        self.lr_conv_layers = tuple(sorted(lr_conv_set))
        self.semantic_layers = tuple(sorted(semantic_set))
        self.anchor_layers = self.pixel_layers

        for i, block in enumerate(self.blocks):
            block.block_id = int(i)
            block.is_pixel_layer = i in pixel_set
            block.is_lr_conv_layer = i in lr_conv_set
            block.is_semantic_layer = i in semantic_set
            with torch.no_grad():
                block.lr_ip_gate.fill_(float(init_gate))
                block.lr_stream_gate.fill_(float(init_gate))
                block.lr_cross_conv_gate.fill_(float(init_gate))
                block.semantic_gate.fill_(float(init_gate))
            if i in semantic_set and isinstance(block.cross_attn, MultiHeadCrossAttention):
                block.cross_attn = DecoupledImageTextCrossAttention.from_text_cross_attn(block.cross_attn)
        return self

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size)
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed

        ip_tokens = None
        local_entry_tokens = None
        sem_tokens = None
        if isinstance(adapter_cond, dict):
            ip_tokens = adapter_cond.get("ip_tokens", adapter_cond.get("cond_tokens", None))
            local_entry_tokens = adapter_cond.get("local_entry_tokens", None)
            sem_tokens = adapter_cond.get("sem_tokens", None)
            if local_entry_tokens is None and adapter_cond.get("cond_map", None) is not None:
                local_entry_tokens = adapter_cond["cond_map"].flatten(2).transpose(1, 2).contiguous()

        lr_ip_tokens = None
        lr_stream_tokens = None
        if ip_tokens is not None:
            lr_ip_tokens = self.lr_ip_proj(self.lr_ip_norm(ip_tokens.to(dtype=x.dtype)))
            lr_stream_tokens = lr_ip_tokens
        if sem_tokens is not None:
            sem_tokens = self.sem_ip_proj(self.sem_ip_norm(sem_tokens.to(dtype=x.dtype)))

        if local_entry_tokens is not None:
            if local_entry_tokens.shape[1] != x.shape[1]:
                raise RuntimeError(
                    f"local_entry_tokens length mismatch: got {local_entry_tokens.shape[1]}, expected {x.shape[1]}"
                )
            local_bias = self.local_entry_proj(self.local_entry_norm(local_entry_tokens.to(dtype=x.dtype)))
            x = x + torch.sigmoid(self.local_entry_gate).to(dtype=x.dtype).view(1, 1, 1) * local_bias

        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

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

        per_block_image_stats = []
        per_block_sem_stats = []
        for block in self.blocks:
            block_out = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                HW=(self.h, self.w),
                lr_ip_tokens=lr_ip_tokens,
                lr_stream_tokens=lr_stream_tokens,
                sem_tokens=sem_tokens,
                sem_scale=None,
                lr_ip_scale=None,
                return_lr_stream=True,
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                **kwargs,
            )
            if isinstance(block_out, tuple):
                x, lr_stream_tokens = block_out
            else:
                x = block_out
            if hasattr(block, "_last_image_stats") and isinstance(block._last_image_stats, dict):
                per_block_image_stats.append(block._last_image_stats)
            if hasattr(block, "_last_semantic_stats") and isinstance(block._last_semantic_stats, dict):
                per_block_sem_stats.append(block._last_semantic_stats)

        self._last_sft_stats = {
            "local_entry_gate": float(torch.sigmoid(self.local_entry_gate.detach()).item()),
        }
        def _mean(key, xs):
            vals = [float(d.get(key, 0.0)) for d in xs]
            return float(sum(vals) / max(1, len(vals)))
        ip_gate_vals = [float(d.get("image_alpha_value", 0.0)) for d in per_block_image_stats]
        self._last_image_stats = {
            "lr_ip_gate_mean": float(sum(ip_gate_vals) / max(1, len(ip_gate_vals))),
            "lr_ip_gate_std": float(torch.tensor(ip_gate_vals).float().std(unbiased=False).item()) if len(ip_gate_vals) > 0 else 0.0,
            "lr_ip_attn_delta_std_mean": _mean("img_delta_std", per_block_image_stats),
            "lr_stream_delta_std_mean": _mean("lr_stream_delta_std", per_block_image_stats),
            "lr_conv_delta_std_mean": _mean("lr_conv_delta_std", per_block_image_stats),
            "semantic_gate_mean": _mean("semantic_gate", per_block_sem_stats),
            "semantic_img_delta_std_mean": _mean("semantic_img_delta_std", per_block_sem_stats),
        }
        self._last_image_cond_stats = self._last_image_stats

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
