import torch
import torch.nn as nn


class SFTLayer(nn.Module):
    def __init__(self, cond_nc=64, feat_nc=1152):
        super().__init__()
        self.scale_conv0 = nn.Conv2d(cond_nc, cond_nc, 1)
        self.scale_conv1 = nn.Conv2d(cond_nc, feat_nc, 1)
        self.shift_conv0 = nn.Conv2d(cond_nc, cond_nc, 1)
        self.shift_conv1 = nn.Conv2d(cond_nc, feat_nc, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, feat, cond):
        scale = self.scale_conv1(self.act(self.scale_conv0(cond)))
        shift = self.shift_conv1(self.act(self.shift_conv0(cond)))
        return feat * (1 + scale) + shift, scale, shift


class ReferenceCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.query_ln = nn.LayerNorm(hidden_size)
        self.context_ln = nn.LayerNorm(hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_ln(query)
        kv = self.context_ln(context)
        out, _ = self.mha(q, kv, kv, need_weights=False)
        return self.out_proj(out)


class LRAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.query_ln = nn.LayerNorm(hidden_size)
        self.context_ln = nn.LayerNorm(hidden_size)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.query_ln(query)
        kv = self.context_ln(context)
        out, _ = self.mha(q, kv, kv, need_weights=False)
        return self.out_proj(out)

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(
        self,
        force_null_caption: bool = True,
        hard_injection_layers=None,
        detail_injection_layers=None,
        injection_layer_to_level=None,
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

        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        self.hard_injection_layers = set(hard_injection_layers or [2, 4, 6, 8, 10, 12])
        self.detail_injection_layers = list(detail_injection_layers or [14, 16, 18, 20, 22, 24])
        self.injection_layer_to_level = dict(injection_layer_to_level or {})
        self.detail_ref_attn = nn.ModuleDict()
        self.detail_lr_attn = nn.ModuleDict()
        self.detail_ref_gate = nn.ParameterDict()
        self.detail_lr_gate = nn.ParameterDict()
        self._last_sft_stats = None
        self._last_detail_attn_stats = None

        for layer_id in self.detail_injection_layers:
            k = str(layer_id)
            self.detail_ref_attn[k] = ReferenceCrossAttention(self.hidden_size, self.num_heads)
            self.detail_lr_attn[k] = LRAttention(self.hidden_size, self.num_heads)
            self.detail_ref_gate[k] = nn.Parameter(torch.full((1,), -8.0))
            self.detail_lr_gate[k] = nn.Parameter(torch.full((1,), -8.0))

    def _resolve_detail_level(self, layer_idx: int) -> str:
        mapped = self.injection_layer_to_level.get(layer_idx, None)
        if isinstance(mapped, str):
            mapped = mapped.lower()
            if mapped in ("low", "mid", "high"):
                return mapped
        if layer_idx in (14, 16):
            return "mid"
        if layer_idx in (18, 20, 22, 24):
            return "high"
        return "mid"

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
        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        cond_map = None
        ref_feats = {}
        lr_feats = {}
        if isinstance(adapter_cond, dict):
            cond_map = adapter_cond.get("cond_map", None)
            ref_feats = {
                "low": adapter_cond.get("ref_low", None),
                "mid": adapter_cond.get("ref_mid", None),
                "high": adapter_cond.get("ref_high", None),
            }
            lr_feats = {
                "low": adapter_cond.get("lr_low", None),
                "mid": adapter_cond.get("lr_mid", None),
                "high": adapter_cond.get("lr_high", None),
            }

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

        sft_scale_means, sft_scale_stds, sft_shift_means, sft_shift_stds = [], [], [], []
        ref_res_ratio_vals, lr_res_ratio_vals = [], []
        cond_red = None
        if cond_map is not None:
            cond_red = self.sft_cond_reduce(cond_map.to(dtype=x.dtype))

        for i, block in enumerate(self.blocks):
            if cond_red is not None and i in self.hard_injection_layers:
                b, n, c = x.shape
                if n != self.h * self.w:
                    raise RuntimeError(f"Token grid mismatch: N={n}, expected H*W={self.h * self.w} from input/patch rule")
                x_map = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                x_map, scale, shift = self.sft_layers[i](x_map, cond_red)
                sft_scale_means.append(float(scale.detach().float().mean().item()))
                sft_scale_stds.append(float(scale.detach().float().std(unbiased=False).item()))
                sft_shift_means.append(float(shift.detach().float().mean().item()))
                sft_shift_stds.append(float(shift.detach().float().std(unbiased=False).item()))
                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()

            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                **kwargs,
            )
            if i in self.detail_injection_layers:
                level = self._resolve_detail_level(i)
                ref_ctx = ref_feats.get(level, None)
                lr_ctx = lr_feats.get(level, None)
                k = str(i)
                if ref_ctx is not None and k in self.detail_ref_attn:
                    ref_tokens = ref_ctx.flatten(2).transpose(1, 2).to(dtype=x.dtype)
                    ref_out = self.detail_ref_attn[k](x, ref_tokens)
                    ref_gate = torch.sigmoid(self.detail_ref_gate[k]).to(dtype=x.dtype).view(1, 1, 1)
                    x = x + ref_gate * ref_out
                    denom = x.detach().float().norm(dim=-1).mean() + 1e-6
                    ref_res_ratio_vals.append(float((ref_gate.detach().float() * ref_out.detach().float().norm(dim=-1).mean() / denom).item()))
                if lr_ctx is not None and k in self.detail_lr_attn:
                    lr_tokens = lr_ctx.flatten(2).transpose(1, 2).to(dtype=x.dtype)
                    lr_out = self.detail_lr_attn[k](x, lr_tokens)
                    lr_gate = torch.sigmoid(self.detail_lr_gate[k]).to(dtype=x.dtype).view(1, 1, 1)
                    x = x + lr_gate * lr_out
                    denom = x.detach().float().norm(dim=-1).mean() + 1e-6
                    lr_res_ratio_vals.append(float((lr_gate.detach().float() * lr_out.detach().float().norm(dim=-1).mean() / denom).item()))

        if len(sft_scale_means) > 0:
            self._last_sft_stats = {
                "sft_scale_mean": float(sum(sft_scale_means) / len(sft_scale_means)),
                "sft_scale_std": float(sum(sft_scale_stds) / len(sft_scale_stds)),
                "sft_shift_mean": float(sum(sft_shift_means) / len(sft_shift_means)),
                "sft_shift_std": float(sum(sft_shift_stds) / len(sft_shift_stds)),
            }
        else:
            self._last_sft_stats = None
        ref_gates = [torch.sigmoid(v.detach().float()).item() for _, v in self.detail_ref_gate.items()]
        lr_gates = [torch.sigmoid(v.detach().float()).item() for _, v in self.detail_lr_gate.items()]
        self._last_detail_attn_stats = {
            "ref_attn_res_ratio": float(sum(ref_res_ratio_vals) / max(1, len(ref_res_ratio_vals))),
            "lr_attn_res_ratio": float(sum(lr_res_ratio_vals) / max(1, len(lr_res_ratio_vals))),
            "detail_ref_gate_mean": float(sum(ref_gates) / max(1, len(ref_gates))),
            "detail_ref_gate_std": float(torch.tensor(ref_gates).std(unbiased=False).item()) if len(ref_gates) > 1 else 0.0,
            "detail_lr_gate_mean": float(sum(lr_gates) / max(1, len(lr_gates))),
            "detail_lr_gate_std": float(torch.tensor(lr_gates).std(unbiased=False).item()) if len(lr_gates) > 1 else 0.0,
        }

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
