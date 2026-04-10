import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
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


class KVInjectAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # 保持“零影响启动”，但不要让整条分支完全无梯度
        self.gamma = nn.Parameter(torch.tensor(1e-3))

        # K/V 投影不能全 0，否则 gamma/out_proj 都拿不到有效梯度
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)

        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)

        # out_proj 保持极小初始化，保证初期扰动很小，但不是严格死零
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, q_tokens: torch.Tensor, cond_tokens: torch.Tensor):
        b, n, c = q_tokens.shape
        m = cond_tokens.shape[1]

        q = q_tokens.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(cond_tokens).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(cond_tokens).view(b, m, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(b, n, c)
        attn = self.out_proj(attn)
        return self.gamma.to(attn.dtype) * attn


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

        # keep hard-layer SFT family
        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        # legacy arg kept for checkpoint/config compatibility; active SR control uses anchor_layers + kv_inject_layers
        _ = hard_injection_layers
        self.sft_candidate_layers = [2, 4, 6, 8]
        self.anchor_layers = set(self.sft_candidate_layers)
        self.kv_inject_layers = set([10, 14, 18, 22, 24, 26])
        self.kv_inject = nn.ModuleDict({
            str(i): KVInjectAttention(hidden_size=self.hidden_size, num_heads=self.num_heads)
            for i in sorted(self.kv_inject_layers)
        })
        self.sft_alpha = nn.ParameterDict({
            "2": nn.Parameter(torch.tensor(1.0)),
            "4": nn.Parameter(torch.tensor(1.0)),
            "6": nn.Parameter(torch.tensor(0.5)),
            "8": nn.Parameter(torch.tensor(0.25)),
        })

        self._last_sft_stats = None
        self._last_kv_stats = None
        self._last_kv_raw_stats = {"raw_mean": 0.0, "raw_std": 0.0}

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        _ = kwargs.pop("sft_strength", 1.0)
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

        cond_red = None
        if cond_map is not None:
            cond_red = self.sft_cond_reduce(cond_map.to(dtype=x.dtype))

        t_norm = timestep.float() / 1000.0
        tau_scalar = t_norm.pow(1.5)
        tau_map = tau_scalar.view(-1, 1, 1, 1).to(x.dtype)
        tau_seq = tau_scalar.view(-1, 1, 1).to(x.dtype)

        for i, block in enumerate(self.blocks):
            if cond_red is not None and i in self.anchor_layers:
                b, n, c = x.shape
                if n != self.h * self.w:
                    raise RuntimeError(f"Token grid mismatch: N={n}, expected H*W={self.h * self.w} from input/patch rule")
                x_map_pre = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                x_map_post, _, _ = self.sft_layers[i](x_map_pre, cond_red, strength=1.0)

                alpha_i = self.sft_alpha[str(i)].to(x_map_pre.dtype)
                x_map = x_map_pre + (x_map_post - x_map_pre) * (alpha_i * tau_map)

                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()

            # detail layers revert to original block-only forward
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
            if cond_tokens is not None and i in self.kv_inject_layers:
                kv_module = self.kv_inject[str(i)]

                # 手动展开一份 raw 输出，便于确认分支不是死的
                b2, n2, c2 = x.shape
                m2 = cond_tokens.shape[1]
                cond_tokens_cast = cond_tokens.to(dtype=x.dtype)

                q = x.view(b2, n2, kv_module.num_heads, kv_module.head_dim).transpose(1, 2)
                k = kv_module.k_proj(cond_tokens_cast).view(b2, m2, kv_module.num_heads, kv_module.head_dim).transpose(1, 2)
                v = kv_module.v_proj(cond_tokens_cast).view(b2, m2, kv_module.num_heads, kv_module.head_dim).transpose(1, 2)

                raw_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
                raw_attn = raw_attn.transpose(1, 2).contiguous().view(b2, n2, c2)
                raw_attn = kv_module.out_proj(raw_attn)

                self._last_kv_raw_stats = {
                    "raw_mean": float(raw_attn.mean().item()),
                    "raw_std": float(raw_attn.std().item()),
                }

                kv_out = kv_module.gamma.to(raw_attn.dtype) * raw_attn
                x = x + kv_out * (1.0 - tau_seq)

        if len(self.anchor_layers) > 0:
            alpha_vals = [self.sft_alpha[str(i)].item() for i in self.sft_candidate_layers]
            self._last_sft_stats = {
                "tau_mean": float(tau_scalar.mean().item()),
                "tau_min": float(tau_scalar.min().item()),
                "tau_max": float(tau_scalar.max().item()),
                "alpha_mean": float(sum(alpha_vals) / len(alpha_vals)),
                "alpha_min": float(min(alpha_vals)),
                "alpha_max": float(max(alpha_vals)),
            }
        else:
            self._last_sft_stats = None
        if len(self.kv_inject) > 0:
            gammas = [self.kv_inject[str(i)].gamma.item() for i in sorted(self.kv_inject_layers)]
            gamma_mean = float(sum(gammas) / len(gammas))
            raw_stats = getattr(self, "_last_kv_raw_stats", {"raw_mean": 0.0, "raw_std": 0.0})
            self._last_kv_stats = {
                "gamma_mean": gamma_mean,
                "gamma_abs_mean": float(sum(abs(g) for g in gammas) / len(gammas)),
                "gamma_values": [float(g) for g in gammas],
                "gamma_min": float(min(gammas)),
                "gamma_max": float(max(gammas)),
                "eff_mean": float((1.0 - tau_scalar.mean().item()) * gamma_mean),
                "raw_mean": float(raw_stats.get("raw_mean", 0.0)),
                "raw_std": float(raw_stats.get("raw_std", 0.0)),
            }
        else:
            self._last_kv_stats = {
                "gamma_mean": 0.0,
                "gamma_abs_mean": 0.0,
                "gamma_values": [],
                "gamma_min": 0.0,
                "gamma_max": 0.0,
                "eff_mean": 0.0,
                "raw_mean": 0.0,
                "raw_std": 0.0,
            }

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
