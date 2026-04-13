import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.nets.adapter import NAFBlock


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


class NAFLocalFidelityFusion(nn.Module):
    def __init__(self, feat_nc=1152, guide_nc=1152, mid_nc=128):
        super().__init__()
        self.feat_proj = nn.Conv2d(feat_nc, mid_nc, 1)
        self.guide_proj = nn.Conv2d(guide_nc, mid_nc, 1)
        self.fuse = nn.Sequential(NAFBlock(mid_nc), NAFBlock(mid_nc))
        self.out_proj = nn.Conv2d(mid_nc, feat_nc, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x_map: torch.Tensor, guide_map: torch.Tensor) -> torch.Tensor:
        h = self.feat_proj(x_map) + self.guide_proj(guide_map)
        h = self.fuse(h)
        return self.out_proj(h)


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(
        self,
        force_null_caption: bool = True,
        anchor_layers=None,
        semantic_layers=None,
        local_fidelity_layers=None,
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
        self.sft_candidate_layers = list(anchor_layers) if anchor_layers is not None else [2, 4, 6, 8]
        self.anchor_layers = set(self.sft_candidate_layers)
        self.semantic_layers = list(semantic_layers) if semantic_layers is not None else [18, 22, 24, 26]
        self.local_fidelity_layers = list(local_fidelity_layers) if local_fidelity_layers is not None else [22, 26]
        for i in self.semantic_layers:
            if 0 <= i < self.depth:
                self.blocks[i].enable_semantic_adapter()
        self.local_fidelity_blocks = nn.ModuleDict({
            str(i): NAFLocalFidelityFusion(feat_nc=self.hidden_size, guide_nc=self.hidden_size, mid_nc=128)
            for i in self.local_fidelity_layers
            if 0 <= i < self.depth
        })
        self.local_fid_alpha = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(1.0))
            for i in self.local_fidelity_layers
            if 0 <= i < self.depth
        })

        default_alpha_init = {2: 1.0, 4: 1.0, 6: 0.5, 8: 0.25}
        self.sft_alpha = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(float(default_alpha_init.get(int(i), 1.0))))
            for i in sorted(self.anchor_layers)
        })

        self._last_sft_stats = None
        self._last_semantic_stats = None
        self._last_local_fidelity_stats = None

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, semantic_tokens=None, **kwargs):
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
        guide_map = None
        if adapter_cond is not None:
            cond_map = adapter_cond.get("cond_map", None)
            guide_map = adapter_cond.get("guide_map", None)

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
        semantic_gate = (1.0 - tau_scalar).view(-1, 1, 1).to(x.dtype)
        late_gate = (1.0 - tau_scalar).view(-1, 1, 1, 1).to(x.dtype)
        sem_stats = []
        local_stats = []
        for i, block in enumerate(self.blocks):
            if cond_red is not None and i in self.anchor_layers:
                b, n, c = x.shape
                if n != self.h * self.w:
                    raise RuntimeError(f"Token grid mismatch: N={n}, expected H*W={self.h * self.w} from input/patch rule")
                x_map_pre = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                x_map_post, _, _ = self.sft_layers[i](x_map_pre, cond_red, strength=sft_strength)

                alpha_i = self.sft_alpha[str(i)].to(x_map_pre.dtype)
                x_map = x_map_pre + (x_map_post - x_map_pre) * (alpha_i * tau_map)
                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()

            # Local fidelity correction:
            # - guide_map is dedicated for late high-resolution fidelity.
            # - cond_map remains dedicated to shallow SFT structure anchors.
            curr_guide = guide_map if guide_map is not None else cond_map
            if curr_guide is not None and str(i) in self.local_fidelity_blocks:
                b, n, c = x.shape
                x_map_pre = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                guide_for_local = curr_guide.to(dtype=x.dtype)
                local_res = self.local_fidelity_blocks[str(i)](x_map_pre, guide_for_local)
                alpha_local = self.local_fid_alpha[str(i)].to(x_map_pre.dtype)
                x_map = x_map_pre + (alpha_local * late_gate) * local_res
                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()
                local_stats.append({
                    "local_res_std": float(local_res.detach().float().std().item()),
                    "local_alpha": float(alpha_local.detach().float().item()),
                    "local_gate_mean": float(late_gate.detach().float().mean().item()),
                })

            block_kwargs = dict(
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                **kwargs,
            )
            if (i in self.semantic_layers) and (semantic_tokens is not None):
                block_kwargs["semantic_tokens"] = semantic_tokens
                block_kwargs["semantic_gate"] = semantic_gate
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, **block_kwargs)
            blk_stats = getattr(block, "_last_semantic_stats", None)
            if blk_stats is not None:
                sem_stats.append(blk_stats)

        if len(self.anchor_layers) > 0:
            alpha_vals = [self.sft_alpha[str(i)].item() for i in self.sft_candidate_layers if str(i) in self.sft_alpha]
            self._last_sft_stats = {
                "tau_mean": float(tau_scalar.mean().item()),
                "tau_min": float(tau_scalar.min().item()),
                "tau_max": float(tau_scalar.max().item()),
                "alpha_mean": float(sum(alpha_vals) / max(1, len(alpha_vals))),
                "alpha_min": float(min(alpha_vals) if len(alpha_vals) > 0 else 0.0),
                "alpha_max": float(max(alpha_vals) if len(alpha_vals) > 0 else 0.0),
                "sft_strength": float(sft_strength),
            }
        else:
            self._last_sft_stats = None

        if len(sem_stats) > 0:
            self._last_semantic_stats = {
                "semantic_out_std": float(sum(s["semantic_out_std"] for s in sem_stats) / len(sem_stats)),
                "semantic_alpha": float(sum(s["semantic_alpha"] for s in sem_stats) / len(sem_stats)),
                "semantic_gate_mean": float(sum(s["semantic_gate_mean"] for s in sem_stats) / len(sem_stats)),
            }
        else:
            self._last_semantic_stats = None

        if len(local_stats) > 0:
            self._last_local_fidelity_stats = {
                "local_res_std": float(sum(s["local_res_std"] for s in local_stats) / len(local_stats)),
                "local_alpha": float(sum(s["local_alpha"] for s in local_stats) / len(local_stats)),
                "local_gate_mean": float(sum(s["local_gate_mean"] for s in local_stats) / len(local_stats)),
            }
        else:
            self._last_local_fidelity_stats = None

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
