import torch
import torch.nn as nn

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


class TASRTimeGate(nn.Module):
    def __init__(self, x_ch, cond_ch, t_embed_dim, hidden_ch=128, use_scale_shift_norm=True):
        super().__init__()
        self.use_scale_shift_norm = bool(use_scale_shift_norm)
        half = hidden_ch // 2
        self.x_proj = nn.Conv2d(x_ch, half, 1)
        self.delta_proj = nn.Conv2d(x_ch, half, 1)
        self.cond_proj = nn.Conv2d(cond_ch, half, 1)
        c_init = hidden_ch + half
        mid_ch = max(64, c_init // 2)
        self.in_layers = nn.Sequential(
            nn.GroupNorm(16, c_init),
            nn.SiLU(),
            nn.Conv2d(c_init, mid_ch, 3, 1, 1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, 2 * mid_ch if self.use_scale_shift_norm else mid_ch),
        )
        self.out_norm = nn.GroupNorm(16, mid_ch)
        self.out_rest = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(mid_ch, max(32, mid_ch // 2), 3, 1, 1),
            nn.SiLU(),
            zero_module(nn.Conv2d(max(32, mid_ch // 2), 1, 3, 1, 1)),
        )

    def forward(self, x_map, delta_map, cond_map, t_emb):
        x_in = self.x_proj(x_map)
        d_in = self.delta_proj(delta_map)
        c_in = self.cond_proj(cond_map)
        h = torch.cat([x_in, d_in, c_in], dim=1)
        h = self.in_layers(h)
        emb = self.emb_layers(t_emb).type_as(h)
        while emb.dim() < h.dim():
            emb = emb[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb, 2, dim=1)
            h = self.out_norm(h) * (1 + scale) + shift
            h = self.out_rest(h)
        else:
            h = h + emb
            h = self.out_rest(self.out_norm(h))
        return h


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

        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        # anchor_layers = early structure-only band.
        self.sft_candidate_layers = list(anchor_layers) if anchor_layers is not None else list(range(0, 8))
        self.anchor_layers = set(self.sft_candidate_layers)
        # semantic_layers = late semantic-only band.
        self.semantic_layers = list(semantic_layers) if semantic_layers is not None else list(range(24, 28))
        for i in self.semantic_layers:
            if 0 <= i < self.depth:
                self.blocks[i].enable_semantic_adapter()

        default_alpha_init = {i: 1.0 for i in self.sft_candidate_layers}
        self.sft_alpha = nn.ParameterDict({
            str(i): nn.Parameter(torch.tensor(float(default_alpha_init.get(int(i), 1.0))))
            for i in sorted(self.anchor_layers)
        })
        self.tasr_time_gates = nn.ModuleDict({
            str(i): TASRTimeGate(
                x_ch=self.hidden_size,
                cond_ch=64,
                t_embed_dim=self.hidden_size,
                hidden_ch=128,
            )
            for i in sorted(self.anchor_layers)
        })

        self._last_sft_stats = None
        self._last_semantic_stats = None

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
        if adapter_cond is not None:
            cond_map = adapter_cond.get("cond_map", None)

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

        t_norm = timestep.float() / 999.0
        semantic_gate = (1.0 - t_norm).pow(2.0).view(-1, 1, 1).to(x.dtype)
        tasr_gate_means, tasr_gate_mins, tasr_gate_maxs, sft_delta_stds = [], [], [], []
        sem_stats = []
        for i, block in enumerate(self.blocks):
            if cond_red is not None and i in self.anchor_layers:
                b, n, c = x.shape
                if n != self.h * self.w:
                    raise RuntimeError(f"Token grid mismatch: N={n}, expected H*W={self.h * self.w} from input/patch rule")
                x_map_pre = x.transpose(1, 2).reshape(b, c, self.h, self.w)
                x_map_post, _, _ = self.sft_layers[i](x_map_pre, cond_red, strength=sft_strength)
                sft_delta = x_map_post - x_map_pre
                # early LR structure is controlled purely by learned timestep-aware gates;
                # no handcrafted time prior is used.
                gate_logits = self.tasr_time_gates[str(i)](x_map_pre, sft_delta, cond_red, t.to(dtype=x_map_pre.dtype))
                gate_spatial = torch.sigmoid(gate_logits)
                alpha_i = self.sft_alpha[str(i)].to(x_map_pre.dtype).view(1, 1, 1, 1)
                x_map = x_map_pre + alpha_i * gate_spatial * sft_delta
                tasr_gate_means.append(float(gate_spatial.detach().float().mean().item()))
                tasr_gate_mins.append(float(gate_spatial.detach().float().min().item()))
                tasr_gate_maxs.append(float(gate_spatial.detach().float().max().item()))
                sft_delta_stds.append(float(sft_delta.detach().float().std().item()))
                x = x_map.reshape(b, c, n).transpose(1, 2).contiguous()

            block_kwargs = dict(
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                **kwargs,
            )
            if (i in self.semantic_layers) and (semantic_tokens is not None):
                block_kwargs["semantic_tokens"] = semantic_tokens
                block_kwargs["semantic_gate"] = semantic_gate
                block_kwargs["semantic_block_id"] = i
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, **block_kwargs)
            blk_stats = getattr(block, "_last_semantic_stats", None)
            if blk_stats is not None:
                sem_stats.append(blk_stats)

        if len(self.anchor_layers) > 0:
            alpha_vals = [self.sft_alpha[str(i)].item() for i in self.sft_candidate_layers if str(i) in self.sft_alpha]
            self._last_sft_stats = {
                "alpha_mean": float(sum(alpha_vals) / max(1, len(alpha_vals))),
                "alpha_min": float(min(alpha_vals) if len(alpha_vals) > 0 else 0.0),
                "alpha_max": float(max(alpha_vals) if len(alpha_vals) > 0 else 0.0),
                "sft_strength": float(sft_strength),
                "tasr_gate_mean": float(sum(tasr_gate_means) / max(1, len(tasr_gate_means))),
                "tasr_gate_min": float(min(tasr_gate_mins) if len(tasr_gate_mins) > 0 else 0.0),
                "tasr_gate_max": float(max(tasr_gate_maxs) if len(tasr_gate_maxs) > 0 else 0.0),
                "sft_delta_std": float(sum(sft_delta_stds) / max(1, len(sft_delta_stds))),
            }
        else:
            self._last_sft_stats = None

        if len(sem_stats) > 0:
            active_ids = [int(s.get("semantic_block_id", -1)) for s in sem_stats if int(s.get("semantic_block_id", -1)) >= 0]
            nonfinite_ids = [int(s.get("semantic_block_id", -1)) for s in sem_stats if bool(s.get("semantic_nonfinite", False)) and int(s.get("semantic_block_id", -1)) >= 0]
            self._last_semantic_stats = {
                "semantic_out_std": float(sum(s["semantic_out_std"] for s in sem_stats) / len(sem_stats)),
                "semantic_alpha": float(sum(s["semantic_alpha"] for s in sem_stats) / len(sem_stats)),
                "semantic_gate_mean": float(sum(s["semantic_gate_mean"] for s in sem_stats) / len(sem_stats)),
                "hpa_text_out_std": float(sum(float(s.get("hpa_text_out_std", 0.0)) for s in sem_stats) / len(sem_stats)),
                "hpa_img_out_std": float(sum(float(s.get("hpa_img_out_std", 0.0)) for s in sem_stats) / len(sem_stats)),
                "semantic_block_ids_active": sorted(list(set(active_ids))),
                "semantic_block_ids_nonfinite": sorted(list(set(nonfinite_ids))),
            }
            if len(nonfinite_ids) > 0:
                sem_tok_std = float(semantic_tokens.detach().float().std().item()) if semantic_tokens is not None else 0.0
                print(
                    f"[Semantic-NonFinite] timestep_min={float(timestep.float().min().item()):.1f} "
                    f"timestep_max={float(timestep.float().max().item()):.1f} sem_tok_std={sem_tok_std:.5f} "
                    f"sem_out_std={self._last_semantic_stats['semantic_out_std']:.5f} "
                    f"semantic_gate_mean={self._last_semantic_stats['semantic_gate_mean']:.5f} "
                    f"nonfinite_blocks={self._last_semantic_stats['semantic_block_ids_nonfinite']}"
                )
        else:
            self._last_semantic_stats = None

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
