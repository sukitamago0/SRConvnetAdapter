import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


def _pick_sparse_adapter_blocks(depth: int):
    if int(depth) == 28:
        return [14, 18, 22, 26]
    ratios = [0.50, 0.64, 0.78, 0.93]
    picks = []
    for r in ratios:
        idx = int(round((depth - 1) * r))
        idx = max(0, min(depth - 1, idx))
        picks.append(idx)
    return sorted(list(dict.fromkeys(picks)))


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(self, force_null_caption: bool = True, **kwargs):
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

        self.injection_layers = list(range(self.depth))
        self.input_adapter_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        self.input_adaln = nn.ModuleList([nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True) for _ in range(self.depth)])
        self.input_res_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(self.depth)])
        self.inject_gate = nn.Parameter(torch.full((self.depth,), -4.0))
        # legacy-unused in MSM-DCA forward: kept for compatibility only.

        self.adapter_ca_block_ids = _pick_sparse_adapter_blocks(self.depth)
        self.adapter_ca_norm_q = nn.ModuleDict()
        self.adapter_ca_layers = nn.ModuleDict()
        self.adapter_ca_out = nn.ModuleDict()
        self.adapter_ca_gate = nn.ParameterDict()

        for bid in self.adapter_ca_block_ids:
            bid_key = str(int(bid))
            num_heads = int(getattr(self.blocks[bid].attn, "num_heads", kwargs.get("num_heads", 16)))
            self.adapter_ca_norm_q[bid_key] = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
            self.adapter_ca_layers[bid_key] = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=num_heads,
                batch_first=True,
            )
            self.adapter_ca_out[bid_key] = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
            # IMPORTANT: keep out-proj zero for zero-impact start, but gate must be non-zero
            # to avoid dead-branch gradients (if gate=0 and out=0 simultaneously, gradients vanish).
            self.adapter_ca_gate[bid_key] = nn.Parameter(torch.ones(1))

            nn.init.zeros_(self.adapter_ca_out[bid_key].weight)
            nn.init.zeros_(self.adapter_ca_out[bid_key].bias)

        ca_heads = {k: int(self.adapter_ca_layers[k].num_heads) for k in self.adapter_ca_layers.keys()}
        print(f"[PixArtAdapterCA] depth={self.depth}, block_ids={self.adapter_ca_block_ids}, heads={ca_heads}")
        print("[PixArtLegacyRoute] cond_route_logits=disabled, early/mid/late path=disabled")
        print("[PixArtLegacyBridge] input_adaln/input_res_proj/inject_gate are legacy-unused in MSM-DCA forward")

        for lin in self.input_adaln:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(lin.bias)
        for lin in self.input_res_proj:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(lin.bias)

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

        memory_tokens = None
        if isinstance(adapter_cond, dict):
            memory_tokens = adapter_cond.get("memory_tokens", None)

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

        for i, block in enumerate(self.blocks):
            adaln_shift, adaln_scale, adaln_alpha = None, None, None

            x = auto_grad_checkpoint(
                block,
                x,
                y,
                t0,
                y_lens,
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                adaln_shift=adaln_shift,
                adaln_scale=adaln_scale,
                adaln_alpha=adaln_alpha,
                **kwargs,
            )

            if (memory_tokens is not None) and (i in self.adapter_ca_block_ids):
                bid_key = str(int(i))
                q = self.adapter_ca_norm_q[bid_key](x.to(dtype=torch.float32))
                kv = memory_tokens.to(dtype=torch.float32)
                delta, _ = self.adapter_ca_layers[bid_key](q, kv, kv, need_weights=False)
                delta = self.adapter_ca_out[bid_key](delta).to(dtype=x.dtype)
                gate = self.adapter_ca_gate[bid_key].to(dtype=x.dtype)
                x = x + gate.view(1, 1, 1) * delta

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
