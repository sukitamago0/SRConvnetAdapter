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


from diffusion.model.builder import MODELS
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer, t2i_modulate
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


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
        self.input_res_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=True) for _ in range(self.depth)])
        self.inject_gate = nn.Parameter(torch.full((self.depth,), -4.0))
        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        self._last_sft_stats = None

        for lin in self.input_res_proj:
            nn.init.normal_(lin.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(lin.bias)

        self.use_lr_mlp_injection = True
        self.lr_inject_on_blocks = list(range(len(self.blocks)))
        self.lr_mlp_fc1 = nn.ModuleList([])
        self.lr_mlp_fc2 = nn.ModuleList([])
        self.lr_inject_dwconv = nn.ModuleList([])
        self.lr_inject_gamma = nn.ParameterList([])
        for blk in self.blocks:
            fc1 = nn.Linear(blk.mlp.fc1.in_features, blk.mlp.fc1.out_features, bias=(blk.mlp.fc1.bias is not None))
            fc2 = nn.Linear(blk.mlp.fc2.in_features, blk.mlp.fc2.out_features, bias=(blk.mlp.fc2.bias is not None))
            fc1.weight.data.copy_(blk.mlp.fc1.weight.data)
            if fc1.bias is not None and blk.mlp.fc1.bias is not None:
                fc1.bias.data.copy_(blk.mlp.fc1.bias.data)
            fc2.weight.data.copy_(blk.mlp.fc2.weight.data)
            if fc2.bias is not None and blk.mlp.fc2.bias is not None:
                fc2.bias.data.copy_(blk.mlp.fc2.bias.data)
            self.lr_mlp_fc1.append(fc1)
            self.lr_mlp_fc2.append(fc2)

            dw = nn.Conv2d(
                blk.mlp.fc1.out_features,
                blk.mlp.fc1.out_features,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=blk.mlp.fc1.out_features,
            )
            nn.init.zeros_(dw.weight)
            if dw.bias is not None:
                nn.init.zeros_(dw.bias)
            self.lr_inject_dwconv.append(dw)
            self.lr_inject_gamma.append(nn.Parameter(torch.tensor(1e-3)))

        self._last_lr_inject_stats = None

    def _forward_block_with_lr_injection(self, block, block_idx, x, lr_tokens, y, t0, y_lens, **kwargs):
        b, _, _ = x.shape
        adaln_shift = kwargs.get("adaln_shift", None)
        adaln_scale = kwargs.get("adaln_scale", None)
        adaln_alpha = kwargs.get("adaln_alpha", None)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.scale_shift_table[None] + t0.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        h = block.norm1(x)
        if adaln_shift is not None and adaln_scale is not None and adaln_alpha is not None:
            h = h * (1.0 + adaln_alpha * adaln_scale.to(h.dtype)) + adaln_alpha * adaln_shift.to(h.dtype)

        h_attn = t2i_modulate(h, shift_msa, scale_msa)
        x = x + block.drop_path(gate_msa * block.attn(h_attn, HW=kwargs.get("HW", None)))
        x = x + block.cross_attn(x, y, y_lens)

        norm_x = t2i_modulate(block.norm2(x), shift_mlp, scale_mlp)
        if self.use_lr_mlp_injection and (lr_tokens is not None) and (block_idx in self.lr_inject_on_blocks):
            phi_x = block.mlp.fc1(norm_x)
            norm_l = block.norm2(lr_tokens)
            eta_l = self.lr_mlp_fc1[block_idx](norm_l)
            eta_map = eta_l.transpose(1, 2).reshape(b, eta_l.shape[-1], self.h, self.w)
            eta_map = self.lr_inject_dwconv[block_idx](eta_map)
            eta_l_conv = eta_map.flatten(2).transpose(1, 2).contiguous()
            phi_x = phi_x + self.lr_inject_gamma[block_idx] * eta_l_conv
            x_mlp_out = block.mlp.fc2(block.mlp.act(phi_x))

            l_mid = block.mlp.act(eta_l)
            l_out = self.lr_mlp_fc2[block_idx](l_mid)
            lr_tokens = lr_tokens + l_out
        else:
            x_mlp_out = block.mlp(norm_x)

        x = x + block.drop_path(gate_mlp * x_mlp_out)
        return x, lr_tokens

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, lr_latent=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self._last_lr_inject_stats = None

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size)
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed
        lr_tokens = None
        if self.use_lr_mlp_injection and (lr_latent is not None):
            lr_tokens = self.x_embedder(lr_latent.to(dtype=x.dtype)) + pos_embed

        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        cond_tokens = None
        cond_map = None
        if isinstance(adapter_cond, dict):
            cond_tokens = adapter_cond.get("cond_tokens", None)
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

        sft_scale_means, sft_scale_stds, sft_shift_means, sft_shift_stds = [], [], [], []
        cond_red = None
        if cond_map is not None:
            cond_red = self.sft_cond_reduce(cond_map.to(dtype=x.dtype))

        for i, block in enumerate(self.blocks):
            if cond_tokens is not None:
                feat = cond_tokens.to(dtype=torch.float32)
                res = self.input_res_proj[i](feat).to(dtype=x.dtype)
                gate = torch.sigmoid(self.inject_gate[i]).to(dtype=x.dtype)
                x = x + gate * res

            if cond_red is not None:
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

            x, lr_tokens = self._forward_block_with_lr_injection(
                block=block,
                block_idx=i,
                x=x,
                lr_tokens=lr_tokens,
                y=y,
                t0=t0,
                y_lens=y_lens,
                HW=(self.h, self.w),
                base_size=self.base_size,
                pe_interpolation=self.pe_interpolation,
                **kwargs,
            )

        if len(sft_scale_means) > 0:
            self._last_sft_stats = {
                "sft_scale_mean": float(sum(sft_scale_means) / len(sft_scale_means)),
                "sft_scale_std": float(sum(sft_scale_stds) / len(sft_scale_stds)),
                "sft_shift_mean": float(sum(sft_shift_means) / len(sft_shift_means)),
                "sft_shift_std": float(sum(sft_shift_stds) / len(sft_shift_stds)),
            }
        else:
            self._last_sft_stats = None

        if self.use_lr_mlp_injection:
            gammas = [float(g.detach().float().mean().item()) for g in self.lr_inject_gamma]
            dw_norms = [float(m.weight.detach().float().norm().item()) for m in self.lr_inject_dwconv]
            mlp_delta = 0.0
            if len(self.lr_mlp_fc1) > 0:
                delta_vals = []
                for i, blk in enumerate(self.blocks):
                    d = (self.lr_mlp_fc1[i].weight.detach().float() - blk.mlp.fc1.weight.detach().float()).norm().item()
                    delta_vals.append(float(d))
                mlp_delta = float(sum(delta_vals) / max(1, len(delta_vals)))
            lr_stream_norm = float(lr_tokens.detach().float().norm().item()) if lr_tokens is not None else 0.0
            self._last_lr_inject_stats = {
                "lr_stream_norm": lr_stream_norm,
                "lr_inject_gamma_mean": float(sum(gammas) / max(1, len(gammas))),
                "lr_inject_gamma_last": float(gammas[-1]) if len(gammas) > 0 else 0.0,
                "lr_inject_dwconv_norm_mean": float(sum(dw_norms) / max(1, len(dw_norms))),
                "lr_inject_dwconv_norm_last": float(dw_norms[-1]) if len(dw_norms) > 0 else 0.0,
                "lr_mlp_delta_norm": mlp_delta,
            }
        else:
            self._last_lr_inject_stats = None

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
