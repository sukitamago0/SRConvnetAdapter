import copy

import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer, t2i_modulate
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed


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


@MODELS.register_module()
class PixArtSigmaSR(PixArtMS):
    def __init__(
        self,
        force_null_caption: bool = True,
        hard_injection_layers=None,
        detail_injection_layers=None,
        injection_layer_to_level=None,
        use_dit4sr_core: bool = True,
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
        self.use_dit4sr_core = bool(use_dit4sr_core)

        self.sft_cond_reduce = nn.Sequential(
            nn.Conv2d(self.hidden_size, 64, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.sft_layers = nn.ModuleList([SFTLayer(cond_nc=64, feat_nc=self.hidden_size) for _ in range(self.depth)])
        self.hard_injection_layers = set(hard_injection_layers or [2, 4, 6, 8, 10, 12])
        self.detail_injection_layers = list(detail_injection_layers or [14, 16, 18, 20, 22, 24])
        self.injection_layer_to_level = dict(injection_layer_to_level or {})

        self.lr_x_embedder = copy.deepcopy(self.x_embedder)

        self.detail_lr_norm1 = nn.ModuleDict()
        self.detail_lr_attn = nn.ModuleDict()
        self.detail_lr_cross_attn = nn.ModuleDict()
        self.detail_lr_norm2 = nn.ModuleDict()
        self.detail_lr_mlp = nn.ModuleDict()
        self.detail_lr_scale_shift_table = nn.ParameterDict()
        self.detail_lr_inject_conv = nn.ModuleDict()

        for layer_id in self.detail_injection_layers:
            i = int(layer_id)
            k = str(layer_id)
            self.detail_lr_norm1[k] = copy.deepcopy(self.blocks[i].norm1)
            self.detail_lr_attn[k] = copy.deepcopy(self.blocks[i].attn)
            self.detail_lr_cross_attn[k] = copy.deepcopy(self.blocks[i].cross_attn)
            self.detail_lr_norm2[k] = copy.deepcopy(self.blocks[i].norm2)
            self.detail_lr_mlp[k] = copy.deepcopy(self.blocks[i].mlp)
            self.detail_lr_scale_shift_table[k] = nn.Parameter(self.blocks[i].scale_shift_table.detach().clone())
            self.detail_lr_inject_conv[k] = nn.Conv2d(
                4 * self.hidden_size,
                4 * self.hidden_size,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=4 * self.hidden_size,
                bias=True,
            )
            nn.init.zeros_(self.detail_lr_inject_conv[k].weight)
            nn.init.zeros_(self.detail_lr_inject_conv[k].bias)

        self._last_sft_stats = None
        self._last_detail_attn_stats = None

    @torch.no_grad()
    def init_lr_embedder_from_x_embedder(self):
        self.lr_x_embedder.load_state_dict(self.x_embedder.state_dict(), strict=True)

    @torch.no_grad()
    def init_detail_lr_stream_from_noise_blocks(self):
        for layer_id in self.detail_injection_layers:
            i = int(layer_id)
            k = str(layer_id)
            self.detail_lr_norm1[k].load_state_dict(self.blocks[i].norm1.state_dict(), strict=True)
            self.detail_lr_attn[k].load_state_dict(self.blocks[i].attn.state_dict(), strict=True)
            self.detail_lr_cross_attn[k].load_state_dict(self.blocks[i].cross_attn.state_dict(), strict=True)
            self.detail_lr_norm2[k].load_state_dict(self.blocks[i].norm2.state_dict(), strict=True)
            self.detail_lr_mlp[k].load_state_dict(self.blocks[i].mlp.state_dict(), strict=True)
            self.detail_lr_scale_shift_table[k].copy_(self.blocks[i].scale_shift_table.detach())

    @staticmethod
    def _mlp_post_fc1(mlp: nn.Module, fc1_out: torch.Tensor) -> torch.Tensor:
        out = mlp.act(fc1_out)
        if hasattr(mlp, "drop1"):
            out = mlp.drop1(out)
        elif hasattr(mlp, "drop"):
            out = mlp.drop(out)
        out = mlp.fc2(out)
        if hasattr(mlp, "drop2"):
            out = mlp.drop2(out)
        elif hasattr(mlp, "drop"):
            out = mlp.drop(out)
        return out

    def _forward_detail_block_dit4sr(self, block, layer_idx, x, lr_stream, y, t0, y_lens, HW, mask, **kwargs):
        del y_lens
        b, nx, _ = x.shape
        _, nl, _ = lr_stream.shape
        h, w = HW
        if nx != h * w or nl != h * w:
            raise RuntimeError(f"Detail block token/grid mismatch at layer={layer_idx}: x_tokens={nx}, lr_tokens={nl}, H*W={h*w}")

        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            block.scale_shift_table[None] + t0.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        k = str(layer_idx)
        shift_msa_l, scale_msa_l, gate_msa_l, shift_mlp_l, scale_mlp_l, gate_mlp_l = (
            self.detail_lr_scale_shift_table[k][None] + t0.reshape(b, 6, -1)
        ).chunk(6, dim=1)

        x1 = t2i_modulate(block.norm1(x), shift_msa_x, scale_msa_x)
        l1 = t2i_modulate(self.detail_lr_norm1[k](lr_stream), shift_msa_l, scale_msa_l)

        joint_x = torch.cat([x1, l1], dim=1)
        joint_x_out = block.attn(joint_x, HW=HW)
        x_attn_out = joint_x_out[:, :nx, :]

        joint_l = torch.cat([l1, x1], dim=1)
        joint_l_out = self.detail_lr_attn[k](joint_l, HW=HW)
        l_attn_out = joint_l_out[:, :nl, :]

        x = x + block.drop_path(gate_msa_x * x_attn_out)
        lr_stream = lr_stream + block.drop_path(gate_msa_l * l_attn_out) + l1
        lr_residual_norm = float(l1.detach().float().norm(dim=-1).mean().item())

        x = x + block.cross_attn(x, y, mask)
        lr_stream = lr_stream + self.detail_lr_cross_attn[k](lr_stream, y, mask)

        x2 = t2i_modulate(block.norm2(x), shift_mlp_x, scale_mlp_x)
        l2 = t2i_modulate(self.detail_lr_norm2[k](lr_stream), shift_mlp_l, scale_mlp_l)

        phi_x = block.mlp.fc1(x2)
        eta_l_original = self.detail_lr_mlp[k].fc1(l2)

        eta_2d = eta_l_original.transpose(1, 2).reshape(b, 4 * self.hidden_size, h, w)
        injected_eta_2d = self.detail_lr_inject_conv[k](eta_2d)
        injected_eta_l = injected_eta_2d.reshape(b, 4 * self.hidden_size, nl).transpose(1, 2).contiguous()
        phi_x = phi_x + injected_eta_l

        x_mlp_out = self._mlp_post_fc1(block.mlp, phi_x)
        l_mlp_out = self._mlp_post_fc1(self.detail_lr_mlp[k], eta_l_original)

        x = x + block.drop_path(gate_mlp_x * x_mlp_out)
        lr_stream = lr_stream + block.drop_path(gate_mlp_l * l_mlp_out)

        stats = {
            "lr_stream_norm": float(lr_stream.detach().float().norm(dim=-1).mean().item()),
            "lr_stream_delta": float((l_attn_out.detach().float().norm(dim=-1).mean()).item()),
            "lr_residual_norm": lr_residual_norm,
            "mlp_inject_conv_norm": float(self.detail_lr_inject_conv[k].weight.detach().float().norm().item()),
        }
        return x, lr_stream, stats

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, lr_latent=None, **kwargs):
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
        if self.use_dit4sr_core:
            if lr_latent is None:
                raise ValueError("lr_latent must be provided when use_dit4sr_core=True")
            lr_latent = lr_latent.to(dtype=x.dtype, device=x.device)
            if lr_latent.shape[-2:] != (self.h * self.patch_size, self.w * self.patch_size):
                raise RuntimeError(
                    f"lr_latent spatial size mismatch: got {tuple(lr_latent.shape[-2:])}, expected {(self.h * self.patch_size, self.w * self.patch_size)}"
                )
            lr_stream = self.lr_x_embedder(lr_latent) + pos_embed
            if lr_stream.shape[1] != x.shape[1]:
                raise RuntimeError(f"Patch grid mismatch between noise/lr streams: x_tokens={x.shape[1]}, lr_tokens={lr_stream.shape[1]}")
        else:
            lr_stream = None

        t = self.t_embedder(timestep)
        if aug_level is not None:
            t = t + self.aug_embedder(aug_level.to(self.dtype))

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            t = t + torch.cat([self.csize_embedder(c_size, bs), self.ar_embedder(ar, bs)], dim=1)

        cond_map = None
        if isinstance(adapter_cond, dict):
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
        lr_stream_norm_vals, lr_stream_delta_vals, lr_residual_norm_vals, mlp_inject_conv_norm_vals = [], [], [], []
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

            if self.use_dit4sr_core and (i in self.detail_injection_layers):
                x, lr_stream, detail_stats = self._forward_detail_block_dit4sr(
                    block=block,
                    layer_idx=i,
                    x=x,
                    lr_stream=lr_stream,
                    y=y,
                    t0=t0,
                    y_lens=y_lens,
                    HW=(self.h, self.w),
                    mask=mask,
                    **kwargs,
                )
                lr_stream_norm_vals.append(float(detail_stats["lr_stream_norm"]))
                lr_stream_delta_vals.append(float(detail_stats["lr_stream_delta"]))
                lr_residual_norm_vals.append(float(detail_stats["lr_residual_norm"]))
                mlp_inject_conv_norm_vals.append(float(detail_stats["mlp_inject_conv_norm"]))
            else:
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

        if len(sft_scale_means) > 0:
            self._last_sft_stats = {
                "sft_scale_mean": float(sum(sft_scale_means) / len(sft_scale_means)),
                "sft_scale_std": float(sum(sft_scale_stds) / len(sft_scale_stds)),
                "sft_shift_mean": float(sum(sft_shift_means) / len(sft_shift_means)),
                "sft_shift_std": float(sum(sft_shift_stds) / len(sft_shift_stds)),
            }
        else:
            self._last_sft_stats = None

        self._last_detail_attn_stats = {
            "lr_stream_norm": float(sum(lr_stream_norm_vals) / max(1, len(lr_stream_norm_vals))),
            "lr_stream_delta": float(sum(lr_stream_delta_vals) / max(1, len(lr_stream_delta_vals))),
            "lr_residual_norm": float(sum(lr_residual_norm_vals) / max(1, len(lr_residual_norm_vals))),
            "mlp_inject_conv_norm": float(sum(mlp_inject_conv_norm_vals) / max(1, len(mlp_inject_conv_norm_vals))),
        }

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
