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
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.nets.PixArt_blocks import TimestepEmbedder, T2IFinalLayer
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.nets.guidesr_guidance import GuideSRGuidanceBranch


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

        self.use_guidesr_guidance = True
        self.guidance_channels = 64
        self.guidance_branch = GuideSRGuidanceBranch(channels=self.guidance_channels, num_frb=4)
        self.guide_proj64 = nn.Conv2d(self.guidance_channels * (8 ** 2), 4, 1)
        self.guide_fuse64 = nn.Conv2d(8, 4, 1)
        self.guide_proj32 = nn.Conv2d(self.guidance_channels * (16 ** 2), self.hidden_size, 1)
        self.guide_fuse32 = nn.Conv2d(self.hidden_size * 2, self.hidden_size, 1)
        for m in [self.guide_proj64, self.guide_proj32]:
            nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(m.bias)
        for m in [self.guide_fuse64, self.guide_fuse32]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self._last_guide_r2 = None
        self._last_guide_stats = None

    def forward(self, x, timestep, y, mask=None, data_info=None, adapter_cond=None, force_drop_ids=None, lr_full_up=None, **kwargs):
        aug_level = kwargs.pop("aug_level", None)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        self._last_guide_r2 = None
        self._last_guide_stats = None
        guide_g32 = None
        guide_g64_norm = 0.0
        guide_g32_norm = 0.0
        if self.use_guidesr_guidance and (lr_full_up is not None):
            guide_out = self.guidance_branch(lr_full_up.to(dtype=self.dtype))
            self._last_guide_r2 = guide_out.get("r2", None)
            g64 = guide_out.get("g64", None)
            guide_g32 = guide_out.get("g32", None)
            if g64 is not None:
                guide_g64_norm = float(g64.detach().float().norm().item())
                g64p = self.guide_proj64(g64.to(dtype=x.dtype))
                x = self.guide_fuse64(torch.cat([x, g64p], dim=1)) + x
            if guide_g32 is not None:
                guide_g32_norm = float(guide_g32.detach().float().norm().item())

        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        pos_embed = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation, base_size=self.base_size)
        ).unsqueeze(0).to(x.device).to(self.dtype)

        x = self.x_embedder(x) + pos_embed
        if guide_g32 is not None:
            x_map = x.transpose(1, 2).reshape(bs, self.hidden_size, self.h, self.w)
            g32p = self.guide_proj32(guide_g32.to(dtype=x_map.dtype))
            x_map = self.guide_fuse32(torch.cat([x_map, g32p], dim=1)) + x_map
            x = x_map.reshape(bs, self.hidden_size, self.h * self.w).transpose(1, 2).contiguous()
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

        guide_r2 = self._last_guide_r2
        self._last_guide_stats = {
            "guide_g64_norm": float(guide_g64_norm),
            "guide_g32_norm": float(guide_g32_norm),
            "guide_fuse64_norm": float(self.guide_fuse64.weight.detach().float().norm().item()),
            "guide_fuse32_norm": float(self.guide_fuse32.weight.detach().float().norm().item()),
            "guide_r2_l1": float(torch.mean(torch.abs(guide_r2.detach().float())).item()) if torch.is_tensor(guide_r2) else 0.0,
        }

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x


@MODELS.register_module()
def PixArtSigmaSR_XL_2(**kwargs):
    return PixArtSigmaSR(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
