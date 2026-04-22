import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModelWithProjection


# Migrated from IP-Adapter official resampler implementation:
# https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, n, _ = latents.shape
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = q.reshape(b, n, h, -1).transpose(1, 2)
        k = k.reshape(b, kv_input.shape[1], h, -1).transpose(1, 2)
        v = v.reshape(b, kv_input.shape[1], h, -1).transpose(1, 2)
        weight = (q * self.scale) @ (k * self.scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(self, dim=1024, depth=8, dim_head=64, heads=16, num_queries=8, embedding_dim=768, output_dim=1024, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads), FeedForward(dim=dim, mult=ff_mult)]))

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class CLIPSemanticAdapter(nn.Module):
    def __init__(self, encoder_name_or_path: str, hidden_size: int = 1152, num_prompt_tokens: int = 16, clip_input_res: int = 224):
        super().__init__()
        self.clip_input_res = int(clip_input_res)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            encoder_name_or_path,
            local_files_only=True,
        )
        self.image_encoder.eval()
        for p in self.image_encoder.parameters():
            p.requires_grad_(False)

        vision_dim = int(self.image_encoder.config.hidden_size)
        self.resampler = Resampler(
            dim=vision_dim,
            depth=4,
            dim_head=64,
            heads=max(1, vision_dim // 64),
            num_queries=int(num_prompt_tokens),
            embedding_dim=vision_dim,
            output_dim=vision_dim,
            ff_mult=4,
        )
        self.proj_norm = nn.LayerNorm(vision_dim)
        self.proj = nn.Linear(vision_dim, int(hidden_size))
        nn.init.normal_(self.proj.weight, std=1e-3)
        nn.init.constant_(self.proj.bias, 0.0)
        self.out_scale = nn.Parameter(torch.tensor(1.0))

        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1), persistent=False)
        self._last_sem_adapter_stats = None

    def forward(self, lr_img_01: torch.Tensor):
        x = F.interpolate(lr_img_01, size=(self.clip_input_res, self.clip_input_res), mode="bicubic", align_corners=False)
        x = (x - self.clip_mean.to(device=x.device, dtype=x.dtype)) / self.clip_std.to(device=x.device, dtype=x.dtype)
        vision_out = self.image_encoder(pixel_values=x, output_hidden_states=True)
        patch_tokens = vision_out.hidden_states[-1][:, 1:, :]
        sem_tokens = self.resampler(patch_tokens)
        preproj_std = float(sem_tokens.detach().float().std().item())
        sem_tokens = self.proj_norm(sem_tokens)
        sem_tokens = self.proj(sem_tokens)
        sem_tokens = torch.clamp(self.out_scale.to(dtype=sem_tokens.dtype) * sem_tokens, -6.0, 6.0)
        postproj_std = float(sem_tokens.detach().float().std().item())
        if not torch.isfinite(sem_tokens).all():
            print(
                f"[SemanticAdapter-NonFinite] sem_tokens_preproj_std={preproj_std:.6f} "
                f"sem_tokens_postproj_std={postproj_std:.6f} sem_out_scale={float(self.out_scale.detach().float().item()):.6f}"
            )
        self._last_sem_adapter_stats = {
            "sem_tokens_preproj_std": preproj_std,
            "sem_tokens_postproj_std": postproj_std,
            "sem_out_scale": float(self.out_scale.detach().float().item()),
        }
        return sem_tokens
