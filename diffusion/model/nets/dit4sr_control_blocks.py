from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, SwiGLU
from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate


class FeedForwardControl(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        self.control_conv = zero_module(nn.Conv2d(inner_dim, inner_dim, 3, stride=1, padding=1, groups=inner_dim))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for i, module in enumerate(self.net):
            hidden_states = module(hidden_states)
            if i == 1:
                hidden_states, hidden_states_control_org = hidden_states.chunk(2, dim=1)
                B, N, C = hidden_states.shape
                h = w = int(np.sqrt(N))
                assert h * w == N
                hidden_states_control = hidden_states_control_org.reshape(B, h, w, C).permute(0, 3, 1, 2)
                hidden_states_control = self.control_conv(hidden_states_control)
                hidden_states_control = hidden_states_control.reshape(B, C, N).permute(0, 2, 1)
                hidden_states = hidden_states + 1.2 * hidden_states_control
                hidden_states = torch.cat([hidden_states, hidden_states_control_org], dim=1)
        return hidden_states


class AttentionZero(Attention):
    def __init__(self,
                 query_dim,
                 cross_attention_dim,
                 added_kv_proj_dim,
                 dim_head,
                 heads,
                 out_dim,
                 context_pre_only,
                 bias,
                 processor,
                 qk_norm,
                 eps):
        super(AttentionZero, self).__init__(
            query_dim=query_dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            dim_head=dim_head,
            heads=heads,
            out_dim=out_dim,
            context_pre_only=context_pre_only,
            bias=bias,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.to_q_control = zero_module(nn.Linear(self.query_dim, self.inner_dim, bias=self.use_bias))
        self.to_k_control = zero_module(nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=self.use_bias))
        self.to_v_control = zero_module(nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=self.use_bias))
        self.to_out_control = nn.Linear(self.inner_dim, self.out_dim, bias=True)
        self.to_out_control.weight.data.copy_(self.to_out[0].weight.data)
        self.to_out_control.bias.data.copy_(self.to_out[0].bias.data)


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class JointAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        hidden_states, hidden_states_control = hidden_states.chunk(2, dim=1)
        hidden_states_control_res = hidden_states_control

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        query_control = attn.to_q_control(attn.to_q(hidden_states_control))
        key_control = attn.to_k_control(attn.to_k(hidden_states_control))
        value_control = attn.to_v_control(attn.to_v(hidden_states_control))

        query_control = query_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_control = key_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_control = value_control.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
            query_control = attn.norm_q(query_control)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            key_control = attn.norm_k(key)

        query = torch.cat([query, query_control], dim=2)
        key = torch.cat([key, key_control], dim=2)
        value = torch.cat([value, value_control], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states, hidden_states_control = hidden_states.chunk(2, dim=1)
        hidden_states_control = hidden_states_control + hidden_states_control_res

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states_control = attn.to_out_control(hidden_states_control)

        hidden_states = torch.cat([hidden_states, hidden_states_control], dim=1)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
