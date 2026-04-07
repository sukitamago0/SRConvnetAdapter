import torch
import torch.nn as nn
import torch.nn.functional as F


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden = int(dim * growth_rate)
        self.conv_0 = nn.Conv2d(dim, hidden, 1, 1, 0)
        self.conv_1 = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.conv_2 = nn.Conv2d(hidden // 2, dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv_2(x)
        return x


class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden = int(dim * growth_rate)
        p_dim = int(hidden * p_rate)
        self.p_dim = p_dim
        self.hidden_dim = hidden
        self.conv_0 = nn.Conv2d(dim, hidden, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)
        self.conv_2 = nn.Conv2d(hidden, dim, 1, 1, 0)

    def forward(self, x):
        x = F.gelu(self.conv_0(x))
        x1, x2 = x.split([self.p_dim, self.hidden_dim - self.p_dim], dim=1)
        if self.training:
            x1 = self.conv_1(x1)
        else:
            x[:, : self.p_dim, :, :] = self.conv_1(x[:, : self.p_dim, :, :])
            x1, x2 = x.split([self.p_dim, self.hidden_dim - self.p_dim], dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_2(x)
        return x


class SMFA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.belt = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        y, x = self.linear_0(x).chunk(2, dim=1)
        x_s = F.adaptive_max_pool2d(x, (8, 8))
        x_s = self.dw_conv(x_s)
        x_v = torch.var(x, dim=(-2, -1), keepdim=True, unbiased=False)
        x_l = x * F.interpolate(self.linear_1(x_s), size=x.shape[-2:], mode="nearest")
        x_l = x_l * self.alpha + x_v * self.belt
        y_d = self.dw_conv(y)
        y_d = F.gelu(y_d)
        out = self.linear_2(x_l + y_d)
        return out


class FMB(nn.Module):
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.smfa = SMFA(dim)
        self.pcfn = PCFN(dim, growth_rate=ffn_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1) + self.smfa(x)
        x = x + self.pcfn(F.normalize(x, dim=1))
        return x
