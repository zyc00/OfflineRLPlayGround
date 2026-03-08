"""
Common modules for vision diffusion models.
Ported from https://github.com/irom-princeton/dppo
"""

import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class SpatialEmb(nn.Module):
    """Compresses ViT patch features into a fixed-size vector using learned weights."""

    def __init__(self, num_patch, patch_dim, prop_dim, proj_dim, dropout):
        super().__init__()
        proj_in_dim = num_patch + prop_dim
        num_proj = patch_dim
        self.patch_dim = patch_dim
        self.prop_dim = prop_dim

        self.input_proj = nn.Sequential(
            nn.Linear(proj_in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )
        self.weight = nn.Parameter(torch.zeros(1, num_proj, proj_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight)

    def forward(self, feat, prop):
        # feat: (B, num_patch, patch_dim) -> (B, patch_dim, num_patch)
        feat = feat.transpose(1, 2)
        if self.prop_dim > 0:
            repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
            feat = torch.cat((feat, repeated_prop), dim=-1)
        y = self.input_proj(feat)
        z = (self.weight * y).sum(1)
        return self.dropout(z)


class RandomShiftsAug:
    """Random spatial shift augmentation for images."""

    def __init__(self, pad=4):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = nn.functional.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return nn.functional.grid_sample(
            x, grid, padding_mode="zeros", align_corners=False
        )


# --- UNet building blocks (flexible versions from DPPO reference) ---

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Activation"""

    def __init__(
        self,
        inp_channels,
        out_channels,
        kernel_size,
        n_groups=None,
        activation_type="Mish",
        eps=1e-5,
    ):
        super().__init__()
        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation_type}")

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            (
                Rearrange("batch channels horizon -> batch channels 1 horizon")
                if n_groups is not None else nn.Identity()
            ),
            (
                nn.GroupNorm(n_groups, out_channels, eps=eps)
                if n_groups is not None else nn.Identity()
            ),
            (
                Rearrange("batch channels 1 horizon -> batch channels horizon")
                if n_groups is not None else nn.Identity()
            ),
            act,
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning for 1D UNet."""

    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=5,
        n_groups=None,
        cond_predict_scale=False,
        larger_encoder=False,
        activation_type="Mish",
        groupnorm_eps=1e-5,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size,
                        n_groups=n_groups, activation_type=activation_type, eps=groupnorm_eps),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=n_groups, activation_type=activation_type, eps=groupnorm_eps),
        ])

        if activation_type == "Mish":
            act = nn.Mish()
        elif activation_type == "ReLU":
            act = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation_type}")

        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels

        if larger_encoder:
            self.cond_encoder = nn.Sequential(
                nn.Linear(cond_dim, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
                act,
                nn.Linear(cond_channels, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            self.cond_encoder = nn.Sequential(
                act,
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0, ...]
            bias = embed[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)
