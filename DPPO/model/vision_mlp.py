"""
Vision Diffusion MLP: MLP diffusion policy with ViT image encoder.
Ported from https://github.com/irom-princeton/dppo
"""

import torch
import torch.nn as nn
import einops
from copy import deepcopy

from DPPO.model.mlp import MLP, ResidualMLP, SinusoidalPosEmb
from DPPO.model.modules import SpatialEmb, RandomShiftsAug


class VisionDiffusionMLP(nn.Module):
    """Diffusion MLP with ViT backbone for image observations."""

    def __init__(
        self,
        backbone,
        action_dim,
        horizon_steps,
        cond_dim,
        img_cond_steps=1,
        time_dim=16,
        mlp_dims=[256, 256],
        activation_type="Mish",
        out_activation_type="Identity",
        use_layernorm=False,
        residual_style=False,
        spatial_emb=128,
        visual_feature_dim=128,
        dropout=0,
        num_img=1,
        augment=False,
    ):
        super().__init__()

        # Vision encoder
        self.backbone = backbone
        if augment:
            self.aug = RandomShiftsAug(pad=4)
        self.augment = augment
        self.num_img = num_img
        self.img_cond_steps = img_cond_steps

        if spatial_emb > 0:
            assert spatial_emb > 1
            if num_img > 1:
                self.compress1 = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim, proj_dim=spatial_emb, dropout=dropout,
                )
                self.compress2 = deepcopy(self.compress1)
            else:
                self.compress = SpatialEmb(
                    num_patch=self.backbone.num_patch,
                    patch_dim=self.backbone.patch_repr_dim,
                    prop_dim=cond_dim, proj_dim=spatial_emb, dropout=dropout,
                )
            visual_feature_dim = spatial_emb * num_img
        else:
            self.compress = nn.Sequential(
                nn.Linear(self.backbone.repr_dim, visual_feature_dim),
                nn.LayerNorm(visual_feature_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )

        # Diffusion MLP
        input_dim = time_dim + action_dim * horizon_steps + visual_feature_dim + cond_dim
        output_dim = action_dim * horizon_steps
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        model = ResidualMLP if residual_style else MLP
        self.mlp_mean = model(
            [input_dim] + mlp_dims + [output_dim],
            activation_type=activation_type,
            out_activation_type=out_activation_type,
            use_layernorm=use_layernorm,
        )
        self.time_dim = time_dim

    def forward(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) diffusion step
        cond: dict with keys "state" (B, To, Do) and "rgb" (B, To, C, H, W)
        """
        B, Ta, Da = x.shape
        _, T_rgb, C, H, W = cond["rgb"].shape

        x = x.view(B, -1)
        state = cond["state"].view(B, -1)

        # Take recent images
        rgb = cond["rgb"][:, -self.img_cond_steps:]

        # Concatenate temporal images by channel
        if self.num_img > 1:
            rgb = rgb.reshape(B, T_rgb, self.num_img, 3, H, W)
            rgb = einops.rearrange(rgb, "b t n c h w -> b n (t c) h w")
        else:
            rgb = einops.rearrange(rgb, "b t c h w -> b (t c) h w")

        rgb = rgb.float()

        # Vision encoding
        if self.num_img > 1:
            rgb1, rgb2 = rgb[:, 0], rgb[:, 1]
            if self.augment:
                rgb1 = self.aug(rgb1)
                rgb2 = self.aug(rgb2)
            feat1 = self.compress1(self.backbone(rgb1), state)
            feat2 = self.compress2(self.backbone(rgb2), state)
            feat = torch.cat([feat1, feat2], dim=-1)
        else:
            if self.augment:
                rgb = self.aug(rgb)
            feat = self.backbone(rgb)
            if isinstance(self.compress, SpatialEmb):
                feat = self.compress(feat, state)
            else:
                feat = self.compress(feat.flatten(1, -1))

        cond_encoded = torch.cat([feat, state], dim=-1)

        time = time.view(B, 1)
        time_emb = self.time_embedding(time).view(B, self.time_dim)
        x = torch.cat([x, time_emb, cond_encoded], dim=-1)
        out = self.mlp_mean(x)
        return out.view(B, Ta, Da)
