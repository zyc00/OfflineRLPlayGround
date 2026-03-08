"""
Vision UNet1D: 1D UNet diffusion policy with ViT image encoder.
Ported from https://github.com/irom-princeton/dppo
"""

import torch
import torch.nn as nn
import einops
from copy import deepcopy

from DPPO.model.mlp import ResidualMLP
from DPPO.model.modules import (
    SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock,
    ResidualBlock1D, SpatialEmb, RandomShiftsAug,
)


class VisionUnet1D(nn.Module):
    """1D UNet with ViT vision backbone for image-conditioned diffusion."""

    def __init__(
        self,
        backbone,
        action_dim,
        img_cond_steps=1,
        cond_dim=None,
        diffusion_step_embed_dim=32,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        smaller_encoder=False,
        cond_mlp_dims=None,
        kernel_size=5,
        n_groups=None,
        activation_type="Mish",
        cond_predict_scale=False,
        groupnorm_eps=1e-5,
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

        # UNet
        dims = [action_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))

        dsed = diffusion_step_embed_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        if cond_mlp_dims is not None:
            self.cond_mlp = ResidualMLP(
                dim_list=[cond_dim] + cond_mlp_dims,
                activation_type=activation_type,
                out_activation_type="Identity",
            )
            cond_block_dim = dsed + cond_mlp_dims[-1] + visual_feature_dim
        else:
            cond_block_dim = dsed + cond_dim + visual_feature_dim
        use_large_encoder = cond_mlp_dims is None and not smaller_encoder

        mid_dim = dims[-1]
        self.mid_modules = nn.ModuleList([
            ResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_block_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder,
                            activation_type=activation_type, groupnorm_eps=groupnorm_eps),
            ResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_block_dim,
                            kernel_size=kernel_size, n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                            larger_encoder=use_large_encoder,
                            activation_type=activation_type, groupnorm_eps=groupnorm_eps),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ResidualBlock1D(dim_in, dim_out, cond_dim=cond_block_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale,
                                larger_encoder=use_large_encoder,
                                activation_type=activation_type, groupnorm_eps=groupnorm_eps),
                ResidualBlock1D(dim_out, dim_out, cond_dim=cond_block_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale,
                                larger_encoder=use_large_encoder,
                                activation_type=activation_type, groupnorm_eps=groupnorm_eps),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_block_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale,
                                larger_encoder=use_large_encoder,
                                activation_type=activation_type, groupnorm_eps=groupnorm_eps),
                ResidualBlock1D(dim_in, dim_in, cond_dim=cond_block_dim,
                                kernel_size=kernel_size, n_groups=n_groups,
                                cond_predict_scale=cond_predict_scale,
                                larger_encoder=use_large_encoder,
                                activation_type=activation_type, groupnorm_eps=groupnorm_eps),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size, n_groups=n_groups,
                        activation_type=activation_type, eps=groupnorm_eps),
            nn.Conv1d(dim, action_dim, 1),
        )

    def forward(self, x, time, cond, **kwargs):
        """
        x: (B, Ta, Da)
        time: (B,) diffusion step
        cond: dict with keys "state" (B, To, Do) and "rgb" (B, To, C, H, W)
        """
        B = len(x)
        _, T_rgb, C, H, W = cond["rgb"].shape

        x = einops.rearrange(x, "b h t -> b t h")
        state = cond["state"].view(B, -1)

        if hasattr(self, "cond_mlp"):
            state = self.cond_mlp(state)

        # Take recent images
        rgb = cond["rgb"][:, -self.img_cond_steps:]

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

        # Time embedding
        if not torch.is_tensor(time):
            time = torch.tensor([time], dtype=torch.long, device=x.device)
        elif torch.is_tensor(time) and len(time.shape) == 0:
            time = time[None].to(x.device)
        time = time.expand(x.shape[0])
        global_feature = self.time_mlp(time)
        global_feature = torch.cat([global_feature, cond_encoded], dim=-1)

        # UNet forward
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
