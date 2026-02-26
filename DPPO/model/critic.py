"""
Critic networks for DPPO.

Ported from https://github.com/irom-princeton/dppo
"""

from typing import Union
import torch
from DPPO.model.mlp import MLP, ResidualMLP


class CriticObs(torch.nn.Module):
    """State-only critic network V(s)."""

    def __init__(
        self,
        cond_dim,
        mlp_dims,
        activation_type="Mish",
        use_layernorm=False,
        residual_style=False,
    ):
        super().__init__()
        mlp_dims_full = [cond_dim] + mlp_dims + [1]
        if residual_style:
            model = ResidualMLP
        else:
            model = MLP
        self.Q1 = model(
            mlp_dims_full,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, cond: Union[dict, torch.Tensor]):
        """
        cond: dict with key state: (B, To, Do) or tensor (B, D)
        """
        if isinstance(cond, dict):
            B = len(cond["state"])
            state = cond["state"].view(B, -1)
        else:
            state = cond
        return self.Q1(state)
