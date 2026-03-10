import torch
import torch.nn as nn
from ..common import MLP

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dims, activation, sigma_init=1.0,
                 cond_steps=1, horizon_steps=1):
        super().__init__()
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.mlp = MLP(input_dim * cond_steps, action_dim * horizon_steps, hidden_dims, activation)
        # Learn log std as a separate parameter (state-independent)
        self.log_std = nn.Parameter(torch.ones(action_dim) * sigma_init)

    def forward(self, x):
        # x: (B, cond_steps, obs_dim) or (B, obs_dim) when cond_steps=1
        B = x.shape[0]
        if x.dim() == 3:
            x = x.reshape(B, -1)
        out = self.mlp(x)
        if self.horizon_steps > 1:
            mean = out.reshape(B, self.horizon_steps, self.action_dim)
        else:
            mean = out  # (B, action_dim) — backward compat
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
