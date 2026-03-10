#!/usr/bin/env python3
"""Toy maze experiment comparing MLP, MIP, and diffusion action fields."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EXIT_POINT = np.array([0.0, 1.0], dtype=np.float32)
GAP_CENTER = np.array([0.5, 0.15], dtype=np.float32)
WALL_X = 0.5
WALL_Y_MIN = 0.3
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/maze_action_field_experiment"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--print-every", type=int, default=250)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diffusion-samples", type=int, default=5)
    parser.add_argument(
        "--diffusion-aggregate",
        type=str,
        default="medoid",
        choices=["mean", "medoid"],
    )
    parser.add_argument("--quiver-grid", type=int, default=20)
    parser.add_argument("--heatmap-grid", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--demo-counts", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument(
        "--scenario2-mode",
        type=str,
        default="aliased_band",
        choices=["hard_piecewise", "aliased_band"],
    )
    parser.add_argument("--alias-bandwidth", type=float, default=0.08)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_rows_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return x / norms


def normalize_rows_torch(x: torch.Tensor) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return x / norms


def unit_vector_to(point: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - point
    norm = np.linalg.norm(delta)
    if norm < EPS:
        return np.zeros(2, dtype=np.float32)
    return (delta / norm).astype(np.float32)


def get_optimal_action(x: float, y: float, scenario: int) -> np.ndarray:
    point = np.array([x, y], dtype=np.float32)
    if scenario == 1:
        return unit_vector_to(point, EXIT_POINT)
    if x < WALL_X:
        return unit_vector_to(point, EXIT_POINT)
    if y >= WALL_Y_MIN:
        return np.array([0.0, -1.0], dtype=np.float32)
    return np.array([-1.0, 0.0], dtype=np.float32)


def get_optimal_actions(points: np.ndarray, scenario: int) -> np.ndarray:
    actions = np.stack(
        [get_optimal_action(float(x), float(y), scenario) for x, y in points],
        axis=0,
    )
    return normalize_rows_np(actions)


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_unit = normalize_rows_np(pred)
    target_unit = normalize_rows_np(target)
    dots = np.sum(pred_unit * target_unit, axis=-1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def make_grid(num_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(0.05, 0.95, num_points, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    return gx, gy, points


def make_heatmap_grid(num_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    return gx, gy, points


def sample_demonstrations(scenario: int, n_demo: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, size=(n_demo, 2)).astype(np.float32)
    actions = get_optimal_actions(points, scenario)
    noisy_actions = actions + rng.normal(0.0, 0.02, size=actions.shape).astype(np.float32)
    noisy_actions = normalize_rows_np(noisy_actions)
    return points, noisy_actions


def observe_points(
    points: np.ndarray,
    scenario: int,
    scenario2_mode: str,
    alias_bandwidth: float,
) -> np.ndarray:
    observed = points.copy()
    if scenario != 2 or scenario2_mode == "hard_piecewise":
        return observed
    mask = (np.abs(observed[:, 0] - WALL_X) <= alias_bandwidth) & (observed[:, 1] >= WALL_Y_MIN)
    observed[mask, 0] = WALL_X
    return observed


def aggregate_diffusion_samples(samples: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return samples.mean(axis=1)
    if mode != "medoid":
        raise ValueError(f"Unknown diffusion aggregate mode: {mode}")
    diffs = samples[:, :, None, :] - samples[:, None, :, :]
    dists = np.sum(diffs * diffs, axis=-1)
    medoid_idx = np.argmin(dists.sum(axis=-1), axis=-1)
    row_idx = np.arange(samples.shape[0])
    return samples[row_idx, medoid_idx]


def diffusion_spread_deg(samples: np.ndarray) -> np.ndarray:
    unit = normalize_rows_np(samples.reshape(-1, 2)).reshape(samples.shape)
    resultant = np.linalg.norm(unit.mean(axis=1), axis=-1)
    resultant = np.clip(resultant, 0.0, 1.0)
    return np.degrees(np.arccos(resultant))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.gelu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class FiLMResidualBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = F.silu(self.linear1(h))
        h = self.linear2(h)
        return x + h


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=values.device,
            dtype=values.dtype,
        )
    )
    angles = values.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class RegressionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(states))


class ToyMIP(nn.Module):
    def __init__(self, hidden_dim: int = 256, depth: int = 4, phase_dim: int = 32):
        super().__init__()
        self.phase_dim = phase_dim
        self.input_proj = nn.Linear(2 + 2 + phase_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, states: torch.Tensor, current_action: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        phase_emb = sinusoidal_embedding(phase, self.phase_dim)
        x = torch.cat([states, current_action, phase_emb], dim=-1)
        x = F.gelu(self.input_proj(x))
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.out_proj(self.out_norm(x)))


class DiffusionDenoiser(nn.Module):
    def __init__(self, hidden_dim: int = 256, depth: int = 4, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.input_proj = nn.Linear(2 + 128, hidden_dim)
        self.blocks = nn.ModuleList([FiLMResidualBlock(hidden_dim, 128) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, states: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        state_cond = self.state_encoder(states)
        time_emb = sinusoidal_embedding(timesteps, self.time_dim)
        time_cond = self.time_encoder(time_emb)
        x = torch.cat([noisy_actions, time_cond], dim=-1)
        x = F.silu(self.input_proj(x))
        for block in self.blocks:
            x = block(x, state_cond)
        return torch.tanh(self.out_proj(self.out_norm(x)))


class DiffusionPolicy(nn.Module):
    def __init__(self, num_steps: int):
        super().__init__()
        self.num_steps = num_steps
        self.denoiser = DiffusionDenoiser()
        betas = torch.linspace(1e-4, 2e-2, num_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1), alpha_cumprod[:-1]], dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod).clamp_min(EPS),
        )

    def training_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        t_index = torch.randint(0, self.num_steps, (batch_size,), device=states.device)
        noise = torch.randn_like(actions)
        sqrt_alpha = self.sqrt_alpha_cumprod[t_index].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t_index].unsqueeze(-1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise
        t_norm = t_index.float() / max(self.num_steps - 1, 1)
        pred_clean = self.denoiser(states, noisy_actions, t_norm)
        return F.mse_loss(pred_clean, actions)

    @torch.no_grad()
    def sample(self, states: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        device = states.device
        batch_size = states.shape[0]
        tiled_states = states.repeat_interleave(num_samples, dim=0)
        x = torch.randn((batch_size * num_samples, 2), device=device)
        for t in reversed(range(self.num_steps)):
            t_index = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
            t_norm = t_index.float() / max(self.num_steps - 1, 1)
            pred_clean = self.denoiser(tiled_states, x, t_norm)
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_cumprod[t]
            alpha_bar_prev = self.alpha_cumprod_prev[t]
            beta_t = self.betas[t]
            coef1 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t).clamp_min(EPS)
            coef2 = (1.0 - alpha_bar_prev) * torch.sqrt(alpha_t) / (1.0 - alpha_bar_t).clamp_min(EPS)
            mean = coef1 * pred_clean + coef2 * x
            if t > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t].clamp_min(1e-6)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
            x = x.clamp(-1.5, 1.5)
        return x.view(batch_size, num_samples, 2)


@dataclass
class ExperimentResult:
    demo_points: np.ndarray
    demo_observed_points: np.ndarray
    demo_actions: np.ndarray
    mlp: RegressionMLP
    mip: ToyMIP
    diffusion: DiffusionPolicy


def check_ground_truth() -> None:
    left = get_optimal_action(0.49, 0.8, scenario=2)
    right = get_optimal_action(0.51, 0.8, scenario=2)
    if np.linalg.norm(left) < 0.99 or np.linalg.norm(right) < 0.99:
        raise RuntimeError("Ground-truth action magnitude check failed")
    if np.dot(left, right) > 0.5:
        raise RuntimeError("Scenario 2 discontinuity check failed")


def train_mlp(
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    epochs: int,
    print_every: int,
    label: str,
) -> RegressionMLP:
    model = RegressionMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = model(states)
        loss = F.mse_loss(pred, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"[{label}] epoch {epoch:4d}/{epochs} mlp_loss={loss.item():.6f}", flush=True)
    return model


def train_mip(
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    epochs: int,
    print_every: int,
    label: str,
) -> ToyMIP:
    model = ToyMIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    t_star = 0.7
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        phase_zero = torch.zeros(states.shape[0], device=device)
        phase_refine = torch.full((states.shape[0],), t_star, device=device)
        pred0 = model(states, torch.zeros_like(actions), phase_zero)
        noise = 0.3 * torch.randn_like(actions)
        pred1 = model(states, actions + noise, phase_refine)
        loss0 = F.mse_loss(pred0, actions)
        loss1 = F.mse_loss(pred1, actions)
        loss = loss0 + loss1
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"[{label}] epoch {epoch:4d}/{epochs} mip_loss={loss.item():.6f} "
                f"(pass1={loss0.item():.6f}, pass2={loss1.item():.6f})",
                flush=True,
            )
    return model


def train_diffusion(
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    epochs: int,
    print_every: int,
    label: str,
    diffusion_steps: int,
) -> DiffusionPolicy:
    model = DiffusionPolicy(num_steps=diffusion_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = model.training_loss(states, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"[{label}] epoch {epoch:4d}/{epochs} diff_loss={loss.item():.6f}", flush=True)
    return model


@torch.no_grad()
def predict_mlp(model: RegressionMLP, points: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).float().to(device)
        pred = model(batch).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_mip(model: ToyMIP, points: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).float().to(device)
        phase_zero = torch.zeros(batch.shape[0], device=device)
        phase_refine = torch.full((batch.shape[0],), 0.7, device=device)
        step1 = model(batch, torch.zeros((batch.shape[0], 2), device=device), phase_zero)
        step2 = model(batch, step1, phase_refine)
        preds.append(step2.cpu().numpy())
    return np.concatenate(preds, axis=0)


@torch.no_grad()
def predict_diffusion(
    model: DiffusionPolicy,
    points: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_samples: int,
) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).float().to(device)
        samples = model.sample(batch, num_samples=num_samples)
        preds.append(samples.cpu().numpy())
    return np.concatenate(preds, axis=0)


def draw_wall(ax: plt.Axes) -> None:
    ax.plot([WALL_X, WALL_X], [WALL_Y_MIN, 1.0], color="black", linewidth=3)


def style_axis(ax: plt.Axes, scenario: int) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    if scenario == 2:
        draw_wall(ax)


def plot_quiver_panel(
    ax: plt.Axes,
    gx: np.ndarray,
    gy: np.ndarray,
    actions: np.ndarray,
    errors: np.ndarray,
    demo_points: np.ndarray,
    scenario: int,
    title: str,
    cmap,
    norm,
) -> None:
    display_actions = normalize_rows_np(actions)
    q = ax.quiver(
        gx,
        gy,
        display_actions[:, 0].reshape(gx.shape),
        display_actions[:, 1].reshape(gy.shape),
        errors.reshape(gx.shape),
        cmap=cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=20,
        width=0.007,
        headwidth=4,
    )
    ax.scatter(demo_points[:, 0], demo_points[:, 1], color="black", s=12, alpha=0.8)
    style_axis(ax, scenario)
    ax.set_title(title, fontsize=9)
    return q


def render_action_field_figure(
    scenario: int,
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    quiver_grid: int,
    batch_size: int,
    diffusion_samples: int,
    diffusion_aggregate: str,
    scenario2_mode: str,
    alias_bandwidth: float,
    device: torch.device,
) -> None:
    gx, gy, grid_points = make_grid(quiver_grid)
    gt_actions = get_optimal_actions(grid_points, scenario)
    observed_grid_points = observe_points(grid_points, scenario, scenario2_mode, alias_bandwidth)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0.0, 180.0)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    demo_counts = list(results_by_demo.keys())
    for col, n_demo in enumerate(demo_counts):
        result = results_by_demo[n_demo]
        row_actions = [
            ("Ground Truth", gt_actions),
            ("MLP", predict_mlp(result.mlp, observed_grid_points, device, batch_size)),
            ("MIP", predict_mip(result.mip, observed_grid_points, device, batch_size)),
            (
                f"Diffusion ({diffusion_aggregate})",
                aggregate_diffusion_samples(
                    predict_diffusion(
                    result.diffusion,
                    observed_grid_points,
                    device,
                    batch_size,
                    diffusion_samples,
                    ),
                    diffusion_aggregate,
                ),
            ),
        ]
        for row, (label, pred_actions) in enumerate(row_actions):
            errors = angular_error_deg(pred_actions, gt_actions)
            mean_error = float(errors.mean())
            title = f"{label}\nN={n_demo}, mean err={mean_error:.1f}°"
            q = plot_quiver_panel(
                axes[row, col],
                gx,
                gy,
                pred_actions,
                errors,
                result.demo_points,
                scenario,
                title,
                cmap,
                norm,
            )
    fig.colorbar(q, ax=axes, shrink=0.82, label="Angular error (deg)")
    fig.suptitle(
        f"Scenario {scenario}: action fields"
        + (" (continuous)" if scenario == 1 else " (wall discontinuity)"),
        fontsize=16,
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_heatmap_figure(
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    heatmap_grid: int,
    batch_size: int,
    diffusion_samples: int,
    diffusion_aggregate: str,
    scenario2_mode: str,
    alias_bandwidth: float,
    device: torch.device,
) -> None:
    gx, gy, grid_points = make_heatmap_grid(heatmap_grid)
    gt_actions = get_optimal_actions(grid_points, scenario=2)
    observed_grid_points = observe_points(grid_points, scenario=2, scenario2_mode=scenario2_mode, alias_bandwidth=alias_bandwidth)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0.0, 180.0)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    demo_counts = list(results_by_demo.keys())
    model_specs = [
        ("MLP", lambda r: predict_mlp(r.mlp, observed_grid_points, device, batch_size)),
        ("MIP", lambda r: predict_mip(r.mip, observed_grid_points, device, batch_size)),
        (
            f"Diffusion ({diffusion_aggregate})",
            lambda r: aggregate_diffusion_samples(
                predict_diffusion(
                    r.diffusion,
                    observed_grid_points,
                    device,
                    batch_size,
                    diffusion_samples,
                ),
                diffusion_aggregate,
            ),
        ),
    ]
    for col, n_demo in enumerate(demo_counts):
        result = results_by_demo[n_demo]
        for row, (label, predictor) in enumerate(model_specs):
            pred_actions = predictor(result)
            errors = angular_error_deg(pred_actions, gt_actions).reshape(gx.shape)
            im = axes[row, col].imshow(
                errors,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap=cmap,
                norm=norm,
                aspect="equal",
            )
            axes[row, col].scatter(result.demo_points[:, 0], result.demo_points[:, 1], color="black", s=8, alpha=0.55)
            style_axis(axes[row, col], scenario=2)
            axes[row, col].set_title(f"{label}\nN={n_demo}, mean err={errors.mean():.1f}°", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.86, label="Angular error (deg)")
    fig.suptitle("Scenario 2: angular error heatmaps", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_diffusion_spread_figure(
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    heatmap_grid: int,
    batch_size: int,
    diffusion_samples: int,
    scenario2_mode: str,
    alias_bandwidth: float,
    device: torch.device,
) -> None:
    gx, gy, grid_points = make_heatmap_grid(heatmap_grid)
    observed_grid_points = observe_points(grid_points, scenario=2, scenario2_mode=scenario2_mode, alias_bandwidth=alias_bandwidth)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(0.0, 90.0)
    demo_counts = list(results_by_demo.keys())
    for col, n_demo in enumerate(demo_counts):
        result = results_by_demo[n_demo]
        samples = predict_diffusion(
            result.diffusion,
            observed_grid_points,
            device,
            batch_size,
            diffusion_samples,
        )
        spread = diffusion_spread_deg(samples).reshape(gx.shape)
        im = axes[col].imshow(
            spread,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        axes[col].scatter(result.demo_points[:, 0], result.demo_points[:, 1], color="white", s=8, alpha=0.55)
        style_axis(axes[col], scenario=2)
        axes[col].set_title(f"Diffusion spread\nN={n_demo}, mean={spread.mean():.1f}°", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.9, label="Sample spread (deg)")
    fig.suptitle("Scenario 2: diffusion sample spread", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def scenario2_filename(mode: str, stem: str) -> str:
    if mode == "hard_piecewise":
        return stem
    return stem.replace("scenario2_", f"scenario2_{mode}_", 1)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    check_ground_truth()

    print(f"Running maze action-field experiment on {device}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(
        f"Scenario 2 mode: {args.scenario2_mode}, alias_bandwidth={args.alias_bandwidth}, "
        f"diffusion_aggregate={args.diffusion_aggregate}",
        flush=True,
    )

    all_results: dict[int, dict[int, ExperimentResult]] = {1: {}, 2: {}}

    for scenario in [1, 2]:
        for n_demo in args.demo_counts:
            dataset_seed = args.seed + scenario * 1000 + n_demo
            demo_points, demo_actions = sample_demonstrations(scenario, n_demo, dataset_seed)
            observed_points = observe_points(
                demo_points,
                scenario=scenario,
                scenario2_mode=args.scenario2_mode,
                alias_bandwidth=args.alias_bandwidth,
            )
            states = torch.from_numpy(observed_points).float().to(device)
            actions = torch.from_numpy(demo_actions).float().to(device)

            print(f"\n=== Scenario {scenario}, N={n_demo} ===", flush=True)

            set_seed(args.seed + scenario * 1000 + n_demo * 10 + 1)
            mlp = train_mlp(states, actions, device, args.epochs, args.print_every, f"S{scenario}-N{n_demo}")

            set_seed(args.seed + scenario * 1000 + n_demo * 10 + 2)
            mip = train_mip(states, actions, device, args.epochs, args.print_every, f"S{scenario}-N{n_demo}")

            set_seed(args.seed + scenario * 1000 + n_demo * 10 + 3)
            diffusion = train_diffusion(
                states,
                actions,
                device,
                args.epochs,
                args.print_every,
                f"S{scenario}-N{n_demo}",
                args.diffusion_steps,
            )

            all_results[scenario][n_demo] = ExperimentResult(
                demo_points=demo_points,
                demo_observed_points=observed_points,
                demo_actions=demo_actions,
                mlp=mlp,
                mip=mip,
                diffusion=diffusion,
            )

    render_action_field_figure(
        scenario=1,
        results_by_demo=all_results[1],
        output_path=args.output_dir / "scenario1_action_fields.png",
        quiver_grid=args.quiver_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        diffusion_aggregate=args.diffusion_aggregate,
        scenario2_mode=args.scenario2_mode,
        alias_bandwidth=args.alias_bandwidth,
        device=device,
    )
    render_action_field_figure(
        scenario=2,
        results_by_demo=all_results[2],
        output_path=args.output_dir / scenario2_filename(args.scenario2_mode, "scenario2_action_fields.png"),
        quiver_grid=args.quiver_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        diffusion_aggregate=args.diffusion_aggregate,
        scenario2_mode=args.scenario2_mode,
        alias_bandwidth=args.alias_bandwidth,
        device=device,
    )
    render_heatmap_figure(
        results_by_demo=all_results[2],
        output_path=args.output_dir / scenario2_filename(args.scenario2_mode, "scenario2_error_heatmaps.png"),
        heatmap_grid=args.heatmap_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        diffusion_aggregate=args.diffusion_aggregate,
        scenario2_mode=args.scenario2_mode,
        alias_bandwidth=args.alias_bandwidth,
        device=device,
    )
    render_diffusion_spread_figure(
        results_by_demo=all_results[2],
        output_path=args.output_dir / scenario2_filename(args.scenario2_mode, "scenario2_diffusion_spread.png"),
        heatmap_grid=args.heatmap_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        scenario2_mode=args.scenario2_mode,
        alias_bandwidth=args.alias_bandwidth,
        device=device,
    )

    print("\nSaved figures:", flush=True)
    for name in [
        "scenario1_action_fields.png",
        scenario2_filename(args.scenario2_mode, "scenario2_action_fields.png"),
        scenario2_filename(args.scenario2_mode, "scenario2_error_heatmaps.png"),
        scenario2_filename(args.scenario2_mode, "scenario2_diffusion_spread.png"),
    ]:
        print(f"  - {args.output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
