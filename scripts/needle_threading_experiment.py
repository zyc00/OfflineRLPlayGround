#!/usr/bin/env python3
"""Needle threading action-field experiment: MLP vs MIP vs diffusion."""

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


EPS = 1e-8
ACTION_SCALE = 1.0
BARRIER_Y = 0.5
TARGET_X = 0.5
GAP_HALF_WIDTH = 0.01
GAP_LEFT = TARGET_X - GAP_HALF_WIDTH
GAP_RIGHT = TARGET_X + GAP_HALF_WIDTH
TARGET = np.array([0.5, 0.0], dtype=np.float32)
GAP_CENTER = np.array([0.5, 0.5], dtype=np.float32)
CRITICAL_X_MIN = 0.45
CRITICAL_X_MAX = 0.55
CRITICAL_Y_MIN = 0.5
CRITICAL_Y_MAX = 0.6
CRITICAL_X_HALF_WIDTH = 0.05
SLIDE_Y_MAX = 0.55
FULL_DOMAIN = (0.0, 1.0, 0.0, 1.0)
ZOOM_DOMAIN = (0.4, 0.6, 0.45, 0.65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/needle_threading_experiment"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=20)
    parser.add_argument("--diffusion-samples", type=int, default=10)
    parser.add_argument("--full-grid", type=int, default=30)
    parser.add_argument("--zoom-grid", type=int, default=40)
    parser.add_argument("--line-points", type=int, default=400)
    parser.add_argument("--multisample-grid", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--batch-size-cap", type=int, default=256)
    parser.add_argument("--demo-counts", type=int, nargs="+", default=[50, 200, 500, 2000])
    parser.add_argument("--critical-demo-ratio", type=float, default=0.3)
    parser.add_argument("--action-scale", type=float, default=1.0)
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
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def unit_vector_to(point: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = target - point
    norm = np.linalg.norm(delta)
    if norm < EPS:
        return np.zeros(2, dtype=np.float32)
    return (delta / norm).astype(np.float32)


def slide_direction(x: float) -> np.ndarray:
    if x < GAP_LEFT:
        return np.array([ACTION_SCALE, 0.0], dtype=np.float32)
    if x > GAP_RIGHT:
        return np.array([-ACTION_SCALE, 0.0], dtype=np.float32)
    return np.array([0.0, -ACTION_SCALE], dtype=np.float32)


def get_optimal_action(x: float, y: float) -> np.ndarray:
    point = np.array([x, y], dtype=np.float32)
    if abs(y - BARRIER_Y) < 1e-6 and abs(x - TARGET_X) > GAP_HALF_WIDTH:
        return slide_direction(x)
    if y <= BARRIER_Y:
        return ACTION_SCALE * unit_vector_to(point, TARGET)

    in_critical_zone = (
        abs(x - TARGET_X) < CRITICAL_X_HALF_WIDTH
        and BARRIER_Y < y < CRITICAL_Y_MAX
    )
    if in_critical_zone:
        return ACTION_SCALE * unit_vector_to(point, GAP_CENTER)
    if y < SLIDE_Y_MAX and abs(x - TARGET_X) > GAP_HALF_WIDTH:
        return slide_direction(x)
    return ACTION_SCALE * unit_vector_to(point, GAP_CENTER)


def get_optimal_actions(points: np.ndarray) -> np.ndarray:
    return np.stack(
        [get_optimal_action(float(x), float(y)) for x, y in points],
        axis=0,
    )


def angular_error_deg(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_unit = normalize_rows_np(pred)
    target_unit = normalize_rows_np(target)
    dots = np.clip(np.sum(pred_unit * target_unit, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def make_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, nx, dtype=np.float32)
    ys = np.linspace(y_min, y_max, ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)
    return gx, gy, points


def make_cross_section_points(num_points: int) -> np.ndarray:
    xs = np.linspace(0.4, 0.6, num_points, dtype=np.float32)
    ys = np.full_like(xs, 0.52)
    return np.stack([xs, ys], axis=-1)


def sample_demonstrations(n_demo: int, seed: int, critical_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_critical = int(round(n_demo * critical_ratio))
    n_uniform = max(n_demo - n_critical, 0)

    uniform_points = rng.uniform(0.0, 1.0, size=(n_uniform, 2)).astype(np.float32)
    critical_points = np.empty((n_critical, 2), dtype=np.float32)
    critical_points[:, 0] = rng.uniform(CRITICAL_X_MIN, CRITICAL_X_MAX, size=n_critical)
    critical_points[:, 1] = rng.uniform(CRITICAL_Y_MIN, CRITICAL_Y_MAX, size=n_critical)

    points = np.concatenate([uniform_points, critical_points], axis=0)
    rng.shuffle(points)
    actions = get_optimal_actions(points)
    noisy_actions = actions + rng.normal(0.0, 0.01 * ACTION_SCALE, size=actions.shape).astype(np.float32)
    return points, noisy_actions


class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoStepMIP(nn.Module):
    def __init__(self, hidden_dim: int = 256, depth: int = 4):
        super().__init__()
        self.step1 = RegressionMLP(2, hidden_dim=hidden_dim, depth=depth)
        self.step2 = RegressionMLP(4, hidden_dim=hidden_dim, depth=depth)

    def forward(self, states: torch.Tensor, noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        coarse = self.step1(states)
        refine_input = coarse
        if noise_std > 0:
            refine_input = coarse.detach() + noise_std * torch.randn_like(coarse)
        refined = self.step2(torch.cat([states, refine_input], dim=-1))
        return coarse, refined

    @torch.no_grad()
    def predict(self, states: torch.Tensor) -> torch.Tensor:
        coarse = self.step1(states)
        return self.step2(torch.cat([states, coarse], dim=-1))


class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.scale(cond)) + self.shift(cond)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return timesteps[:, None]
    freqs = torch.exp(
        torch.linspace(
            math.log(1.0),
            math.log(1000.0),
            half,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
    )
    angles = timesteps[:, None] * freqs[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DiffusionDenoiser(nn.Module):
    def __init__(self, hidden_dim: int = 256, depth: int = 4, cond_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_dim = time_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        self.in_proj = nn.Linear(2 + time_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(depth)])
        self.film_layers = nn.ModuleList([FiLMLayer(cond_dim, hidden_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, states: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        cond = self.state_encoder(states)
        time_emb = sinusoidal_embedding(timesteps, self.time_dim)
        x = self.in_proj(torch.cat([noisy_actions, time_emb], dim=-1))
        for linear, film in zip(self.hidden_layers, self.film_layers):
            x = linear(x)
            x = film(x, cond)
            x = F.silu(x)
        return self.out_proj(x)


class DiffusionPolicy(nn.Module):
    def __init__(self, num_steps: int = 20, data_scale: float = 1.0):
        super().__init__()
        self.num_steps = num_steps
        self.data_scale = data_scale
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
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alpha_cumprod - 1.0))
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod).clamp_min(EPS)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance", torch.log(posterior_variance.clamp_min(1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod).clamp_min(EPS),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod).clamp_min(EPS),
        )

    def training_loss(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.shape[0]
        t_index = torch.randint(0, self.num_steps, (batch_size,), device=states.device)
        actions = actions / self.data_scale
        noise = torch.randn_like(actions)
        sqrt_alpha = self.sqrt_alpha_cumprod[t_index].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t_index].unsqueeze(-1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise
        eps_pred = self.denoiser(states, noisy_actions, t_index.float())
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, states: torch.Tensor, num_samples: int) -> torch.Tensor:
        batch_size = states.shape[0]
        tiled_states = states.repeat_interleave(num_samples, dim=0)
        x = torch.randn((batch_size * num_samples, 2), device=states.device)
        for t in reversed(range(self.num_steps)):
            t_index = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps_pred = self.denoiser(tiled_states, x, t_index.float())
            x0_pred = (
                self.sqrt_recip_alphas_cumprod[t] * x
                - self.sqrt_recipm1_alphas_cumprod[t] * eps_pred
            )
            x0_pred = x0_pred.clamp(-1.0, 1.0)
            mean = self.posterior_mean_coef1[t] * x0_pred + self.posterior_mean_coef2[t] * x
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.exp(0.5 * self.posterior_log_variance[t]) * noise
            else:
                x = mean
            x = x.clamp(-1.5, 1.5)
        return x.view(batch_size, num_samples, 2) * self.data_scale


@dataclass
class ExperimentResult:
    demo_points: np.ndarray
    demo_actions: np.ndarray
    mlp: RegressionMLP
    mip: TwoStepMIP
    diffusion: DiffusionPolicy


def validate_ground_truth() -> None:
    left = get_optimal_action(0.499, 0.52)
    right = get_optimal_action(0.501, 0.52)
    below = get_optimal_action(0.5, 0.25)
    if left[0] <= 0 or right[0] >= 0:
        raise RuntimeError("Critical-zone lateral sign flip check failed")
    if below[1] >= 0:
        raise RuntimeError("Below-barrier target direction check failed")


def iterate_minibatches(
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    perm = torch.randperm(states.shape[0], device=states.device)
    batches = []
    for start in range(0, states.shape[0], batch_size):
        idx = perm[start:start + batch_size]
        batches.append((states[idx], actions[idx]))
    return batches


def train_mlp(
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    epochs: int,
    print_every: int,
    label: str,
    action_scale: float,
) -> RegressionMLP:
    model = RegressionMLP(2).to(states.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaled_actions = actions / action_scale
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_items = 0
        for batch_states, batch_actions in iterate_minibatches(states, scaled_actions, batch_size):
            optimizer.zero_grad()
            pred = model(batch_states)
            loss = F.mse_loss(pred, batch_actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * batch_states.shape[0]
            n_items += batch_states.shape[0]
        scheduler.step()
        avg_loss = loss_sum / max(n_items, 1)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"[{label}] epoch {epoch:4d}/{epochs} mlp_loss={avg_loss:.6f}", flush=True)
    return model


def train_mip(
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    epochs: int,
    print_every: int,
    label: str,
    action_scale: float,
) -> TwoStepMIP:
    model = TwoStepMIP().to(states.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaled_actions = actions / action_scale
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        coarse_sum = 0.0
        refine_sum = 0.0
        n_items = 0
        for batch_states, batch_actions in iterate_minibatches(states, scaled_actions, batch_size):
            optimizer.zero_grad()
            coarse, refined = model(batch_states, noise_std=0.3)
            coarse_loss = F.mse_loss(coarse, batch_actions)
            refine_loss = F.mse_loss(refined, batch_actions)
            loss = coarse_loss + refine_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * batch_states.shape[0]
            coarse_sum += coarse_loss.item() * batch_states.shape[0]
            refine_sum += refine_loss.item() * batch_states.shape[0]
            n_items += batch_states.shape[0]
        avg_loss = loss_sum / max(n_items, 1)
        avg_coarse = coarse_sum / max(n_items, 1)
        avg_refine = refine_sum / max(n_items, 1)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(
                f"[{label}] epoch {epoch:4d}/{epochs} mip_loss={avg_loss:.6f} "
                f"(step1={avg_coarse:.6f}, step2={avg_refine:.6f})",
                flush=True,
            )
    return model


def train_diffusion(
    states: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    epochs: int,
    print_every: int,
    label: str,
    diffusion_steps: int,
    action_scale: float,
) -> DiffusionPolicy:
    model = DiffusionPolicy(num_steps=diffusion_steps, data_scale=action_scale).to(states.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_items = 0
        for batch_states, batch_actions in iterate_minibatches(states, actions, batch_size):
            optimizer.zero_grad()
            loss = model.training_loss(batch_states, batch_actions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * batch_states.shape[0]
            n_items += batch_states.shape[0]
        avg_loss = loss_sum / max(n_items, 1)
        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"[{label}] epoch {epoch:4d}/{epochs} diff_loss={avg_loss:.6f}", flush=True)
    return model


@torch.no_grad()
def predict_mlp(model: RegressionMLP, points: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).float().to(device)
        preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds, axis=0) * ACTION_SCALE


@torch.no_grad()
def predict_mip(model: TwoStepMIP, points: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    preds = []
    model.eval()
    for start in range(0, len(points), batch_size):
        batch = torch.from_numpy(points[start:start + batch_size]).float().to(device)
        preds.append(model.predict(batch).cpu().numpy())
    return np.concatenate(preds, axis=0) * ACTION_SCALE


@torch.no_grad()
def predict_diffusion_samples(
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
        preds.append(model.sample(batch, num_samples=num_samples).cpu().numpy())
    return np.concatenate(preds, axis=0)


def draw_barrier(ax: plt.Axes) -> None:
    ax.plot([0.0, GAP_LEFT], [BARRIER_Y, BARRIER_Y], color="black", linewidth=3)
    ax.plot([GAP_RIGHT, 1.0], [BARRIER_Y, BARRIER_Y], color="black", linewidth=3)


def style_axis(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    draw_barrier(ax)


def plot_quiver_panel(
    ax: plt.Axes,
    gx: np.ndarray,
    gy: np.ndarray,
    actions: np.ndarray,
    errors: np.ndarray,
    demo_points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    cmap,
    norm,
):
    display = normalize_rows_np(actions)
    q = ax.quiver(
        gx,
        gy,
        display[:, 0].reshape(gx.shape),
        display[:, 1].reshape(gy.shape),
        errors.reshape(gx.shape),
        cmap=cmap,
        norm=norm,
        angles="xy",
        scale_units="xy",
        scale=max(gx.shape[0], gy.shape[0]) * 0.95,
        width=0.004,
        headwidth=4,
    )
    ax.scatter(demo_points[:, 0], demo_points[:, 1], color="black", s=6, alpha=0.6)
    style_axis(ax, xlim, ylim)
    ax.set_title(title, fontsize=9)
    return q


def render_action_field_figure(
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    grid_size: int,
    batch_size: int,
    diffusion_samples: int,
    device: torch.device,
) -> None:
    gx, gy, points = make_grid(*FULL_DOMAIN, grid_size, grid_size)
    gt_actions = get_optimal_actions(points)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0.0, 180.0)
    fig, axes = plt.subplots(4, 4, figsize=(17, 17), constrained_layout=True)
    demo_counts = list(results_by_demo.keys())
    for col, n_demo in enumerate(demo_counts):
        result = results_by_demo[n_demo]
        diffusion_mean = predict_diffusion_samples(
            result.diffusion,
            points,
            device,
            batch_size,
            diffusion_samples,
        ).mean(axis=1)
        predictions = [
            ("Ground Truth", gt_actions),
            ("MLP", predict_mlp(result.mlp, points, device, batch_size)),
            ("MIP", predict_mip(result.mip, points, device, batch_size)),
            ("Diffusion", diffusion_mean),
        ]
        for row, (label, pred) in enumerate(predictions):
            errors = angular_error_deg(pred, gt_actions)
            title = f"{label}\nN={n_demo}, mean err={errors.mean():.1f}°"
            q = plot_quiver_panel(
                axes[row, col],
                gx,
                gy,
                pred,
                errors,
                result.demo_points,
                xlim=(0.0, 1.0),
                ylim=(0.0, 1.0),
                title=title,
                cmap=cmap,
                norm=norm,
            )
    fig.colorbar(q, ax=axes, shrink=0.86, label="Angular error (deg)")
    fig.suptitle("Needle Threading: full action field", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_zoom_figure(
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    grid_size: int,
    batch_size: int,
    diffusion_samples: int,
    device: torch.device,
) -> None:
    gx, gy, points = make_grid(*ZOOM_DOMAIN, grid_size, grid_size)
    gt_actions = get_optimal_actions(points)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0.0, 180.0)
    fig, axes = plt.subplots(4, 4, figsize=(17, 17), constrained_layout=True)
    demo_counts = list(results_by_demo.keys())
    for col, n_demo in enumerate(demo_counts):
        result = results_by_demo[n_demo]
        diffusion_mean = predict_diffusion_samples(
            result.diffusion,
            points,
            device,
            batch_size,
            diffusion_samples,
        ).mean(axis=1)
        predictions = [
            ("Ground Truth", gt_actions),
            ("MLP", predict_mlp(result.mlp, points, device, batch_size)),
            ("MIP", predict_mip(result.mip, points, device, batch_size)),
            ("Diffusion", diffusion_mean),
        ]
        for row, (label, pred) in enumerate(predictions):
            errors = angular_error_deg(pred, gt_actions)
            title = f"{label}\nN={n_demo}, mean err={errors.mean():.1f}°"
            q = plot_quiver_panel(
                axes[row, col],
                gx,
                gy,
                pred,
                errors,
                result.demo_points,
                xlim=(ZOOM_DOMAIN[0], ZOOM_DOMAIN[1]),
                ylim=(ZOOM_DOMAIN[2], ZOOM_DOMAIN[3]),
                title=title,
                cmap=cmap,
                norm=norm,
            )
    fig.colorbar(q, ax=axes, shrink=0.86, label="Angular error (deg)")
    fig.suptitle("Needle Threading: critical-zone zoom", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_cross_section_figure(
    results_by_demo: dict[int, ExperimentResult],
    output_path: Path,
    line_points: int,
    batch_size: int,
    diffusion_samples: int,
    device: torch.device,
) -> None:
    points = make_cross_section_points(line_points)
    gt_actions = get_optimal_actions(points)
    xs = points[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.ravel()
    colors = {
        "MLP": "#1f77b4",
        "MIP": "#ff7f0e",
        "Diffusion": "#2ca02c",
    }
    for ax, (n_demo, result) in zip(axes, results_by_demo.items()):
        curves = {
            "MLP": angular_error_deg(predict_mlp(result.mlp, points, device, batch_size), gt_actions),
            "MIP": angular_error_deg(predict_mip(result.mip, points, device, batch_size), gt_actions),
            "Diffusion": angular_error_deg(
                predict_diffusion_samples(result.diffusion, points, device, batch_size, diffusion_samples).mean(axis=1),
                gt_actions,
            ),
        }
        for label, errors in curves.items():
            ax.plot(xs, errors, color=colors[label], linewidth=2, label=label)
        ax.axvline(GAP_LEFT, color="black", linestyle="--", linewidth=1)
        ax.axvline(TARGET_X, color="black", linestyle=":", linewidth=1)
        ax.axvline(GAP_RIGHT, color="black", linestyle="--", linewidth=1)
        ax.set_xlim(0.4, 0.6)
        ax.set_ylim(0.0, 180.0)
        ax.set_xlabel("x at y=0.52")
        ax.set_ylabel("Angular error (deg)")
        ax.set_title(f"N={n_demo}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Critical-zone cross-section at y=0.52", fontsize=16)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_diffusion_multisample_figure(
    result: ExperimentResult,
    output_path: Path,
    grid_size: int,
    batch_size: int,
    diffusion_samples: int,
    device: torch.device,
) -> None:
    gx, gy, points = make_grid(*ZOOM_DOMAIN, grid_size, grid_size)
    samples = predict_diffusion_samples(result.diffusion, points, device, batch_size, diffusion_samples)
    mean_actions = samples.mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 9), constrained_layout=True)
    for sample_idx in range(diffusion_samples):
        display = normalize_rows_np(samples[:, sample_idx])
        ax.quiver(
            gx,
            gy,
            display[:, 0].reshape(gx.shape),
            display[:, 1].reshape(gy.shape),
            color="gray",
            angles="xy",
            scale_units="xy",
            scale=16,
            width=0.002,
            alpha=0.12,
        )
    mean_display = normalize_rows_np(mean_actions)
    ax.quiver(
        gx,
        gy,
        mean_display[:, 0].reshape(gx.shape),
        mean_display[:, 1].reshape(gy.shape),
        color="crimson",
        angles="xy",
        scale_units="xy",
        scale=16,
        width=0.006,
    )
    visible = (
        (result.demo_points[:, 0] >= ZOOM_DOMAIN[0])
        & (result.demo_points[:, 0] <= ZOOM_DOMAIN[1])
        & (result.demo_points[:, 1] >= ZOOM_DOMAIN[2])
        & (result.demo_points[:, 1] <= ZOOM_DOMAIN[3])
    )
    ax.scatter(result.demo_points[visible, 0], result.demo_points[visible, 1], color="black", s=10, alpha=0.7)
    style_axis(ax, (ZOOM_DOMAIN[0], ZOOM_DOMAIN[1]), (ZOOM_DOMAIN[2], ZOOM_DOMAIN[3]))
    ax.set_title("Diffusion multi-sample view (N=500)\nGray = 10 samples, red = mean", fontsize=13)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    global ACTION_SCALE
    ACTION_SCALE = args.action_scale
    set_seed(args.seed)
    validate_ground_truth()

    print(f"Running needle threading experiment on {device}", flush=True)
    print(f"Output directory: {args.output_dir}", flush=True)
    print(f"Action scale: {args.action_scale}", flush=True)

    results_by_demo: dict[int, ExperimentResult] = {}
    for n_demo in args.demo_counts:
        dataset_seed = args.seed + n_demo * 100
        demo_points, demo_actions = sample_demonstrations(
            n_demo=n_demo,
            seed=dataset_seed,
            critical_ratio=args.critical_demo_ratio,
        )
        states = torch.from_numpy(demo_points).float().to(device)
        actions = torch.from_numpy(demo_actions).float().to(device)
        batch_size = min(n_demo, args.batch_size_cap)

        print(f"\n=== N={n_demo} ===", flush=True)
        set_seed(args.seed + n_demo * 100 + 1)
        mlp = train_mlp(
            states,
            actions,
            batch_size,
            args.epochs,
            args.print_every,
            f"N{n_demo}",
            args.action_scale,
        )
        set_seed(args.seed + n_demo * 100 + 2)
        mip = train_mip(
            states,
            actions,
            batch_size,
            args.epochs,
            args.print_every,
            f"N{n_demo}",
            args.action_scale,
        )
        set_seed(args.seed + n_demo * 100 + 3)
        diffusion = train_diffusion(
            states,
            actions,
            batch_size,
            args.epochs,
            args.print_every,
            f"N{n_demo}",
            args.diffusion_steps,
            args.action_scale,
        )
        results_by_demo[n_demo] = ExperimentResult(
            demo_points=demo_points,
            demo_actions=demo_actions,
            mlp=mlp,
            mip=mip,
            diffusion=diffusion,
        )

    render_action_field_figure(
        results_by_demo,
        args.output_dir / "needle_threading_action_fields.png",
        grid_size=args.full_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        device=device,
    )
    render_zoom_figure(
        results_by_demo,
        args.output_dir / "needle_threading_critical_zoom.png",
        grid_size=args.zoom_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        device=device,
    )
    render_cross_section_figure(
        results_by_demo,
        args.output_dir / "needle_threading_cross_section.png",
        line_points=args.line_points,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        device=device,
    )
    chosen_demo_count = 500 if 500 in results_by_demo else max(args.demo_counts)
    diffusion_multisample_name = f"needle_threading_diffusion_multisample_n{chosen_demo_count}.png"
    result_n500 = results_by_demo[chosen_demo_count]
    render_diffusion_multisample_figure(
        result_n500,
        args.output_dir / diffusion_multisample_name,
        grid_size=args.multisample_grid,
        batch_size=args.eval_batch_size,
        diffusion_samples=args.diffusion_samples,
        device=device,
    )

    print("\nSaved figures:", flush=True)
    for name in [
        "needle_threading_action_fields.png",
        "needle_threading_critical_zoom.png",
        "needle_threading_cross_section.png",
        diffusion_multisample_name,
    ]:
        print(f"  - {args.output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
