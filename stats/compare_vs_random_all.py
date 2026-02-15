"""Compare dataset action advantages against randomly sampled actions: MC, GAE, IQL.

Uses each method's own random baseline where available. IQL is shown twice:
once against its own Q-network random baseline, once against MC's rollout baseline.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from data.offline_dataset import OfflineRLDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DISPLAY_NAMES = ["MC", "GAE", "IQL", "IQL (MC rand)"]
METHOD_COLORS = {"MC": "C0", "GAE": "C1", "IQL": "C2", "IQL (MC rand)": "C3"}
SUCCESS_COLOR = "#2ecc71"
FAILURE_COLOR = "#e74c3c"


def load_estimates(path):
    data = torch.load(path)
    return {k: v.numpy() for k, v in data.items()}


def build_transition_metadata(trajectories, N):
    traj_id = np.zeros(N, dtype=int)
    timestep = np.zeros(N, dtype=int)
    is_success = np.zeros(N, dtype=bool)
    traj_return = np.zeros(N, dtype=float)

    for i, traj in enumerate(trajectories):
        idx = traj["flat_indices"].numpy()
        success = (traj["rewards"] > 0.5).any().item()
        ret = traj["rewards"].sum().item()
        traj_id[idx] = i
        timestep[idx] = np.arange(len(idx))
        is_success[idx] = success
        traj_return[idx] = ret

    return {
        "traj_id": traj_id,
        "timestep": timestep,
        "is_success": is_success,
        "traj_return": traj_return,
    }


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------


def plot_scatter_vs_random(ax, q_dataset, mean_q_random, name):
    ax.scatter(mean_q_random, q_dataset, alpha=0.3, s=10, edgecolors="none",
               color=METHOD_COLORS[name])
    lo = min(mean_q_random.min(), q_dataset.min())
    hi = max(mean_q_random.max(), q_dataset.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    ax.set_xlabel("mean Q(s, a_random)")
    ax.set_ylabel(f"Q(s, a_dataset)")
    frac_above = (q_dataset > mean_q_random).mean()
    ax.set_title(f"{name}: dataset vs random")
    ax.annotate(
        f"P(Q_dataset > Q_random) = {frac_above:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )


def plot_delta_distribution(ax, deltas, name, is_success):
    succ = deltas[is_success]
    fail = deltas[~is_success]

    lo = min(succ.min(), fail.min()) if len(succ) > 0 and len(fail) > 0 else 0
    hi = max(succ.max(), fail.max()) if len(succ) > 0 and len(fail) > 0 else 1
    bins = np.linspace(lo, hi, 35)

    ax.hist(succ, bins=bins, alpha=0.55, label=f"success (n={len(succ)})",
            density=True, color=SUCCESS_COLOR, edgecolor="white", linewidth=0.3)
    ax.hist(fail, bins=bins, alpha=0.55, label=f"failure (n={len(fail)})",
            density=True, color=FAILURE_COLOR, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="gray", ls="--", lw=0.8)

    ks_stat, ks_p = stats.ks_2samp(succ, fail)
    ax.annotate(
        f"KS = {ks_stat:.3f} (p={ks_p:.2e})\n"
        f"succ mean={succ.mean():.3f}\nfail mean={fail.mean():.3f}",
        xy=(0.03, 0.78), xycoords="axes fraction", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.set_title(f"{name}: Q_dataset \u2212 mean Q_random", fontsize=10)
    ax.set_xlabel("Q_dataset \u2212 mean Q_random")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="upper right")
    return {"ks_stat": ks_stat, "ks_p": ks_p}


def plot_frac_beaten(ax, methods_frac):
    for name in DISPLAY_NAMES:
        frac = methods_frac[name]
        ax.hist(frac, bins=30, alpha=0.35, label=name,
                color=METHOD_COLORS[name], density=True,
                edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("fraction of random actions beaten")
    ax.set_ylabel("density")
    ax.set_title("Per-state fraction beaten")
    ax.legend(fontsize=6)


def plot_timestep_delta(ax, methods_delta, meta, max_t):
    timesteps = meta["timestep"]
    is_success = meta["is_success"]

    for name in DISPLAY_NAMES:
        delta = methods_delta[name]
        for mask_val in [True, False]:
            means, ts = [], []
            outcome_mask = is_success == mask_val
            for t in range(max_t):
                m = (timesteps == t) & outcome_mask
                if m.sum() < 2:
                    continue
                ts.append(t)
                means.append(delta[m].mean())

            ls = "-" if mask_val else "--"
            alpha = 1.0 if mask_val else 0.6
            ax.plot(ts, means,
                    label=f"{name} {'succ' if mask_val else 'fail'}",
                    color=METHOD_COLORS[name], ls=ls, linewidth=1.5, alpha=alpha)

    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean (Q_dataset \u2212 Q_random)")
    ax.set_title("Advantage over random by timestep", fontsize=10)
    ax.legend(fontsize=5, ncol=2)


def plot_timestep_frac_beaten(ax, methods_frac, meta, max_t):
    timesteps = meta["timestep"]
    is_success = meta["is_success"]

    for name in DISPLAY_NAMES:
        frac = methods_frac[name]
        for mask_val in [True, False]:
            means, ts = [], []
            outcome_mask = is_success == mask_val
            for t in range(max_t):
                m = (timesteps == t) & outcome_mask
                if m.sum() < 2:
                    continue
                ts.append(t)
                means.append(frac[m].mean())

            ls = "-" if mask_val else "--"
            alpha = 1.0 if mask_val else 0.6
            ax.plot(ts, means,
                    label=f"{name} {'succ' if mask_val else 'fail'}",
                    color=METHOD_COLORS[name], ls=ls, linewidth=1.5, alpha=alpha)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("timestep")
    ax.set_ylabel("frac beaten")
    ax.set_title("Fraction beaten over timestep", fontsize=10)
    ax.legend(fontsize=5, ncol=2)
    ax.set_ylim(-0.05, 1.05)


def plot_roc(ax, methods_delta, is_success):
    aucs = {}
    for name in DISPLAY_NAMES:
        delta = methods_delta[name]
        X = delta.reshape(-1, 1)
        y = is_success.astype(int)
        clf = LogisticRegression(solver="lbfgs", max_iter=1000)
        clf.fit(X, y)
        y_prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=METHOD_COLORS[name], linewidth=1.5)
        aucs[name] = auc

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC: advantage over random \u2192 success", fontsize=10)
    ax.legend(fontsize=7)
    return aucs


def plot_summary_table(ax, metrics):
    ax.axis("off")
    rows = []
    for name in DISPLAY_NAMES:
        m = metrics[name]
        rows.append([
            name,
            f"{m['mean_delta']:.4f}",
            f"{m['frac_above']:.3f}",
            f"{m['mean_frac_beaten']:.3f}",
            f"{m['ks_stat']:.3f}",
            f"{m['auc']:.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Method", "Mean delta", "P(Q>Q_rand)", "Mean frac beaten",
                    "KS stat", "AUC"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title("Summary metrics", fontsize=10, pad=15)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dataset",
        default="data/datasets/pickcube_expert_eval.pt",
        help="path to eval dataset .pt file",
    )
    parser.add_argument(
        "--mc",
        default="data/datasets/mc_estimates_gamma0.8_iters10.pt",
        help="path to MC estimates (must contain random_action_values)",
    )
    parser.add_argument(
        "--gae",
        default="data/datasets/gae_estimates_gamma0.8_lambda0.95.pt",
        help="path to GAE estimates",
    )
    parser.add_argument(
        "--iql",
        default="data/datasets/iql_estimates_gamma0.8_tau0.7.pt",
        help="path to IQL estimates",
    )
    parser.add_argument(
        "--dataset-num-envs", type=int, default=16,
        help="number of parallel envs used during data collection",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument(
        "--output",
        default="stats/dataset_vs_random_all.png",
        help="save figure to this path",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading eval dataset: {args.eval_dataset}")
    dataset = OfflineRLDataset([args.eval_dataset], False, False)
    N = len(dataset)
    trajectories = dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    n_success = sum(1 for t in trajectories if (t["rewards"] > 0.5).any())
    print(f"  {len(trajectories)} trajectories: {n_success} success, "
          f"{len(trajectories) - n_success} failure")

    meta = build_transition_metadata(trajectories, N)
    max_t = int(meta["timestep"].max()) + 1

    print("Loading estimates...")
    mc = load_estimates(args.mc)
    gae = load_estimates(args.gae)
    iql = load_estimates(args.iql)

    assert "random_action_values" in mc, (
        "MC estimates must contain random_action_values. Re-run mc.py."
    )
    assert "random_action_values" in iql, (
        "IQL estimates must contain random_action_values. Re-run iql.py."
    )

    mc_mean_q_random = mc["mean_random_action_values"]
    mc_random_q = mc["random_action_values"]
    iql_mean_q_random = iql["mean_random_action_values"]
    iql_random_q = iql["random_action_values"]

    # Build per-entry Q_dataset, mean_Q_random, random_Q for all 4 entries
    methods_q_dataset = {
        "MC": mc["action_values"],
        "GAE": gae["action_values"],
        "IQL": iql["action_values"],
        "IQL (MC rand)": iql["action_values"],
    }
    methods_mean_q_random = {
        "MC": mc_mean_q_random,
        "GAE": mc_mean_q_random,       # GAE has no own random — use MC
        "IQL": iql_mean_q_random,       # IQL's own Q-network baseline
        "IQL (MC rand)": mc_mean_q_random,  # MC rollout baseline
    }
    methods_random_q = {
        "MC": mc_random_q,
        "GAE": mc_random_q,
        "IQL": iql_random_q,
        "IQL (MC rand)": mc_random_q,
    }

    methods_delta = {}
    methods_frac = {}
    for name in DISPLAY_NAMES:
        q_dataset = methods_q_dataset[name]
        methods_delta[name] = q_dataset - methods_mean_q_random[name]
        methods_frac[name] = (
            (q_dataset[:, None] > methods_random_q[name]).mean(axis=1)
        )

    for name in DISPLAY_NAMES:
        d = methods_delta[name]
        f = methods_frac[name]
        print(f"  {name}: delta mean={d.mean():.4f}, std={d.std():.4f}, "
              f"frac_beaten mean={f.mean():.3f}")

    # Create figure (4x4)
    fig = plt.figure(figsize=(24, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 4)

    # --- Row 0: Scatter Q_dataset vs mean Q_random ---
    for col, name in enumerate(DISPLAY_NAMES):
        ax = fig.add_subplot(gs[0, col])
        plot_scatter_vs_random(
            ax, methods_q_dataset[name], methods_mean_q_random[name], name
        )

    # --- Row 1: Delta distribution by outcome ---
    all_ks = {}
    for col, name in enumerate(DISPLAY_NAMES):
        ax = fig.add_subplot(gs[1, col])
        all_ks[name] = plot_delta_distribution(
            ax, methods_delta[name], name, meta["is_success"]
        )

    # --- Row 2: Shared temporal plots ---
    ax = fig.add_subplot(gs[2, 0])
    plot_frac_beaten(ax, methods_frac)

    ax = fig.add_subplot(gs[2, 1:3])
    plot_timestep_delta(ax, methods_delta, meta, max_t)

    ax = fig.add_subplot(gs[2, 3])
    plot_timestep_frac_beaten(ax, methods_frac, meta, max_t)

    # --- Row 3: ROC + summary ---
    ax = fig.add_subplot(gs[3, 0])
    aucs = plot_roc(ax, methods_delta, meta["is_success"])

    all_metrics = {}
    for name in DISPLAY_NAMES:
        all_metrics[name] = {
            "mean_delta": methods_delta[name].mean(),
            "frac_above": (
                methods_q_dataset[name] > methods_mean_q_random[name]
            ).mean(),
            "mean_frac_beaten": methods_frac[name].mean(),
            "ks_stat": all_ks[name]["ks_stat"],
            "auc": aucs[name],
        }

    ax = fig.add_subplot(gs[3, 1:])
    plot_summary_table(ax, all_metrics)

    fig.suptitle(
        "Dataset Action vs Random Actions: MC, GAE, IQL, IQL (MC rand)",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
