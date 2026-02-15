"""Compare dataset action advantages against randomly sampled actions.

Uses MC-estimated Q(s, a_random) as a shared baseline to evaluate whether
MC and GAE correctly identify dataset actions as better than random.
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

METHOD_COLORS = {"MC": "C0", "GAE": "C1"}
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


def plot_scatter_vs_random(ax, q_dataset, mean_q_random, method_name):
    """Scatter Q(s, a_dataset) vs mean Q(s, a_random)."""
    ax.scatter(mean_q_random, q_dataset, alpha=0.3, s=10, edgecolors="none",
               color=METHOD_COLORS[method_name])
    lo = min(mean_q_random.min(), q_dataset.min())
    hi = max(mean_q_random.max(), q_dataset.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    ax.set_xlabel("mean Q(s, a_random)")
    ax.set_ylabel(f"{method_name} Q(s, a_dataset)")
    frac_above = (q_dataset > mean_q_random).mean()
    ax.set_title(f"{method_name}: dataset vs random")
    ax.annotate(
        f"P(Q_dataset > Q_random) = {frac_above:.3f}",
        xy=(0.05, 0.92), xycoords="axes fraction", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )


def plot_delta_distribution(ax, deltas, method_name, is_success):
    """Histogram of Q_dataset - mean Q_random, split by outcome."""
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
    ax.set_title(f"{method_name}: Q_dataset \u2212 mean Q_random", fontsize=10)
    ax.set_xlabel("Q_dataset \u2212 mean Q_random")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="upper right")
    return {"ks_stat": ks_stat, "ks_p": ks_p}


def plot_frac_beaten(ax, methods_frac, is_success):
    """Overlaid histograms of per-state fraction of random actions beaten."""
    for name, frac in methods_frac.items():
        ax.hist(frac, bins=30, alpha=0.45, label=name,
                color=METHOD_COLORS[name], density=True,
                edgecolor="white", linewidth=0.3)
    ax.axvline(0.5, color="gray", ls="--", lw=0.8, label="chance")
    ax.set_xlabel("fraction of random actions beaten")
    ax.set_ylabel("density")
    ax.set_title("Per-state fraction beaten")
    ax.legend(fontsize=7)


def plot_timestep_delta(ax, methods_delta, meta, max_t):
    """Mean (Q_dataset - mean Q_random) over timestep, by method and outcome."""
    timesteps = meta["timestep"]
    is_success = meta["is_success"]

    for name, delta in methods_delta.items():
        for outcome, mask_val, color_base in [
            ("succ", True, METHOD_COLORS[name]),
            ("fail", False, None),
        ]:
            means, ts = [], []
            outcome_mask = is_success == mask_val
            for t in range(max_t):
                m = (timesteps == t) & outcome_mask
                if m.sum() < 2:
                    continue
                ts.append(t)
                means.append(delta[m].mean())

            ls = "-" if mask_val else "--"
            c = color_base if mask_val else METHOD_COLORS[name]
            alpha = 1.0 if mask_val else 0.6
            ax.plot(ts, means, label=f"{name} {'succ' if mask_val else 'fail'}",
                    color=c, ls=ls, linewidth=1.5, alpha=alpha)

    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("mean (Q_dataset \u2212 Q_random)")
    ax.set_title("Advantage over random by timestep", fontsize=10)
    ax.legend(fontsize=7)


def plot_timestep_frac_beaten(ax, methods_frac, meta, max_t):
    """Per-timestep fraction of random actions beaten, by method and outcome."""
    timesteps = meta["timestep"]
    is_success = meta["is_success"]

    for name, frac in methods_frac.items():
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
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)


def plot_roc(ax, methods_delta, is_success):
    """ROC from logistic regression on (Q_dataset - mean Q_random) -> success."""
    aucs = {}
    for name, delta in methods_delta.items():
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
    ax.legend(fontsize=8)
    return aucs


def plot_summary_table(ax, metrics):
    ax.axis("off")
    rows = []
    for name in ["MC", "GAE"]:
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
        "--dataset-num-envs", type=int, default=16,
        help="number of parallel envs used during data collection",
    )
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument(
        "--output",
        default="stats/dataset_vs_random.png",
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

    assert "random_action_values" in mc, (
        "MC estimates must contain random_action_values. Re-run mc.py."
    )

    mean_q_random = mc["mean_random_action_values"]  # (N,)
    random_q = mc["random_action_values"]             # (N, K)

    # Compute deltas and frac beaten per method
    methods_q_dataset = {"MC": mc["action_values"], "GAE": gae["action_values"]}
    methods_delta = {}
    methods_frac = {}
    for name, q_dataset in methods_q_dataset.items():
        methods_delta[name] = q_dataset - mean_q_random
        methods_frac[name] = (q_dataset[:, None] > random_q).mean(axis=1)

    for name in ["MC", "GAE"]:
        d = methods_delta[name]
        f = methods_frac[name]
        print(f"  {name}: delta mean={d.mean():.4f}, std={d.std():.4f}, "
              f"frac_beaten mean={f.mean():.3f}")

    # Create figure (3x3)
    fig = plt.figure(figsize=(18, 16), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    all_metrics = {}

    # --- Row 0: Direct comparison ---
    # (0,0) MC scatter
    ax = fig.add_subplot(gs[0, 0])
    plot_scatter_vs_random(ax, mc["action_values"], mean_q_random, "MC")

    # (0,1) GAE scatter
    ax = fig.add_subplot(gs[0, 1])
    plot_scatter_vs_random(ax, gae["action_values"], mean_q_random, "GAE")

    # (0,2) Fraction beaten histogram
    ax = fig.add_subplot(gs[0, 2])
    plot_frac_beaten(ax, methods_frac, meta["is_success"])

    # --- Row 1: By trajectory outcome ---
    # (1,0) MC delta distribution by outcome
    ax = fig.add_subplot(gs[1, 0])
    mc_ks = plot_delta_distribution(ax, methods_delta["MC"], "MC", meta["is_success"])

    # (1,1) GAE delta distribution by outcome
    ax = fig.add_subplot(gs[1, 1])
    gae_ks = plot_delta_distribution(ax, methods_delta["GAE"], "GAE", meta["is_success"])

    # (1,2) Per-timestep delta
    ax = fig.add_subplot(gs[1, 2])
    plot_timestep_delta(ax, methods_delta, meta, max_t)

    # --- Row 2: Deeper analysis ---
    # (2,0) Per-timestep fraction beaten
    ax = fig.add_subplot(gs[2, 0])
    plot_timestep_frac_beaten(ax, methods_frac, meta, max_t)

    # (2,1) ROC
    ax = fig.add_subplot(gs[2, 1])
    aucs = plot_roc(ax, methods_delta, meta["is_success"])

    # (2,2) Summary table
    for name, ks in [("MC", mc_ks), ("GAE", gae_ks)]:
        all_metrics[name] = {
            "mean_delta": methods_delta[name].mean(),
            "frac_above": (methods_q_dataset[name] > mean_q_random).mean(),
            "mean_frac_beaten": methods_frac[name].mean(),
            "ks_stat": ks["ks_stat"],
            "auc": aucs[name],
        }
    ax = fig.add_subplot(gs[2, 2])
    plot_summary_table(ax, all_metrics)

    fig.suptitle(
        "Dataset Action vs Random Actions: MC and GAE",
        fontsize=13, fontweight="bold",
    )

    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
