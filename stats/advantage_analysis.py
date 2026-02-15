"""Comprehensive advantage estimation analysis: MC vs GAE vs IQL.

Produces a 3x3 figure analyzing signal quality, rank consistency,
and predictive power of each advantage method under sparse rewards.
"""

import argparse
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from data.offline_dataset import OfflineRLDataset


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_estimates(path: str) -> dict[str, np.ndarray]:
    data = torch.load(path)
    return {k: v.numpy() for k, v in data.items()}


def build_transition_metadata(
    trajectories: list[dict], N: int
) -> dict[str, np.ndarray]:
    """Build per-transition metadata from extracted trajectories.

    Returns arrays indexed by flat position in the dataset:
        traj_id, timestep, is_success, traj_return
    """
    traj_id = np.zeros(N, dtype=int)
    timestep = np.zeros(N, dtype=int)
    is_success = np.zeros(N, dtype=bool)
    traj_return = np.zeros(N, dtype=float)

    for i, traj in enumerate(trajectories):
        idx = traj["flat_indices"].numpy()
        traj_len = len(idx)
        success = (traj["rewards"] > 0.5).any().item()
        ret = traj["rewards"].sum().item()

        traj_id[idx] = i
        timestep[idx] = np.arange(traj_len)
        is_success[idx] = success
        traj_return[idx] = ret

    return {
        "traj_id": traj_id,
        "timestep": timestep,
        "is_success": is_success,
        "traj_return": traj_return,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((nx - 1) * x.std(ddof=1) ** 2 + (ny - 1) * y.std(ddof=1) ** 2)
        / (nx + ny - 2)
    )
    return (x.mean() - y.mean()) / max(pooled_std, 1e-8)


def discrimination_score(success_adv: np.ndarray, failure_adv: np.ndarray) -> float:
    """Fraction of (success, failure) pairs where success advantage > failure."""
    if len(success_adv) == 0 or len(failure_adv) == 0:
        return float("nan")
    count = 0
    total = len(success_adv) * len(failure_adv)
    for s in success_adv:
        count += (s > failure_adv).sum()
    return count / total


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

METHOD_COLORS = {"MC": "C0", "GAE": "C1", "IQL": "C2"}
SUCCESS_COLOR = "#2ecc71"
FAILURE_COLOR = "#e74c3c"


def plot_advantage_distribution(
    ax, advantages: np.ndarray, is_success: np.ndarray, method_name: str
):
    """Overlaid histograms of A(s,a) for success vs failure trajectories."""
    succ = advantages[is_success]
    fail = advantages[~is_success]

    lo = min(succ.min(), fail.min()) if len(succ) > 0 and len(fail) > 0 else 0
    hi = max(succ.max(), fail.max()) if len(succ) > 0 and len(fail) > 0 else 1
    bins = np.linspace(lo, hi, 35)

    ax.hist(succ, bins=bins, alpha=0.55, label=f"success (n={len(succ)})",
            density=True, color=SUCCESS_COLOR, edgecolor="white", linewidth=0.3)
    ax.hist(fail, bins=bins, alpha=0.55, label=f"failure (n={len(fail)})",
            density=True, color=FAILURE_COLOR, edgecolor="white", linewidth=0.3)

    # Annotate separation metrics
    ks_stat, ks_p = stats.ks_2samp(succ, fail)
    d = cohens_d(succ, fail)
    ax.annotate(
        f"KS = {ks_stat:.3f} (p={ks_p:.2e})\nCohen's d = {d:.2f}",
        xy=(0.03, 0.88), xycoords="axes fraction", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.set_title(f"{method_name}: A(s,a) by outcome", fontsize=10)
    ax.set_xlabel("A(s,a)")
    ax.set_ylabel("density")
    ax.legend(fontsize=7, loc="upper right")

    return {"ks_stat": ks_stat, "ks_p": ks_p, "cohens_d": d}


def plot_advantage_magnitude(ax, methods: dict, meta: dict, max_t: int):
    """Mean |A(s,a)| vs timestep for each method."""
    timesteps = meta["timestep"]

    for name, adv in methods.items():
        means, stderrs, ts = [], [], []
        for t in range(max_t):
            mask = timesteps == t
            if mask.sum() < 2:
                continue
            vals = np.abs(adv[mask])
            ts.append(t)
            means.append(vals.mean())
            stderrs.append(vals.std() / np.sqrt(len(vals)))

        ts, means, stderrs = np.array(ts), np.array(means), np.array(stderrs)
        ax.plot(ts, means, label=name, color=METHOD_COLORS[name], linewidth=1.5)
        ax.fill_between(ts, means - stderrs, means + stderrs,
                        alpha=0.15, color=METHOD_COLORS[name])

    ax.set_xlabel("timestep")
    ax.set_ylabel("mean |A(s,a)|")
    ax.set_title("Advantage magnitude over timestep", fontsize=10)
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", ls=":", lw=0.5)


def plot_discrimination(ax, methods: dict, meta: dict, max_t: int):
    """Action discrimination score per timestep."""
    timesteps = meta["timestep"]
    is_success = meta["is_success"]

    for name, adv in methods.items():
        scores, ts = [], []
        for t in range(max_t):
            mask = timesteps == t
            succ_adv = adv[mask & is_success]
            fail_adv = adv[mask & ~is_success]
            if len(succ_adv) == 0 or len(fail_adv) == 0:
                continue
            ts.append(t)
            scores.append(discrimination_score(succ_adv, fail_adv))

        ax.plot(ts, scores, label=name, color=METHOD_COLORS[name],
                linewidth=1.5, marker=".", markersize=3)

    ax.axhline(0.5, color="gray", ls="--", lw=1, label="chance")
    ax.set_xlabel("timestep")
    ax.set_ylabel("P(success A > failure A)")
    ax.set_title("Action discrimination score", fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(-0.05, 1.05)


def plot_rank_correlation(ax, methods: dict, meta: dict):
    """Pairwise Spearman rho by timestep bin (early/mid/late)."""
    timesteps = meta["timestep"]
    bins = [("early\n(0-16)", (0, 17)), ("mid\n(17-33)", (17, 34)),
            ("late\n(34-50)", (34, 51))]
    pairs = [("MC", "GAE"), ("MC", "IQL"), ("GAE", "IQL")]
    pair_colors = ["C3", "C4", "C5"]

    x = np.arange(len(bins))
    width = 0.25
    all_rhos = {}

    for j, (pair_name, pair) in enumerate(zip(
        ["MC-GAE", "MC-IQL", "GAE-IQL"], pairs
    )):
        rhos = []
        for bin_label, (lo, hi) in bins:
            mask = (timesteps >= lo) & (timesteps < hi)
            n = mask.sum()
            if n < 3:
                rhos.append(float("nan"))
                continue
            rho, _ = stats.spearmanr(methods[pair[0]][mask], methods[pair[1]][mask])
            rhos.append(rho)
        all_rhos[pair_name] = rhos
        bars = ax.bar(x + j * width, rhos, width, label=pair_name,
                      color=pair_colors[j], alpha=0.8)
        for bar, rho in zip(bars, rhos):
            if not np.isnan(rho):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{rho:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x + width)
    ax.set_xticklabels([b[0] for b in bins], fontsize=8)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Rank correlation by timestep", fontsize=10)
    ax.legend(fontsize=7)
    ax.axhline(0, color="gray", ls=":", lw=0.5)

    return all_rhos


def plot_roc_curves(ax, methods: dict, is_success: np.ndarray):
    """ROC curves from logistic regression on A(s,a) -> trajectory success."""
    aucs = {}
    for name, adv in methods.items():
        X = adv.reshape(-1, 1)
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
    ax.set_title("ROC: A(s,a) predicting success", fontsize=10)
    ax.legend(fontsize=8)

    return aucs


def plot_return_correlation(ax, methods: dict, meta: dict, max_t: int):
    """Per-timestep Pearson correlation between A(s,a) and trajectory return."""
    timesteps = meta["timestep"]
    traj_return = meta["traj_return"]

    for name, adv in methods.items():
        corrs, ts = [], []
        for t in range(max_t):
            mask = timesteps == t
            if mask.sum() < 3:
                continue
            ret = traj_return[mask]
            # Skip if all returns identical (correlation undefined)
            if ret.std() < 1e-8:
                continue
            r, _ = stats.pearsonr(adv[mask], ret)
            ts.append(t)
            corrs.append(r)

        ax.plot(ts, corrs, label=name, color=METHOD_COLORS[name],
                linewidth=1.5, marker=".", markersize=3)

    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("timestep")
    ax.set_ylabel("Pearson r(A, return)")
    ax.set_title("Per-timestep return correlation", fontsize=10)
    ax.legend(fontsize=8)


def plot_summary_table(ax, all_metrics: dict):
    """Summary statistics table."""
    ax.axis("off")
    rows = []
    for name in ["MC", "GAE", "IQL"]:
        m = all_metrics[name]
        rows.append([
            name,
            f"{m['ks_stat']:.3f}",
            f"{m['cohens_d']:.2f}",
            f"{m['auc']:.3f}",
            f"{m['mean_disc']:.3f}",
            f"{m['mean_rho']:.3f}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["Method", "KS stat", "Cohen's d", "AUC",
                    "Disc. score", "Mean rho"],
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
        help="path to MC estimates",
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
    parser.add_argument("--output", default=None, help="save figure to this path")
    parser.add_argument("--csv-output", default=None, help="save summary CSV")
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    print(f"Loading eval dataset: {args.eval_dataset}")
    dataset = OfflineRLDataset([args.eval_dataset], False, False)
    N = len(dataset)
    trajectories = dataset.extract_trajectories(
        num_envs=args.dataset_num_envs, gamma=args.gamma
    )
    n_success = sum(1 for t in trajectories if (t["rewards"] > 0.5).any())
    n_fail = len(trajectories) - n_success
    print(f"  {len(trajectories)} trajectories: {n_success} success, {n_fail} failure")

    meta = build_transition_metadata(trajectories, N)
    max_t = int(meta["timestep"].max()) + 1

    print("Loading estimates...")
    mc_est = load_estimates(args.mc)
    gae_est = load_estimates(args.gae)
    iql_est = load_estimates(args.iql)

    # Build method advantage dict
    methods = {
        "MC": mc_est["advantages"],
        "GAE": gae_est["advantages"],
        "IQL": iql_est["advantages"],
    }
    for name, adv in methods.items():
        assert len(adv) == N, f"{name} has {len(adv)} entries, expected {N}"
        print(f"  {name}: mean={adv.mean():.4f}, std={adv.std():.4f}")

    # ---------------------------------------------------------------
    # 2. Create figure
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    all_metrics = {}

    # Row 0: Advantage distributions by outcome
    for col, name in enumerate(["MC", "GAE", "IQL"]):
        ax = fig.add_subplot(gs[0, col])
        m = plot_advantage_distribution(ax, methods[name], meta["is_success"], name)
        all_metrics[name] = m

    # Row 1, Col 0: Advantage magnitude over timestep
    ax = fig.add_subplot(gs[1, 0])
    plot_advantage_magnitude(ax, methods, meta, max_t)

    # Row 1, Col 1: Action discrimination score
    ax = fig.add_subplot(gs[1, 1])
    plot_discrimination(ax, methods, meta, max_t)

    # Compute mean discrimination score per method
    for name, adv in methods.items():
        scores = []
        for t in range(max_t):
            mask = meta["timestep"] == t
            succ_adv = adv[mask & meta["is_success"]]
            fail_adv = adv[mask & ~meta["is_success"]]
            if len(succ_adv) > 0 and len(fail_adv) > 0:
                scores.append(discrimination_score(succ_adv, fail_adv))
        all_metrics[name]["mean_disc"] = np.nanmean(scores) if scores else float("nan")

    # Row 1, Col 2: Pairwise rank correlation by timestep bin
    ax = fig.add_subplot(gs[1, 2])
    rho_dict = plot_rank_correlation(ax, methods, meta)

    # Compute mean rho per method (average over pairs involving that method)
    for name in ["MC", "GAE", "IQL"]:
        rhos = []
        for pair_name, pair_rhos in rho_dict.items():
            if name in pair_name.split("-"):
                rhos.extend([r for r in pair_rhos if not np.isnan(r)])
        all_metrics[name]["mean_rho"] = np.mean(rhos) if rhos else float("nan")

    # Row 2, Col 0: ROC curves
    ax = fig.add_subplot(gs[2, 0])
    aucs = plot_roc_curves(ax, methods, meta["is_success"])
    for name in ["MC", "GAE", "IQL"]:
        all_metrics[name]["auc"] = aucs[name]

    # Row 2, Col 1: Per-timestep return correlation
    ax = fig.add_subplot(gs[2, 1])
    plot_return_correlation(ax, methods, meta, max_t)

    # Row 2, Col 2: Summary table
    ax = fig.add_subplot(gs[2, 2])
    plot_summary_table(ax, all_metrics)

    fig.suptitle(
        "Advantage Estimation Analysis: MC vs GAE vs IQL (sparse reward)",
        fontsize=13, fontweight="bold",
    )

    # ---------------------------------------------------------------
    # 3. Output
    # ---------------------------------------------------------------

    # Print summary
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    header = f"{'Method':<8} {'KS stat':>8} {'KS p-val':>10} {'Cohen d':>9} "
    header += f"{'AUC':>7} {'Disc.':>7} {'Mean rho':>9}"
    print(header)
    print("-" * 75)
    for name in ["MC", "GAE", "IQL"]:
        m = all_metrics[name]
        print(
            f"{name:<8} {m['ks_stat']:>8.3f} {m['ks_p']:>10.2e} "
            f"{m['cohens_d']:>9.2f} {m['auc']:>7.3f} "
            f"{m['mean_disc']:>7.3f} {m['mean_rho']:>9.3f}"
        )
    print("=" * 75)

    # Save CSV
    if args.csv_output:
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "ks_stat", "ks_p", "cohens_d",
                             "auc", "disc_score", "mean_rho"])
            for name in ["MC", "GAE", "IQL"]:
                m = all_metrics[name]
                writer.writerow([name, m["ks_stat"], m["ks_p"], m["cohens_d"],
                                 m["auc"], m["mean_disc"], m["mean_rho"]])
        print(f"Saved CSV to {args.csv_output}")

    # Save or show figure
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
