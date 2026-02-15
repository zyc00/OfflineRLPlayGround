"""Compare MC, GAE, and IQL estimates: values, action-values, advantages, and rank correlation.

Extends compare_gae_mc.py to three methods. Columns are V(s), Q(s,a), A(s,a).
Rows show pairwise scatter plots, overlaid histograms, difference histograms,
rank correlations, and a summary table.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

METHOD_NAMES = ["MC", "GAE", "IQL"]
METHOD_COLORS = {"MC": "C0", "GAE": "C1", "IQL": "C2"}
PAIRS = [("MC", "GAE"), ("MC", "IQL"), ("GAE", "IQL")]
QUANTITIES = [("values", "V(s)"), ("action_values", "Q(s,a)"), ("advantages", "A(s,a)")]


def load_estimates(path):
    data = torch.load(path)
    return {k: v.numpy() for k, v in data.items()}


def scatter_with_diagonal(ax, x, y, xlabel, ylabel, title):
    ax.scatter(x, y, alpha=0.3, s=10, edgecolors="none")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "r--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    r, _ = stats.pearsonr(x, y)
    ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=9)


def rank_correlation_plot(ax, adv_a, adv_b, name_a, name_b):
    ranks_a = stats.rankdata(adv_a)
    ranks_b = stats.rankdata(adv_b)

    ax.scatter(ranks_a, ranks_b, alpha=0.3, s=10, edgecolors="none")
    lo, hi = 0, len(ranks_a) + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel(f"{name_a} advantage rank")
    ax.set_ylabel(f"{name_b} advantage rank")
    ax.set_title(f"Rank: {name_a} vs {name_b}")

    tau, tau_p = stats.kendalltau(adv_a, adv_b)
    rho, rho_p = stats.spearmanr(adv_a, adv_b)
    ax.annotate(
        f"Spearman \u03c1 = {rho:.3f} (p={rho_p:.2e})\n"
        f"Kendall \u03c4 = {tau:.3f} (p={tau_p:.2e})",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=8,
    )
    return rho, tau


def overlaid_histogram(ax, data_dict, label):
    all_vals = np.concatenate(list(data_dict.values()))
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)
    for name in METHOD_NAMES:
        ax.hist(data_dict[name], bins=bins, alpha=0.4, label=name,
                color=METHOD_COLORS[name], density=True,
                edgecolor="white", linewidth=0.3)
    ax.set_title(f"{label} distribution")
    ax.set_ylabel("density")
    ax.set_xlabel(label)
    ax.legend(fontsize=7)


def difference_histogram(ax, vals_a, vals_b, name_a, name_b, label):
    diff = vals_b - vals_a
    ax.hist(diff, bins=40, alpha=0.7, color="C4")
    ax.axvline(0, color="r", ls="--", lw=1)
    ax.set_title(f"{label}: {name_b} \u2212 {name_a}")
    ax.set_ylabel("count")
    ax.set_xlabel(f"{name_b} \u2212 {name_a}")
    ax.annotate(
        f"mean={diff.mean():.4f}\nstd={diff.std():.4f}",
        xy=(0.05, 0.78),
        xycoords="axes fraction",
        fontsize=8,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mc",
        default="data/datasets/mc_estimates_gamma0.8_iters10.pt",
        help="path to MC estimates .pt file",
    )
    parser.add_argument(
        "--gae",
        default="data/datasets/gae_estimates_gamma0.8_lambda0.95.pt",
        help="path to GAE estimates .pt file",
    )
    parser.add_argument(
        "--iql",
        default="data/datasets/iql_estimates_gamma0.8_tau0.7.pt",
        help="path to IQL estimates .pt file",
    )
    parser.add_argument(
        "--output",
        default="stats/compare_gae_mc_iql.png",
        help="save figure to this path",
    )
    args = parser.parse_args()

    mc = load_estimates(args.mc)
    gae = load_estimates(args.gae)
    iql = load_estimates(args.iql)
    estimates = {"MC": mc, "GAE": gae, "IQL": iql}

    assert mc["values"].shape == gae["values"].shape == iql["values"].shape, (
        f"Shape mismatch: MC {mc['values'].shape}, "
        f"GAE {gae['values'].shape}, IQL {iql['values'].shape}"
    )

    # 7 rows x 3 cols
    fig = plt.figure(figsize=(18, 32), constrained_layout=True)
    gs = fig.add_gridspec(7, 3)

    # --- Rows 0-2: Pairwise scatter plots (V, Q, A columns) ---
    for row, (name_a, name_b) in enumerate(PAIRS):
        for col, (key, label) in enumerate(QUANTITIES):
            ax = fig.add_subplot(gs[row, col])
            scatter_with_diagonal(
                ax,
                estimates[name_a][key],
                estimates[name_b][key],
                f"{name_a} {label}",
                f"{name_b} {label}",
                f"{label}: {name_a} vs {name_b}",
            )

    # --- Row 3: Overlaid histograms (V, Q, A) ---
    for col, (key, label) in enumerate(QUANTITIES):
        ax = fig.add_subplot(gs[3, col])
        overlaid_histogram(
            ax, {name: est[key] for name, est in estimates.items()}, label
        )

    # --- Row 4: Pairwise difference histograms (V, Q, A) ---
    for col, (key, label) in enumerate(QUANTITIES):
        ax = fig.add_subplot(gs[4, col])
        # Show all 3 pairs overlaid
        all_diffs = {}
        for name_a, name_b in PAIRS:
            all_diffs[(name_a, name_b)] = estimates[name_b][key] - estimates[name_a][key]
        all_vals = np.concatenate(list(all_diffs.values()))
        bins = np.linspace(all_vals.min(), all_vals.max(), 40)
        pair_colors = ["C3", "C4", "C5"]
        for i, (name_a, name_b) in enumerate(PAIRS):
            diff = all_diffs[(name_a, name_b)]
            ax.hist(diff, bins=bins, alpha=0.4, label=f"{name_b}\u2212{name_a}",
                    color=pair_colors[i], density=True,
                    edgecolor="white", linewidth=0.3)
        ax.axvline(0, color="r", ls="--", lw=1)
        ax.set_title(f"{label} differences")
        ax.set_ylabel("density")
        ax.set_xlabel("difference")
        ax.legend(fontsize=7)

    # --- Row 5: Rank correlation plots ---
    all_rho, all_tau = {}, {}
    for col, (name_a, name_b) in enumerate(PAIRS):
        ax = fig.add_subplot(gs[5, col])
        rho, tau = rank_correlation_plot(
            ax,
            estimates[name_a]["advantages"],
            estimates[name_b]["advantages"],
            name_a, name_b,
        )
        all_rho[(name_a, name_b)] = rho
        all_tau[(name_a, name_b)] = tau

    # --- Row 6: Summary table (spanning full width) ---
    ax_table = fig.add_subplot(gs[6, :])
    ax_table.axis("off")
    rows = []
    for key, label in QUANTITIES:
        r_mc_gae, _ = stats.pearsonr(mc[key], gae[key])
        r_mc_iql, _ = stats.pearsonr(mc[key], iql[key])
        r_gae_iql, _ = stats.pearsonr(gae[key], iql[key])
        rows.append([
            label,
            f"{mc[key].mean():.4f}",
            f"{gae[key].mean():.4f}",
            f"{iql[key].mean():.4f}",
            f"{mc[key].std():.4f}",
            f"{gae[key].std():.4f}",
            f"{iql[key].std():.4f}",
            f"{r_mc_gae:.3f}",
            f"{r_mc_iql:.3f}",
            f"{r_gae_iql:.3f}",
        ])
    rows.append([
        "A ranks",
        "", "", "", "", "", "",
        f"\u03c1={all_rho[('MC', 'GAE')]:.3f}",
        f"\u03c1={all_rho[('MC', 'IQL')]:.3f}",
        f"\u03c1={all_rho[('GAE', 'IQL')]:.3f}",
    ])

    table = ax_table.table(
        cellText=rows,
        colLabels=[
            "", "MC mean", "GAE mean", "IQL mean",
            "MC std", "GAE std", "IQL std",
            "r(MC,GAE)", "r(MC,IQL)", "r(GAE,IQL)",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)
    ax_table.set_title("Summary statistics", fontsize=11, pad=15)

    fig.suptitle(
        "Estimate Comparison: MC vs GAE vs IQL",
        fontsize=14, fontweight="bold",
    )

    fig.savefig(args.output, dpi=150)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
