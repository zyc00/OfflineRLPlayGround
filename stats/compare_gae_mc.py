"""Compare MC and GAE estimates: values, action-values, advantages, and rank correlation."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


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


def rank_correlation_plot(ax, mc_adv, gae_adv):
    mc_ranks = stats.rankdata(mc_adv)
    gae_ranks = stats.rankdata(gae_adv)

    ax.scatter(mc_ranks, gae_ranks, alpha=0.3, s=10, edgecolors="none")
    lo, hi = 0, len(mc_ranks) + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("MC advantage rank")
    ax.set_ylabel("GAE advantage rank")
    ax.set_title("Advantage rank correlation")

    tau, tau_p = stats.kendalltau(mc_adv, gae_adv)
    rho, rho_p = stats.spearmanr(mc_adv, gae_adv)
    ax.annotate(
        f"Spearman \u03c1 = {rho:.3f} (p={rho_p:.2e})\n"
        f"Kendall \u03c4 = {tau:.3f} (p={tau_p:.2e})",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        fontsize=8,
    )
    return rho, tau


def distribution_comparison(axes, mc_vals, gae_vals, name):
    """Overlaid histograms + difference histogram."""
    ax_hist, ax_diff = axes

    bins = np.linspace(
        min(mc_vals.min(), gae_vals.min()),
        max(mc_vals.max(), gae_vals.max()),
        40,
    )
    ax_hist.hist(mc_vals, bins=bins, alpha=0.5, label="MC", density=True)
    ax_hist.hist(gae_vals, bins=bins, alpha=0.5, label="GAE", density=True)
    ax_hist.set_title(f"{name} distribution")
    ax_hist.legend(fontsize=8)
    ax_hist.set_ylabel("density")

    diff = gae_vals - mc_vals
    ax_diff.hist(diff, bins=40, alpha=0.7, color="C2")
    ax_diff.axvline(0, color="r", ls="--", lw=1)
    ax_diff.set_title(f"{name}: GAE − MC")
    ax_diff.set_ylabel("count")
    ax_diff.annotate(
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
        default="data/datasets/gae_estimates_gamma0.8_lambda0.9.pt",
        help="path to GAE estimates .pt file",
    )
    parser.add_argument("--output", default=None, help="save figure to this path")
    args = parser.parse_args()

    mc = load_estimates(args.mc)
    gae = load_estimates(args.gae)

    assert mc["values"].shape == gae["values"].shape, (
        f"Shape mismatch: MC {mc['values'].shape} vs GAE {gae['values'].shape}"
    )

    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)

    # --- Row 1: scatter plots (MC vs GAE) ---
    for col, key, label in [
        (0, "values", "V(s)"),
        (1, "action_values", "Q(s,a)"),
        (2, "advantages", "A(s,a)"),
    ]:
        ax = fig.add_subplot(gs[0, col])
        scatter_with_diagonal(ax, mc[key], gae[key], f"MC {label}", f"GAE {label}", label)

    # --- Row 2-3: distribution comparisons ---
    for col, key, label in [
        (0, "values", "V(s)"),
        (1, "action_values", "Q(s,a)"),
        (2, "advantages", "A(s,a)"),
    ]:
        ax_hist = fig.add_subplot(gs[1, col])
        ax_diff = fig.add_subplot(gs[2, col])
        distribution_comparison((ax_hist, ax_diff), mc[key], gae[key], label)

    # --- Row 4: rank correlation ---
    ax_rank = fig.add_subplot(gs[3, 0])
    rho, tau = rank_correlation_plot(ax_rank, mc["advantages"], gae["advantages"])

    # Summary stats table
    ax_table = fig.add_subplot(gs[3, 1:])
    ax_table.axis("off")
    rows = []
    for key, label in [
        ("values", "V(s)"),
        ("action_values", "Q(s,a)"),
        ("advantages", "A(s,a)"),
    ]:
        r, _ = stats.pearsonr(mc[key], gae[key])
        rows.append([
            label,
            f"{mc[key].mean():.4f}",
            f"{gae[key].mean():.4f}",
            f"{mc[key].std():.4f}",
            f"{gae[key].std():.4f}",
            f"{r:.3f}",
        ])
    rows.append([
        "Adv ranks",
        "", "", "", "",
        f"\u03c1={rho:.3f}, \u03c4={tau:.3f}",
    ])

    table = ax_table.table(
        cellText=rows,
        colLabels=["", "MC mean", "GAE mean", "MC std", "GAE std", "Correlation"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_table.set_title("Summary statistics", fontsize=11, pad=10)

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved figure to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
