import csv
import glob
import os
import re


LOG_GLOB = "logs/p4d_3k_sweep/p4d3k_*.log"
OUT_DIR = "logs/p4d_3k_sweep"

EVAL_RE = re.compile(r"EVAL @ iter (\d+): gpu_sr=([0-9.]+)")
BEST_RE = re.compile(r"Finetuning complete\. Best sr_once=([0-9.]+)")
NAME_RE = re.compile(
    r"p4d3k_ne(?P<n_envs>\d+)_ns(?P<n_steps>\d+)_g(?P<gamma>[0-9p]+)"
    r"_ep(?P<update_epochs>\d+)_mb(?P<minibatch_size>\d+)_gd(?P<gamma_denoising>[0-9p]+)"
    r"_clip(?P<is_clip_ratio>[0-9p]+)_itr(?P<n_train_itr>\d+)_seed(?P<seed>\d+)\.log$"
)


def unfloat(text: str) -> float:
    return float(text.replace("p", "."))


def parse(path: str):
    m = NAME_RE.search(os.path.basename(path))
    if not m:
        return None

    evals = []
    best_reported = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            me = EVAL_RE.search(line)
            if me:
                evals.append((int(me.group(1)), float(me.group(2))))
            mb = BEST_RE.search(line)
            if mb:
                best_reported = float(mb.group(1))

    evals.sort(key=lambda x: x[0])
    ys = [y for _, y in evals]
    best_sr = max(ys) if ys else 0.0
    final_sr = ys[-1] if ys else 0.0
    best_iter = evals[ys.index(best_sr)][0] if ys else 0

    row = {
        "path": path,
        "n_envs": int(m.group("n_envs")),
        "n_steps": int(m.group("n_steps")),
        "gamma": unfloat(m.group("gamma")),
        "update_epochs": int(m.group("update_epochs")),
        "minibatch_size": int(m.group("minibatch_size")),
        "gamma_denoising": unfloat(m.group("gamma_denoising")),
        "is_clip_ratio": unfloat(m.group("is_clip_ratio")),
        "n_train_itr": int(m.group("n_train_itr")),
        "seed": int(m.group("seed")),
        "best_sr": best_sr,
        "final_sr": final_sr,
        "best_iter": best_iter,
        "best_sr_reported": best_reported if best_reported is not None else best_sr,
        "n_eval": len(evals),
    }
    row["traj_per_iter"] = round(row["n_envs"] * row["n_steps"] / 25.0)
    row["total_traj"] = row["traj_per_iter"] * row["n_train_itr"]
    return row


def main():
    paths = sorted(glob.glob(LOG_GLOB))
    if not paths:
        print(f"No logs found: {LOG_GLOB}")
        return

    rows = [r for r in (parse(p) for p in paths) if r is not None]
    rows.sort(key=lambda r: (-r["best_sr"], -r["final_sr"], r["n_envs"], r["n_steps"]))

    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(OUT_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# p4d 3k Sweep Summary\n\n")
        f.write("| Rank | best SR | final SR | best iter | envs | steps | gamma | epochs | mb | gamma_denoising | clip | iters | total traj | log |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for idx, row in enumerate(rows, start=1):
            f.write(
                f"| {idx} | {row['best_sr']:.3f} | {row['final_sr']:.3f} | {row['best_iter']} | "
                f"{row['n_envs']} | {row['n_steps']} | {row['gamma']:.3f} | {row['update_epochs']} | "
                f"{row['minibatch_size']} | {row['gamma_denoising']:.3f} | {row['is_clip_ratio']:.3f} | "
                f"{row['n_train_itr']} | {row['total_traj']} | `{os.path.basename(row['path'])}` |\n"
            )

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
