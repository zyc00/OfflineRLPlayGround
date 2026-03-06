import csv
import glob
import os
import re
from collections import defaultdict

LOG_GLOB = "logs/net_struct_ablation/netab_*_seed*.log"
OUT_DIR = "logs/net_struct_ablation"

EVAL_RE = re.compile(r"EVAL @ iter (\d+): gpu_sr=([0-9.]+)")
BEST_RE = re.compile(r"Finetuning complete\. Best sr_once=([0-9.]+)")
NAME_RE = re.compile(r"netab_(.+)_seed(\d+)\.log$")


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, v ** 0.5


def parse(path):
    m = NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    model = m.group(1)
    seed = int(m.group(2))

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

    if not evals:
        return {
            "model": model,
            "seed": seed,
            "best_sr": 0.0,
            "final_sr": 0.0,
            "auc_sr": 0.0,
            "n_eval": 0,
        }

    evals.sort(key=lambda x: x[0])
    ys = [x[1] for x in evals]
    return {
        "model": model,
        "seed": seed,
        "best_sr": max(ys),
        "final_sr": ys[-1],
        "auc_sr": sum(ys) / len(ys),
        "n_eval": len(ys),
        "best_sr_reported": best_reported if best_reported is not None else max(ys),
    }


def main():
    paths = sorted(glob.glob(LOG_GLOB))
    if not paths:
        print(f"No logs found: {LOG_GLOB}")
        return

    rows = [r for r in (parse(p) for p in paths) if r is not None]
    rows.sort(key=lambda r: (r["model"], r["seed"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)

    detail_csv = os.path.join(OUT_DIR, "summary.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["model"]].append(r)

    stage_rows = []
    for model in sorted(grouped.keys()):
        g = grouped[model]
        bm, bs = mean_std([x["best_sr"] for x in g])
        fm, fs = mean_std([x["final_sr"] for x in g])
        am, asd = mean_std([x["auc_sr"] for x in g])
        stage_rows.append({
            "model": model,
            "n_seeds": len(g),
            "best_sr_mean": bm,
            "best_sr_std": bs,
            "final_sr_mean": fm,
            "final_sr_std": fs,
            "auc_sr_mean": am,
            "auc_sr_std": asd,
        })

    stage_csv = os.path.join(OUT_DIR, "summary_model.csv")
    with open(stage_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        w.writeheader()
        w.writerows(stage_rows)

    report = os.path.join(OUT_DIR, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Network Structure Ablation\n\n")
        f.write("Fixed finetune config = scripts/run_dppo_finetune.sh settings.\n\n")
        f.write("Note: backbone comparison uses available checkpoints; pretrain data may differ.\n\n")
        for s in stage_rows:
            f.write(
                f"- {s['model']}: best={s['best_sr_mean']:.3f}±{s['best_sr_std']:.3f}, "
                f"final={s['final_sr_mean']:.3f}±{s['final_sr_std']:.3f}, "
                f"auc={s['auc_sr_mean']:.3f}±{s['auc_sr_std']:.3f}\n"
            )

    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {stage_csv}")
    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
