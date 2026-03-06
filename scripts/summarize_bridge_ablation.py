import csv
import glob
import os
import re
from collections import defaultdict

LOG_GLOB = "logs/bridge_ablation/bridge_*.log"
OUT_DIR = "logs/bridge_ablation"

EVAL_RE = re.compile(r"EVAL @ iter (\d+): gpu_sr=([0-9.]+)")
ITER_RE = re.compile(r"Iter (\d+)/(\d+)")
BEST_RE = re.compile(r"Finetuning complete\. Best sr_once=([0-9.]+)")
NAME_RE = re.compile(r"bridge_(.+)_seed(\d+)\.log$")


def auc(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def parse_log(path):
    m = NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    stage = m.group(1)
    seed = int(m.group(2))

    eval_pairs = []
    iters_seen = []
    best_final = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            mm = EVAL_RE.search(line)
            if mm:
                eval_pairs.append((int(mm.group(1)), float(mm.group(2))))
            mi = ITER_RE.search(line)
            if mi:
                iters_seen.append(int(mi.group(1)))
            mb = BEST_RE.search(line)
            if mb:
                best_final = float(mb.group(1))

    if not eval_pairs:
        return {
            "stage": stage,
            "seed": seed,
            "best_sr": 0.0,
            "final_sr": 0.0,
            "auc_sr": 0.0,
            "best_iter": -1,
            "n_eval": 0,
            "n_iter_logged": max(iters_seen) if iters_seen else 0,
            "collapse_flag": 1,
        }

    eval_pairs.sort(key=lambda x: x[0])
    eval_srs = [x[1] for x in eval_pairs]
    best_sr = max(eval_srs)
    best_iter = eval_pairs[eval_srs.index(best_sr)][0]
    final_sr = eval_srs[-1]
    auc_sr = auc(eval_srs)
    collapse_flag = 1 if (best_sr > 0 and final_sr < 0.7 * best_sr) else 0

    return {
        "stage": stage,
        "seed": seed,
        "best_sr": best_sr,
        "final_sr": final_sr,
        "auc_sr": auc_sr,
        "best_iter": best_iter,
        "n_eval": len(eval_srs),
        "n_iter_logged": max(iters_seen) if iters_seen else 0,
        "collapse_flag": collapse_flag,
        "best_sr_reported": best_final if best_final is not None else best_sr,
    }


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return m, v ** 0.5


def main():
    paths = sorted(glob.glob(LOG_GLOB))
    if not paths:
        print(f"No logs found: {LOG_GLOB}")
        return

    rows = []
    for p in paths:
        item = parse_log(p)
        if item is not None:
            rows.append(item)

    rows.sort(key=lambda r: (r["stage"], r["seed"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    detail_csv = os.path.join(OUT_DIR, "summary.csv")
    all_fields = []
    field_set = set()
    for r in rows:
        for k in r.keys():
            if k not in field_set:
                field_set.add(k)
                all_fields.append(k)

    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader()
        w.writerows(rows)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["stage"]].append(r)

    stage_rows = []
    for stage in sorted(grouped.keys()):
        g = grouped[stage]
        best_m, best_s = mean_std([x["best_sr"] for x in g])
        fin_m, fin_s = mean_std([x["final_sr"] for x in g])
        auc_m, auc_s = mean_std([x["auc_sr"] for x in g])
        collapse_m, _ = mean_std([x["collapse_flag"] for x in g])
        stage_rows.append({
            "stage": stage,
            "n_seeds": len(g),
            "best_sr_mean": best_m,
            "best_sr_std": best_s,
            "final_sr_mean": fin_m,
            "final_sr_std": fin_s,
            "auc_sr_mean": auc_m,
            "auc_sr_std": auc_s,
            "collapse_rate": collapse_m,
        })

    stage_csv = os.path.join(OUT_DIR, "summary_stage.csv")
    with open(stage_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        w.writeheader()
        w.writerows(stage_rows)

    stage_order = [
        "s0_reinforce_is_matched",
        "s1_continuous_return",
        "s2_gamma_denoising",
        "s3_value_baseline_td1",
        "s3b_value_baseline_td1_warmup",
        "s4_gae",
        "s4b_gae_warmup",
        "s5_norm_adv",
        "s6_critic_warmup",
        "s7_full_dppo_equiv",
    ]
    idx = {s: i for i, s in enumerate(stage_order)}
    stage_rows_sorted = sorted(stage_rows, key=lambda x: idx.get(x["stage"], 999))

    deltas = []
    for i in range(1, len(stage_rows_sorted)):
        prev = stage_rows_sorted[i - 1]
        cur = stage_rows_sorted[i]
        deltas.append({
            "transition": f"{prev['stage']} -> {cur['stage']}",
            "delta_best_sr_mean": cur["best_sr_mean"] - prev["best_sr_mean"],
            "delta_final_sr_mean": cur["final_sr_mean"] - prev["final_sr_mean"],
            "delta_auc_sr_mean": cur["auc_sr_mean"] - prev["auc_sr_mean"],
        })

    best_jump = max(deltas, key=lambda x: x["delta_best_sr_mean"]) if deltas else None

    report = os.path.join(OUT_DIR, "core_factor_report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# Bridge Ablation Report\n\n")
        f.write("## Stage Summary (mean over seeds)\n\n")
        for s in stage_rows_sorted:
            f.write(
                f"- {s['stage']}: best={s['best_sr_mean']:.3f}±{s['best_sr_std']:.3f}, "
                f"final={s['final_sr_mean']:.3f}±{s['final_sr_std']:.3f}, "
                f"auc={s['auc_sr_mean']:.3f}±{s['auc_sr_std']:.3f}, "
                f"collapse_rate={s['collapse_rate']:.2f}\n"
            )
        f.write("\n## Transition Deltas\n\n")
        for d in deltas:
            f.write(
                f"- {d['transition']}: "
                f"d_best={d['delta_best_sr_mean']:+.3f}, "
                f"d_final={d['delta_final_sr_mean']:+.3f}, "
                f"d_auc={d['delta_auc_sr_mean']:+.3f}\n"
            )
        if best_jump:
            f.write("\n## Candidate Core Factor\n\n")
            f.write(f"Largest best-SR jump: **{best_jump['transition']}** ({best_jump['delta_best_sr_mean']:+.3f}).\n")

    print(f"Wrote: {detail_csv}")
    print(f"Wrote: {stage_csv}")
    print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
