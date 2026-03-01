#!/usr/bin/env python3
"""一键查看实验进度。

Usage:
    python check_exp.py                          # 列出所有最近的实验
    python check_exp.py runs/dppo_pretrain/xxx   # 查看指定实验
    python check_exp.py peg                      # 模糊匹配
    python check_exp.py --all                    # 显示所有实验（不限最近）
"""

import os
import sys
import glob
import time
from datetime import datetime, timedelta

def find_runs(pattern=None, base_dirs=None, show_all=False):
    """Find run directories containing TensorBoard event files."""
    if base_dirs is None:
        base_dirs = ["runs"]

    results = []
    for base in base_dirs:
        for root, dirs, files in os.walk(base):
            for f in files:
                if f.startswith("events.out.tfevents"):
                    evt_path = os.path.join(root, f)
                    mtime = os.path.getmtime(evt_path)
                    results.append((root, evt_path, mtime))
                    break  # one per directory

    # Deduplicate by directory
    seen = set()
    unique = []
    for root, evt, mt in results:
        if root not in seen:
            seen.add(root)
            unique.append((root, evt, mt))
    results = unique

    # Filter by pattern
    if pattern:
        if os.path.isdir(pattern):
            results = [(r, e, m) for r, e, m in results if r == pattern or r.startswith(pattern)]
        else:
            pattern_lower = pattern.lower()
            results = [(r, e, m) for r, e, m in results if pattern_lower in r.lower()]

    # Sort by modification time (newest first)
    results.sort(key=lambda x: -x[2])

    # Default: only show last 24h
    if not show_all and not pattern:
        cutoff = time.time() - 86400
        results = [(r, e, m) for r, e, m in results if m > cutoff]

    return results


def load_tb(run_dir):
    """Load TensorBoard data from a run directory."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir)
    ea.Reload()
    return ea


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def show_run(run_dir, verbose=True):
    """Display detailed info for a single run."""
    ea = load_tb(run_dir)
    tags = ea.Tags().get("scalars", [])

    if not tags:
        print(f"  (no scalar data)")
        return

    # Categorize tags
    eval_tags = sorted([t for t in tags if "eval" in t or "success" in t or "sr" in t.lower()])
    loss_tags = sorted([t for t in tags if "loss" in t])
    other_tags = sorted([t for t in tags if t not in eval_tags and t not in loss_tags])

    # Get time info from any tag
    all_events = []
    for tag in tags:
        events = ea.Scalars(tag)
        if events:
            all_events.extend(events)

    if not all_events:
        print(f"  (no events)")
        return

    all_events.sort(key=lambda e: e.wall_time)
    t_start = all_events[0].wall_time
    t_end = all_events[-1].wall_time
    duration = t_end - t_start

    start_str = datetime.fromtimestamp(t_start).strftime("%m-%d %H:%M")
    last_str = datetime.fromtimestamp(t_end).strftime("%m-%d %H:%M")

    # Check if still running (last event within 5 min)
    is_running = (time.time() - t_end) < 300
    status = "🟢 RUNNING" if is_running else "⏹ STOPPED"

    print(f"  Status: {status}  |  Started: {start_str}  |  Last: {last_str}  |  Duration: {format_time(duration)}")

    # Speed estimation from loss (most frequent tag)
    speed_tag = loss_tags[0] if loss_tags else tags[0]
    speed_events = ea.Scalars(speed_tag)
    if len(speed_events) >= 10:
        recent = speed_events[-10:]
        dt = recent[-1].wall_time - recent[0].wall_time
        ds = recent[-1].step - recent[0].step
        if dt > 0 and ds > 0:
            speed = ds / dt
            print(f"  Speed: {speed:.1f} it/s")

            # Try to estimate total iters from step progression
            max_step = max(e.step for e in speed_events)
            if is_running:
                # Guess total from common round numbers
                for total in [1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
                    if total > max_step:
                        eta = (total - max_step) / speed
                        print(f"  Progress: {max_step}/{total} ({max_step/total*100:.1f}%)  ETA: {format_time(eta)}")
                        break

    # Eval metrics
    if eval_tags:
        print(f"\n  --- Eval ---")
        for tag in eval_tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            # Show last 10 evals
            show_events = events[-10:]
            vals = [(e.step, e.value) for e in show_events]

            # Find best
            best_step, best_val = max([(e.step, e.value) for e in events], key=lambda x: x[1])

            tag_short = tag.split("/")[-1]
            val_strs = [f"{s}:{v:.3f}" for s, v in vals]
            print(f"  {tag_short}: {' → '.join(val_strs)}")
            print(f"    best={best_val:.3f} @ iter {best_step}")

    # Loss
    if loss_tags and verbose:
        print(f"\n  --- Loss ---")
        for tag in loss_tags:
            events = ea.Scalars(tag)
            if not events:
                continue
            tag_short = tag.split("/")[-1]
            first_val = events[0].value
            last_val = events[-1].value
            min_val = min(e.value for e in events)
            print(f"  {tag_short}: {first_val:.6f} → {last_val:.6f}  (min={min_val:.6f}, {len(events)} records)")

    # Other metrics (compact)
    if other_tags and verbose:
        print(f"\n  --- Other ({len(other_tags)} tags) ---")
        for tag in other_tags[:8]:
            events = ea.Scalars(tag)
            if events:
                tag_short = tag.split("/")[-1]
                last = events[-1]
                print(f"  {tag_short}: {last.value:.4f} @ iter {last.step}")
        if len(other_tags) > 8:
            print(f"  ... and {len(other_tags) - 8} more")


def list_runs(runs):
    """Show a compact list of runs."""
    print(f"\n{'='*80}")
    print(f"Found {len(runs)} experiment(s)")
    print(f"{'='*80}")

    for i, (run_dir, evt_path, mtime) in enumerate(runs):
        mtime_str = datetime.fromtimestamp(mtime).strftime("%m-%d %H:%M")
        is_running = (time.time() - mtime) < 300
        status = "🟢" if is_running else "⏹"

        # Get a quick summary
        try:
            ea = load_tb(run_dir)
            tags = ea.Tags().get("scalars", [])
            eval_tags = [t for t in tags if "eval" in t or "success" in t]

            summary = ""
            if eval_tags:
                for tag in eval_tags[:2]:
                    events = ea.Scalars(tag)
                    if events:
                        best = max(events, key=lambda e: e.value)
                        last = events[-1]
                        tag_short = tag.split("/")[-1]
                        summary += f"  {tag_short}={last.value:.3f}(best={best.value:.3f}@{best.step})"

            loss_tags = [t for t in tags if "loss" in t]
            if loss_tags and not summary:
                events = ea.Scalars(loss_tags[0])
                if events:
                    summary = f"  loss={events[-1].value:.6f}@{events[-1].step}"
        except Exception:
            summary = "  (error reading)"

        # Shorten run_dir for display
        display = run_dir
        if display.startswith("runs/"):
            display = display[5:]

        print(f"\n{status} [{i+1}] {display}  ({mtime_str}){summary}")


def main():
    pattern = None
    show_all = False

    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("""check_exp.py — 一键查看实验进度

Usage:
    python check_exp.py                          列出最近24h的实验
    python check_exp.py <pattern>                模糊匹配实验名
    python check_exp.py runs/dppo_pretrain/xxx   查看指定目录
    python check_exp.py -a, --all                显示所有实验（不限24h）
    python check_exp.py -h, --help               显示帮助

Examples:
    python check_exp.py peg          匹配所有含"peg"的实验
    python check_exp.py 1M           匹配含"1M"的实验
    python check_exp.py fbc          匹配filtered BC实验
    python check_exp.py --all        列出所有历史实验""")
        return

    if "--all" in args:
        show_all = True
        args.remove("--all")
    if "-a" in args:
        show_all = True
        args.remove("-a")

    if args:
        pattern = args[0]

    runs = find_runs(pattern, show_all=show_all)

    if not runs:
        if pattern:
            print(f"No experiments matching '{pattern}'")
        else:
            print("No recent experiments (last 24h). Use --all or -a to show all.")
        return

    if len(runs) == 1 or pattern:
        # Show detailed view for matched runs
        for run_dir, evt_path, mtime in runs[:5]:
            print(f"\n{'='*80}")
            print(f"📊 {run_dir}")
            print(f"{'='*80}")
            show_run(run_dir, verbose=True)

        if len(runs) > 5:
            print(f"\n... and {len(runs) - 5} more. Showing top 5.")
    else:
        # Show compact list
        list_runs(runs)


if __name__ == "__main__":
    main()
