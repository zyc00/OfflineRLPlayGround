#!/usr/bin/env python3
"""
Merge ManiSkill-style trajectory H5 files by concatenating traj groups.

This keeps only fields needed by DPPO IL pretraining (`obs`, `actions`) unless
`--copy-all` is set.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import h5py


def iter_traj_keys(h5_file: h5py.File) -> list[str]:
    return sorted(
        [k for k in h5_file.keys() if k.startswith("traj_")],
        key=lambda x: int(x.split("_")[1]),
    )


def copy_group(src: h5py.Group, dst: h5py.Group, keys: Iterable[str] | None) -> None:
    if keys is None:
        for key in src.keys():
            src.copy(key, dst, name=key)
        return

    for key in keys:
        if key not in src:
            raise KeyError(f"Missing key '{key}' in source group {src.name}")
        src.copy(key, dst, name=key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base H5 dataset")
    parser.add_argument("--extra", nargs="+", required=True, help="Extra H5 datasets to append")
    parser.add_argument("--output", required=True, help="Merged output H5")
    parser.add_argument(
        "--copy-all",
        action="store_true",
        help="Copy all traj fields instead of only obs/actions",
    )
    args = parser.parse_args()

    base = os.path.expanduser(args.base)
    extras = [os.path.expanduser(p) for p in args.extra]
    output = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    keys_to_copy = None if args.copy_all else ("obs", "actions")

    total = 0
    with h5py.File(output, "w") as fout:
        for path in [base] + extras:
            with h5py.File(path, "r") as fin:
                traj_keys = iter_traj_keys(fin)
                for traj_key in traj_keys:
                    dst = fout.create_group(f"traj_{total}")
                    copy_group(fin[traj_key], dst, keys_to_copy)
                    total += 1
                print(f"Appended {len(traj_keys)} trajs from {path}")

    print(f"Merged total: {total} trajs -> {output}")


if __name__ == "__main__":
    main()
