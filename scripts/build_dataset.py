#!/usr/bin/env python3
"""Generate group-leakage-free ID + OOD dual-tool benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_rl_flywheel.data.generator import build_benchmark_dataset, save_dataset_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-free agentic RL benchmark dataset.")
    parser.add_argument("--out", type=str, default="data/generated", help="Output dataset directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-id", type=int, default=400)
    parser.add_argument("--total-ood", type=int, default=60)
    args = parser.parse_args()

    bundle = build_benchmark_dataset(
        seed=args.seed,
        total_id=args.total_id,
        total_ood=args.total_ood,
    )
    out_path = save_dataset_bundle(bundle, Path(args.out))
    print(f"Dataset written to {out_path}: {bundle.metadata['split_counts']}")


if __name__ == "__main__":
    main()
