#!/usr/bin/env python3
"""Synthesize coldstart SFT trajectories via DeepSeek-V4.1-Flash (`deepseek-flash`)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from agent_rl_flywheel.backends.tiny_local import SimpleByteTokenizer
from agent_rl_flywheel.data.generator import load_dataset_bundle
from agent_rl_flywheel.data.teacher_deepseek import DeepSeekTeacherClient, synthesize_coldstart_sft
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness
from agent_rl_flywheel.env.verifier import RLVRVerifier
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize verified SFT trajectories with DeepSeek-V4.1-Flash.")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--out", type=str, default="outputs/teacher_coldstart.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", type=str, default="deepseek-flash")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com")
    args = parser.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("Please set DEEPSEEK_API_KEY before running teacher synthesis.")

    bundle = load_dataset_bundle(args.data_dir)
    sympy_tool = SymPyTool()
    retriever = HybridRetriever(bundle.corpus)
    harness = MultiTurnAgentHarness(
        sympy_tool=sympy_tool,
        retriever=retriever,
        verifier=RLVRVerifier(sympy_tool=sympy_tool),
    )
    tok = SimpleByteTokenizer()
    client = DeepSeekTeacherClient(model=args.model, base_url=args.base_url)
    tasks = bundle.splits.get("train_coldstart", [])[: args.limit]
    accepted = synthesize_coldstart_sft(
        tasks=tasks,
        client=client,
        harness=harness,
        encode_fn=tok.encode,
        out_path=Path(args.out),
    )
    print(f"Synthesized {len(accepted)} / {len(tasks)} verified trajectories -> {args.out}")


if __name__ == "__main__":
    main()
