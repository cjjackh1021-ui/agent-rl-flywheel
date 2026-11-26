#!/usr/bin/env python3
"""Run end-to-end experiment stages on TinyCausalLM (Mac) or HFQLoRA (RTX 4070 SUPER 12GB)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from agent_rl_flywheel.backends.hf_qlora import HFQLoRABackend
from agent_rl_flywheel.backends.tiny_local import TinyCausalLMBackend
from agent_rl_flywheel.data.generator import load_dataset_bundle
from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.data.teacher_deepseek import DeepSeekTeacherClient, build_prefix_preserving_opd_and_dpo
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness, build_prompt_and_gold_trajectory
from agent_rl_flywheel.env.proxy import AgentProxyGateway
from agent_rl_flywheel.env.verifier import RLVRVerifier
from agent_rl_flywheel.eval.evaluator import EvalResult, evaluate_policy, render_ablation_markdown
from agent_rl_flywheel.rl.trainer import AgentRLTrainer
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


def _make_mock_teacher_for_smoke(tasks_by_id: dict[str, TaskItem]) -> DeepSeekTeacherClient:
    """Deterministic offline mock used solely during `--backend tiny --stage smoke_all`."""

    def _mock_complete(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        user_msg = messages[1]["content"]
        matched = next((t for t in tasks_by_id.values() if t.prompt == user_msg), None)
        if matched is None:
            return "<think>fallback</think><answer>0</answer>"
        asst_seen = sum(1 for m in messages if m["role"] == "assistant")
        if asst_seen < len(matched.gold_turns):
            return matched.gold_turns[asst_seen]["assistant_text"]
        return matched.gold_turns[-1]["assistant_text"]

    return DeepSeekTeacherClient(api_key="smoke-mock", mock_fn=_mock_complete)


def _load_backend(backend_name: str, cfg: dict[str, Any]) -> Any:
    model_cfg = cfg.get("model", {})
    if backend_name == "tiny":
        return TinyCausalLMBackend(
            max_seq_len=1024,
            seed=int(cfg.get("seed", 42)),
        )
    if backend_name == "hf_qlora":
        return HFQLoRABackend(
            model_name=str(model_cfg.get("student_model", "Qwen/Qwen3.5-2B")),
            qlora_4bit=bool(model_cfg.get("qlora_4bit", True)),
            lora_r=int(model_cfg.get("lora_r", 16)),
            lora_alpha=int(model_cfg.get("lora_alpha", 32)),
            lora_dropout=float(model_cfg.get("lora_dropout", 0.05)),
            lora_target_modules=model_cfg.get("lora_target_modules"),
            gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
        )
    raise ValueError(f"Unknown backend: {backend_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Agentic Post-Training Flywheel experiments.")
    parser.add_argument("--config", type=str, default="configs/qwen35_2b_4070.yaml")
    parser.add_argument("--backend", choices=["tiny", "hf_qlora"], default="tiny")
    parser.add_argument(
        "--stage",
        choices=["eval", "sft", "opd_dpo", "grpo", "dapo_ragen2", "smoke_all"],
        default="smoke_all",
    )
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--out-dir", type=str, default="outputs/run")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) if Path(args.config).exists() else {}
    bundle = load_dataset_bundle(args.data_dir)
    sympy_tool = SymPyTool()
    retriever = HybridRetriever(bundle.corpus)
    harness_cfg = cfg.get("harness", {})
    model_cfg = cfg.get("model", {})
    verifier = RLVRVerifier(
        sympy_tool=sympy_tool,
        soft_max_tokens=int(harness_cfg.get("soft_max_tokens", 1400)),
        hard_max_tokens=int(harness_cfg.get("hard_max_tokens", 2048)),
        overlong_penalty_max=float(harness_cfg.get("overlong_penalty_max", 0.30)),
    )
    harness = MultiTurnAgentHarness(
        sympy_tool=sympy_tool,
        retriever=retriever,
        verifier=verifier,
        max_turns=int(harness_cfg.get("max_turns", 4)),
        max_new_tokens_per_turn=32 if args.backend == "tiny" else int(model_cfg.get("max_new_tokens_per_turn", 384)),
        max_seq_len=1024 if args.backend == "tiny" else int(model_cfg.get("max_seq_len", 2048)),
    )
    backend = _load_backend(args.backend, cfg)
    out_path = Path(args.out_dir)
    trainer = AgentRLTrainer(backend=backend, harness=harness, out_dir=out_path, seed=int(cfg.get("seed", 42)))

    coldstart_tasks = bundle.splits.get("train_coldstart", [])
    rl_tasks = bundle.splits.get("train_rl", [])
    id_eval_tasks = bundle.splits.get("locked_test_id", [])[: (8 if args.backend == "tiny" else 100)]
    ood_eval_tasks = bundle.splits.get("locked_test_ood", [])[: (6 if args.backend == "tiny" else 60)]
    train_cfg = cfg.get("training", {})
    r2_cfg = cfg.get("ragen2", {})

    results: list[EvalResult] = []

    def _eval_both(label: str) -> None:
        results.append(evaluate_policy(label, "locked_test_id", id_eval_tasks, backend, harness))
        results.append(evaluate_policy(label, "locked_test_ood", ood_eval_tasks, backend, harness))

    if args.stage in {"eval", "smoke_all"}:
        _eval_both("Exp0_Base")

    if args.stage in {"sft", "smoke_all"}:
        sft_subset = coldstart_tasks[: (8 if args.backend == "tiny" else len(coldstart_tasks))]
        sft_trajs = [
            build_prompt_and_gold_trajectory(t, t.gold_turns, backend.encode, harness)
            for t in sft_subset
            if t.gold_turns
        ]
        trainer.train_sft(
            sft_trajs,
            epochs=1 if args.backend == "tiny" else int(train_cfg.get("sft_epochs", 2)),
            batch_size=2 if args.backend == "tiny" else 4,
            lr=float(train_cfg.get("sft_lr", 1.5e-4)),
        )
        _eval_both("Exp1a_Static_SFT")

    if args.stage in {"opd_dpo", "smoke_all"}:
        opd_subset = rl_tasks[: (6 if args.backend == "tiny" else 100)]
        gw = AgentProxyGateway(backend)
        student_rollouts = [harness.run_rollout(t, gw, temperature=0.8) for t in opd_subset]
        tasks_map = {t.task_id: t for t in opd_subset}
        teacher = (
            _make_mock_teacher_for_smoke(tasks_map)
            if args.backend == "tiny"
            else DeepSeekTeacherClient(
                model=str(cfg.get("teacher", {}).get("teacher_model", "deepseek-flash")),
                base_url=str(cfg.get("teacher", {}).get("teacher_base_url", "https://api.deepseek.com")),
            )
        )
        corrections = build_prefix_preserving_opd_and_dpo(
            tasks_by_id=tasks_map,
            failed_trajectories=student_rollouts,
            client=teacher,
            harness=harness,
            encode_fn=backend.encode,
        )
        if corrections["opd_trajectories"]:
            trainer.train_sft(
                corrections["opd_trajectories"],
                epochs=1,
                batch_size=2,
                lr=float(train_cfg.get("sft_lr", 1.5e-4)),
                stage_name="prefix_opd_sft",
            )
        if corrections["dpo_pairs"]:
            trainer.train_dpo(
                corrections["dpo_pairs"],
                epochs=1 if args.backend == "tiny" else int(train_cfg.get("dpo_epochs", 1)),
                batch_size=2,
                lr=float(train_cfg.get("dpo_lr", 5e-5)),
                beta=float(train_cfg.get("dpo_beta", 0.1)),
            )
        _eval_both("Exp1b_Prefix_OPD_DPO")

    if args.stage in {"grpo", "smoke_all"}:
        trainer.train_rl(
            rl_tasks[: (12 if args.backend == "tiny" else len(rl_tasks))],
            algo="grpo",
            use_ragen2=False,
            steps=2 if args.backend == "tiny" else int(train_cfg.get("rl_steps", 120)),
            prompts_per_step=2 if args.backend == "tiny" else int(train_cfg.get("prompts_per_step", 4)),
            group_size=2 if args.backend == "tiny" else int(train_cfg.get("group_size", 4)),
            lr=float(train_cfg.get("rl_lr", 2e-5)),
            kl_beta=float(train_cfg.get("kl_beta", 0.02)),
        )
        _eval_both("Exp2a_GRPO")

    if args.stage in {"dapo_ragen2", "smoke_all"}:
        trainer.train_rl(
            rl_tasks[: (12 if args.backend == "tiny" else len(rl_tasks))],
            algo="dapo",
            use_ragen2=bool(r2_cfg.get("enabled", True)),
            steps=2 if args.backend == "tiny" else int(train_cfg.get("rl_steps", 120)),
            prompts_per_step=2 if args.backend == "tiny" else int(train_cfg.get("prompts_per_step", 4)),
            group_size=2 if args.backend == "tiny" else int(train_cfg.get("group_size", 4)),
            lr=float(train_cfg.get("rl_lr", 2e-5)),
            clip_eps=float(train_cfg.get("clip_eps_low", 0.2)),
            clip_eps_high=float(train_cfg.get("clip_eps_high", 0.28)),
            kl_beta=float(train_cfg.get("kl_beta", 0.02)),
            dynamic_sampling=bool(train_cfg.get("dynamic_sampling", True)),
            snr_min_reward_std=float(r2_cfg.get("snr_min_reward_std", 0.05)),
            snr_top_k_ratio=float(r2_cfg.get("snr_top_k_ratio", 0.75)),
            mi_collapse_threshold=float(r2_cfg.get("mi_collapse_threshold", 0.15)),
        )
        _eval_both("Exp2b_DAPO_RAGEN2")

    report = render_ablation_markdown(results)
    report_path = out_path / "ablation_report.md"
    report_path.write_text(report, encoding="utf-8")
    backend.save(str(out_path / "checkpoint"))
    print(report)
    print(f"Saved report & checkpoint to {out_path}")


if __name__ == "__main__":
    main()
