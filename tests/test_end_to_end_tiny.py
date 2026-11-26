"""End-to-end tiny-backend smoke test across Base -> SFT -> OPD/DPO -> GRPO -> DAPO+RAGEN2."""

from __future__ import annotations

from pathlib import Path

from agent_rl_flywheel.backends.tiny_local import TinyCausalLMBackend
from agent_rl_flywheel.data.generator import build_benchmark_dataset
from agent_rl_flywheel.data.teacher_deepseek import DeepSeekTeacherClient, build_prefix_preserving_opd_and_dpo
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness, build_prompt_and_gold_trajectory
from agent_rl_flywheel.env.proxy import AgentProxyGateway
from agent_rl_flywheel.eval.evaluator import evaluate_policy, render_ablation_markdown
from agent_rl_flywheel.rl.trainer import AgentRLTrainer
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


def test_end_to_end_tiny_flywheel(tmp_path: Path) -> None:
    bundle = build_benchmark_dataset(seed=99, total_id=24, total_ood=6)
    sympy_tool = SymPyTool()
    retriever = HybridRetriever(bundle.corpus)
    harness = MultiTurnAgentHarness(
        sympy_tool=sympy_tool,
        retriever=retriever,
        max_turns=3,
        max_new_tokens_per_turn=24,
        max_seq_len=1024,
    )
    backend = TinyCausalLMBackend(d_model=32, n_heads=4, n_layers=2, max_seq_len=1024, seed=99)
    trainer = AgentRLTrainer(backend=backend, harness=harness, out_dir=tmp_path / "run", seed=99)

    cold_tasks = bundle.splits["train_coldstart"][:4]
    rl_tasks = bundle.splits["train_rl"][:4]
    id_eval = bundle.splits["locked_test_id"][:3]
    ood_eval = bundle.splits["locked_test_ood"][:3]

    r0 = evaluate_policy("Exp0_Base", "locked_test_id", id_eval, backend, harness, temperature=0.0)
    sft_trajs = [build_prompt_and_gold_trajectory(t, t.gold_turns, backend.encode, harness) for t in cold_tasks]
    sft_stats = trainer.train_sft(sft_trajs, epochs=1, batch_size=2, lr=5e-4)
    assert sft_stats["steps"] >= 2

    # Same-vocab Token-OPD smoke
    fake_teacher_lp = [[-0.2] * len(t.token_ids) for t in sft_trajs]
    opd_stats = trainer.train_token_opd(sft_trajs, fake_teacher_lp, epochs=1, batch_size=2)
    assert opd_stats["steps"] >= 2

    # Prefix-preserving On-Policy DPO with mock DeepSeek teacher
    gw = AgentProxyGateway(backend)
    rollouts = [harness.run_rollout(t, gw, temperature=0.8) for t in rl_tasks]
    tmap = {t.task_id: t for t in rl_tasks}

    def _mock_ds(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        t = next(item for item in rl_tasks if item.prompt == messages[1]["content"])
        asst_cnt = sum(1 for m in messages if m["role"] == "assistant")
        return t.gold_turns[min(asst_cnt, len(t.gold_turns) - 1)]["assistant_text"]

    teacher = DeepSeekTeacherClient(api_key="tiny-e2e", mock_fn=_mock_ds)
    corr = build_prefix_preserving_opd_and_dpo(tmap, rollouts, teacher, harness, backend.encode)
    if corr["dpo_pairs"]:
        dpo_stats = trainer.train_dpo(corr["dpo_pairs"], epochs=1, batch_size=2)
        assert dpo_stats["steps"] >= 1

    grpo_hist = trainer.train_rl(rl_tasks, algo="grpo", use_ragen2=False, steps=2, prompts_per_step=2, group_size=2)
    dapo_hist = trainer.train_rl(rl_tasks, algo="dapo", use_ragen2=True, steps=2, prompts_per_step=2, group_size=2)
    assert len(grpo_hist) == 2 and len(dapo_hist) == 2
    assert all(row["retokenization_events"] == 0 for row in dapo_hist)

    r_final_id = evaluate_policy("Exp2b_DAPO_RAGEN2", "locked_test_id", id_eval, backend, harness, temperature=0.0)
    r_final_ood = evaluate_policy("Exp2b_DAPO_RAGEN2", "locked_test_ood", ood_eval, backend, harness, temperature=0.0)
    md = render_ablation_markdown([r0, r_final_id, r_final_ood])
    assert "Exp2b_DAPO_RAGEN2" in md
    assert (tmp_path / "run" / "train_metrics.jsonl").exists()
