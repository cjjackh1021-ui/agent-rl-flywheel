"""Tests for zero-retokenization ExactTokenBuffer, `<obs>` masking, and group_id split isolation."""

from __future__ import annotations

from pathlib import Path

from agent_rl_flywheel.backends.tiny_local import SimpleByteTokenizer
from agent_rl_flywheel.data.generator import (
    build_benchmark_dataset,
    load_dataset_bundle,
    save_dataset_bundle,
)
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness
from agent_rl_flywheel.env.proxy import AgentProxyGateway, TurnSample
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


class _ScriptedExactBackend:
    """Test backend emitting multi-turn responses and tracking exact prefix token identity."""

    def __init__(self, responses: list[str]) -> None:
        self.tok = SimpleByteTokenizer()
        self.responses = list(responses)
        self.seen_prefixes: list[list[int]] = []
        self._idx = 0

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids)

    def generate_with_logprobs(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.7,
    ) -> TurnSample:
        self.seen_prefixes.append(list(prompt_token_ids))
        text = self.responses[min(self._idx, len(self.responses) - 1)]
        self._idx += 1
        ids = self.encode(text)[:max_new_tokens]
        return TurnSample(text=self.decode(ids), token_ids=ids, logprobs=[-0.25] * len(ids))


def test_harness_exact_token_preservation_and_obs_mask() -> None:
    bundle = build_benchmark_dataset(seed=7, total_id=20, total_ood=4)
    rag_task = next(t for t in bundle.splits["train_rl"] if t.task_type == "rag")
    sympy_tool = SymPyTool()
    retriever = HybridRetriever(bundle.corpus)
    harness = MultiTurnAgentHarness(sympy_tool=sympy_tool, retriever=retriever, max_turns=4, max_new_tokens_per_turn=300)

    backend = _ScriptedExactBackend([step["assistant_text"] for step in rag_task.gold_turns])
    gateway = AgentProxyGateway(backend)
    traj = harness.run_rollout(rag_task, gateway, temperature=0.0)

    assert traj.retokenization_events == 0
    assert traj.reward_breakdown["acc"] == 1.0
    assert traj.reward_breakdown["format"] == 1.0
    assert len(traj.token_ids) == len(traj.assistant_mask) == len(traj.old_logprobs)

    # Prompt tokens must be strictly masked out (0)
    assert all(m == 0 for m in traj.assistant_mask[: traj.prompt_len])

    # Every `<obs>` token span must be masked out (0), while assistant tokens must be 1
    asst_sum = sum(t.assistant_token_count for t in traj.turns)
    obs_sum = sum(t.obs_token_count for t in traj.turns)
    assert obs_sum > 0
    assert sum(traj.assistant_mask) == asst_sum
    assert len(traj.token_ids) == traj.prompt_len + asst_sum + obs_sum

    # Second turn prefix must begin identically with turn-0 tokens + turn-0 obs tokens (zero drift)
    assert len(backend.seen_prefixes) == 2
    assert backend.seen_prefixes[1][: len(backend.seen_prefixes[0])] == backend.seen_prefixes[0]


def test_dataset_group_leakage_isolation_and_roundtrip(tmp_path: Path) -> None:
    bundle = build_benchmark_dataset(seed=123, total_id=80, total_ood=16)
    seen_groups: set[str] = set()
    for split_name, tasks in bundle.splits.items():
        assert len(tasks) > 0
        split_groups = {t.group_id for t in tasks}
        assert len(seen_groups & split_groups) == 0, f"Leakage in {split_name}"
        seen_groups |= split_groups

    saved_dir = save_dataset_bundle(bundle, tmp_path / "ds")
    reloaded = load_dataset_bundle(saved_dir)
    assert set(reloaded.splits.keys()) == set(bundle.splits.keys())
    assert len(reloaded.corpus) == len(bundle.corpus)
