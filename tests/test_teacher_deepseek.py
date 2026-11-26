"""Tests for mocked DeepSeek-V4.1-Flash synthesis and prefix-preserving OPD + DPO pairs."""

from __future__ import annotations

from pathlib import Path

from agent_rl_flywheel.backends.tiny_local import SimpleByteTokenizer
from agent_rl_flywheel.data.generator import build_benchmark_dataset
from agent_rl_flywheel.data.teacher_deepseek import (
    DeepSeekTeacherClient,
    build_prefix_preserving_opd_and_dpo,
    synthesize_coldstart_sft,
)
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness
from agent_rl_flywheel.env.proxy import AgentProxyGateway, TurnSample
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


class _PrefixFailStudent:
    """Calls valid hybrid_search on turn 0, then gives wrong answer on turn 1."""

    def __init__(self, first_turn_tool: str) -> None:
        self.tok = SimpleByteTokenizer()
        self.first_turn_tool = first_turn_tool
        self.step = 0

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tok.decode(token_ids)

    def generate_with_logprobs(self, prompt_token_ids: list[int], max_new_tokens: int, temperature: float = 0.7) -> TurnSample:
        text = self.first_turn_tool if self.step == 0 else "<think>wrong guess</think><answer>999999</answer>"
        self.step += 1
        ids = self.encode(text)[:max_new_tokens]
        return TurnSample(text=self.decode(ids), token_ids=ids, logprobs=[-0.3] * len(ids))


def test_deepseek_teacher_coldstart_and_prefix_opd_dpo(tmp_path: Path) -> None:
    bundle = build_benchmark_dataset(seed=11, total_id=20, total_ood=4)
    tasks = bundle.splits["train_coldstart"][:4]
    tasks_by_prompt = {t.prompt: t for t in tasks}

    def _mock_ds(messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        t = tasks_by_prompt[messages[1]["content"]]
        asst_cnt = sum(1 for m in messages if m["role"] == "assistant")
        return t.gold_turns[min(asst_cnt, len(t.gold_turns) - 1)]["assistant_text"]

    client = DeepSeekTeacherClient(api_key="unit-test-key", mock_fn=_mock_ds)
    assert client.model == "deepseek-flash"
    assert client.base_url == "https://api.deepseek.com"

    sympy_tool = SymPyTool()
    retriever = HybridRetriever(bundle.corpus)
    harness = MultiTurnAgentHarness(sympy_tool=sympy_tool, retriever=retriever)
    tok = SimpleByteTokenizer()

    cache_file = tmp_path / "coldstart.jsonl"
    accepted = synthesize_coldstart_sft(tasks, client, harness, tok.encode, out_path=cache_file)
    assert len(accepted) == 4
    # Resumable reload should recover the same 4 trajectories cleanly
    reloaded = synthesize_coldstart_sft(tasks, client, harness, tok.encode, out_path=cache_file)
    assert len(reloaded) == 4

    # Prefix-preserving correction on a RAG task where turn 0 tool succeeded and turn 1 failed
    rag_task = next(t for t in tasks if t.task_type == "rag")
    stu_backend = _PrefixFailStudent(rag_task.gold_turns[0]["assistant_text"])
    failed_traj = harness.run_rollout(rag_task, AgentProxyGateway(stu_backend))
    assert failed_traj.reward_breakdown["acc"] == 0.0
    assert failed_traj.turns[0].tool_success is True

    out = build_prefix_preserving_opd_and_dpo(
        tasks_by_id={rag_task.task_id: rag_task},
        failed_trajectories=[failed_traj],
        client=client,
        harness=harness,
        encode_fn=tok.encode,
    )
    assert len(out["opd_trajectories"]) == 1
    assert len(out["dpo_pairs"]) == 1

    opd_traj = out["opd_trajectories"][0]
    pair = out["dpo_pairs"][0]
    # Prefix turn 0 must be masked 0; corrective turn 1 must be masked 1
    prefix_turn0_len = len(tok.encode(rag_task.gold_turns[0]["assistant_text"]))
    prompt_len = opd_traj.prompt_len
    assert all(m == 0 for m in opd_traj.assistant_mask[: prompt_len + prefix_turn0_len])
    assert sum(opd_traj.assistant_mask) > 0
    assert pair.chosen_token_ids[: prompt_len + prefix_turn0_len] == pair.rejected_token_ids[: prompt_len + prefix_turn0_len]
    assert pair.chosen_text != pair.rejected_text
