"""Tests for AST-sandboxed SymPy tool, BM25+RRF retriever, and RLVR anti-hacking verifier."""

from __future__ import annotations

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.env.verifier import RLVRVerifier
from agent_rl_flywheel.protocol import TokenizedTrajectory, ToolCall, TurnRecord, parse_assistant_turn
from agent_rl_flywheel.tools.hybrid_search import DocumentChunk, HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


def test_sympy_sandbox_blocks_unsafe_and_checks_equivalence() -> None:
    tool = SymPyTool()
    assert tool.evaluate("__import__('os').system('echo hi')")["ok"] is False
    assert tool.evaluate("eval('1+1')")["ok"] is False
    assert tool.evaluate("2**1000")["ok"] is False
    assert tool.evaluate("2**3**4")["ok"] is False

    ok_res = tool.evaluate("(3 + 5) * 4 / 2")
    assert ok_res["ok"] is True
    assert abs(float(ok_res["numeric"]) - 16.0) < 1e-9

    assert tool.are_equivalent("1/2", "0.5") is True
    assert tool.are_equivalent("(x + 1)^2", "x**2 + 2*x + 1") is True
    assert tool.are_equivalent("42 (doc_id_0001)", "42") is True
    assert tool.are_equivalent("41", "42") is False


def test_hybrid_retriever_bm25_rrf_hits_target_doc() -> None:
    chunks = [
        DocumentChunk("doc_0", "g0", "缓存阈值手册 MANUAL-0", "标准阈值参数为 73，用于控制缓存。"),
        DocumentChunk("doc_1", "g1", "声纹水印参数 MANUAL-1", "水印帧长规定为 0.5 秒。"),
        DocumentChunk("doc_2", "g2", "推理吞吐配置 MANUAL-2", "最大并发序列限制为 16。"),
    ]
    retriever = HybridRetriever(chunks)
    hits = retriever.search("MANUAL-0 缓存阈值 标准阈值", top_k=2)
    assert len(hits) == 2
    assert hits[0]["doc_id"] == "doc_0"


def test_rlvr_verifier_anti_reward_hacking_and_overlong() -> None:
    tool = SymPyTool()
    verifier = RLVRVerifier(sympy_tool=tool, soft_max_tokens=100, hard_max_tokens=200, overlong_penalty_max=0.30)
    task = TaskItem(
        task_id="t1",
        group_id="g1",
        task_type="math",
        split="val",
        prompt="计算 6*7",
        gold_answer="42",
        requires_tool=True,
    )

    # Case 1: spinning tool repeatedly without getting answer right -> capped & penalized
    bad_turn1 = TurnRecord(
        turn_idx=0,
        assistant_text="<think>try</think><tool_call>{\"name\":\"sympy_eval\",\"arguments\":{\"expr\":\"6*7\"}}</tool_call>",
        parsed=parse_assistant_turn("<think>try</think><tool_call>{\"name\":\"sympy_eval\",\"arguments\":{\"expr\":\"6*7\"}}</tool_call>"),
        tool_success=True,
        is_duplicate_call=False,
    )
    bad_turn2 = TurnRecord(
        turn_idx=1,
        assistant_text=bad_turn1.assistant_text,
        parsed=bad_turn1.parsed,
        tool_success=True,
        is_duplicate_call=True,
    )
    bad_turn3 = TurnRecord(
        turn_idx=2,
        assistant_text="<think>guess</think><answer>99</answer>",
        parsed=parse_assistant_turn("<think>guess</think><answer>99</answer>"),
    )
    traj_spin = TokenizedTrajectory(
        task_id="t1",
        group_id="g1",
        token_ids=[3] * 160,  # 150 response tokens -> triggers overlong shaping
        assistant_mask=[0] * 10 + [1] * 150,
        old_logprobs=[0.0] * 160,
        prompt_len=10,
        turn_count=3,
        tool_calls=[ToolCall("sympy_eval", {"expr": "6*7"})] * 2,
        turns=[bad_turn1, bad_turn2, bad_turn3],
        final_answer="99",
    )
    res_bad = verifier.score(task, traj_spin)
    assert res_bad.acc == 0.0
    assert res_bad.tool <= 0.10 + 1e-9
    assert "tool_spin_without_acc" in res_bad.hacking_flags
    assert "overlong_response" in res_bad.hacking_flags
    assert res_bad.penalty > 0.15

    # Case 2: clean tool + exact answer + valid format -> max 0.90 reward
    good_turn1 = bad_turn1
    good_turn2 = TurnRecord(
        turn_idx=1,
        assistant_text="<think>done</think><answer>42</answer>",
        parsed=parse_assistant_turn("<think>done</think><answer>42</answer>"),
    )
    traj_good = TokenizedTrajectory(
        task_id="t1",
        group_id="g1",
        token_ids=[3] * 50,
        assistant_mask=[0] * 10 + [1] * 40,
        old_logprobs=[0.0] * 50,
        prompt_len=10,
        turn_count=2,
        tool_calls=[ToolCall("sympy_eval", {"expr": "6*7"})],
        turns=[good_turn1, good_turn2],
        final_answer="42",
    )
    res_good = verifier.score(task, traj_good)
    assert abs(res_good.total - 0.9) < 1e-6
    assert res_good.hacking_flags == []
