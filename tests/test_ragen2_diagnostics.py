"""Tests for RAGEN-2 template-collapse MI diagnostics and SNR-aware group filtering."""

from __future__ import annotations

import torch

from agent_rl_flywheel.protocol import TokenizedTrajectory, TurnRecord, parse_assistant_turn
from agent_rl_flywheel.rl.ragen2 import (
    compute_online_mi_proxy,
    compute_token_entropy,
    snr_aware_group_filter,
)


def _dummy_traj(task_id: str, think_text: str, reward: float) -> TokenizedTrajectory:
    raw = f"<think>{think_text}</think><answer>1</answer>"
    rec = TurnRecord(turn_idx=0, assistant_text=raw, parsed=parse_assistant_turn(raw))
    return TokenizedTrajectory(
        task_id=task_id,
        group_id=task_id,
        token_ids=[3, 4, 5],
        assistant_mask=[0, 1, 1],
        old_logprobs=[0.0, -0.5, -0.5],
        prompt_len=1,
        turn_count=1,
        turns=[rec],
        final_answer="1",
        raw_text=raw,
        reward=reward,
    )


def test_ragen2_detects_template_collapse_despite_healthy_entropy() -> None:
    # Uniform high-entropy logits -> entropy looks healthy
    logits = torch.zeros(1, 5, 16)
    mask = torch.tensor([[0, 1, 1, 1, 1]])
    ent = compute_token_entropy(logits, mask)
    assert ent > 2.0

    # Different prompts emitting identical boilerplate reasoning -> template collapse!
    boilerplate = "First let us carefully read the problem and analyze every step methodically."
    collapsed = [
        _dummy_traj("task_0", boilerplate, 0.2),
        _dummy_traj("task_1", boilerplate, 0.2),
        _dummy_traj("task_2", boilerplate, 0.2),
    ]
    diag_collapsed = compute_online_mi_proxy(collapsed, collapse_threshold=0.15, observed_entropy=ent)
    assert diag_collapsed["mi_proxy"] < 0.05
    assert diag_collapsed["template_collapse_detected"] is True

    # Diverse input-specific reasoning across prompts -> healthy MI, no collapse
    diverse = [
        _dummy_traj("task_0", "Compute 12*8 using sympy_eval and verify linear root x=7.", 0.9),
        _dummy_traj("task_1", "Call hybrid_search for MANUAL-0042 cache prefix parameter doc_id.", 0.9),
        _dummy_traj("task_2", "Integrate cubic polynomial derivative at point t=5 via SymPy.", 0.9),
    ]
    diag_diverse = compute_online_mi_proxy(diverse, collapse_threshold=0.15, observed_entropy=ent)
    assert diag_diverse["mi_proxy"] > 0.30
    assert diag_diverse["template_collapse_detected"] is False


def test_snr_aware_group_filter_keeps_high_variance_prompts() -> None:
    trajs = [
        # Low-SNR group (all equal rewards)
        _dummy_traj("low_snr", "a", 0.2),
        _dummy_traj("low_snr", "a", 0.2),
        # Medium-SNR group
        _dummy_traj("mid_snr", "b", 0.2),
        _dummy_traj("mid_snr", "b", 0.6),
        # High-SNR group
        _dummy_traj("high_snr", "c", 0.0),
        _dummy_traj("high_snr", "c", 0.9),
    ]
    filtered, stats = snr_aware_group_filter(trajs, min_std=0.05, top_k_ratio=0.5)
    kept_ids = {t.task_id for t in filtered}
    assert kept_ids == {"high_snr"}
    assert stats["kept_groups"] == 1.0
    assert stats["total_groups"] == 3.0
