"""RAGEN-2 (arXiv:2604.06268) reasoning-collapse diagnostics and SNR-aware prompt filtering."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from agent_rl_flywheel.protocol import TokenizedTrajectory, THINK_RE


def compute_token_entropy(logits: torch.Tensor, assistant_mask: torch.Tensor) -> float:
    """Compute average per-token Shannon entropy strictly over trainable assistant tokens."""
    shift_logits = logits[:, :-1, :].float()
    shift_mask = assistant_mask[:, 1:].float()
    denom = float(shift_mask.sum().item())
    if denom <= 0.0:
        return 0.0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(((entropy * shift_mask).sum() / denom).detach().item())


def _extract_reasoning_text(traj: TokenizedTrajectory) -> str:
    """Extract `<think>` content across turns, falling back to full assistant turns."""
    chunks: list[str] = []
    for turn in traj.turns:
        if turn.parsed.think:
            chunks.append(turn.parsed.think)
        elif turn.assistant_text:
            found = THINK_RE.findall(turn.assistant_text)
            if found:
                chunks.extend(found)
            else:
                chunks.append(turn.assistant_text)
    if not chunks and traj.raw_text:
        chunks.extend(THINK_RE.findall(traj.raw_text))
    return " ".join(chunks).strip()


def _text_features(text: str) -> Counter[str]:
    """Mixed word + CJK/ASCII bigram features for lightweight online MI proxy estimation."""
    feats: Counter[str] = Counter()
    cleaned = re.sub(r"\s+", " ", text.lower()).strip()
    if not cleaned:
        return feats
    for word in re.findall(r"[a-zA-Z0-9_.-]+", cleaned):
        feats[f"w:{word}"] += 1
    compact = cleaned.replace(" ", "")
    for i in range(max(0, len(compact) - 1)):
        feats[f"b:{compact[i : i + 2]}"] += 1
    if not feats:
        feats[cleaned] += 1
    return feats


def compute_online_mi_proxy(
    trajectories: list[TokenizedTrajectory],
    collapse_threshold: float = 0.15,
    entropy_healthy_min: float = 0.30,
    observed_entropy: float | None = None,
) -> dict[str, float | bool]:
    """Estimate cross-input Mutual Information proxy via Jensen-Shannon divergence & cosine gap.

    In RAGEN-2, standard token entropy only measures within-input diversity and misses
    *template collapse* (high entropy boilerplate repeated across distinct inputs).
    """
    by_prompt: dict[str, Counter[str]] = {}
    for traj in trajectories:
        text = _extract_reasoning_text(traj)
        feats = _text_features(text)
        bucket = by_prompt.setdefault(traj.task_id, Counter())
        bucket.update(feats)

    prompt_keys = sorted(by_prompt.keys())
    if len(prompt_keys) < 2:
        return {
            "mi_proxy": 0.0,
            "js_divergence": 0.0,
            "cross_prompt_diversity": 0.0,
            "prompt_count": float(len(prompt_keys)),
            "template_collapse_detected": False,
        }

    vocab = sorted({tok for cnt in by_prompt.values() for tok in cnt})
    if not vocab:
        return {
            "mi_proxy": 0.0,
            "js_divergence": 0.0,
            "cross_prompt_diversity": 0.0,
            "prompt_count": float(len(prompt_keys)),
            "template_collapse_detected": True,
        }

    tok_to_idx = {tok: i for i, tok in enumerate(vocab)}
    mat = np.zeros((len(prompt_keys), len(vocab)), dtype=np.float64)
    for r, pk in enumerate(prompt_keys):
        for tok, c in by_prompt[pk].items():
            mat[r, tok_to_idx[tok]] = float(c)

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    probs = mat / row_sums
    marginal = probs.mean(axis=0)

    # Jensen-Shannon divergence normalized by ln(N) -> [0, 1]
    js = 0.0
    eps = 1e-12
    for r in range(probs.shape[0]):
        p = probs[r]
        nonzero = p > eps
        js += float(np.sum(p[nonzero] * np.log((p[nonzero] + eps) / (marginal[nonzero] + eps)))) / probs.shape[0]
    norm_factor = math.log(max(2, len(prompt_keys)))
    js_norm = min(1.0, max(0.0, js / norm_factor))

    # Mean pairwise (1 - cosine) across prompt centroids
    l2 = np.linalg.norm(probs, axis=1, keepdims=True)
    l2 = np.where(l2 > 0.0, l2, 1.0)
    unit = probs / l2
    cos_mat = unit @ unit.T
    n = cos_mat.shape[0]
    off_diag = (float(np.sum(cos_mat)) - float(np.trace(cos_mat))) / max(1, n * (n - 1))
    cross_div = min(1.0, max(0.0, 1.0 - off_diag))

    mi_proxy = round(0.5 * js_norm + 0.5 * cross_div, 6)
    entropy_ok = observed_entropy is None or observed_entropy >= entropy_healthy_min
    collapse_detected = bool(mi_proxy < collapse_threshold and entropy_ok)

    return {
        "mi_proxy": mi_proxy,
        "js_divergence": round(js_norm, 6),
        "cross_prompt_diversity": round(cross_div, 6),
        "prompt_count": float(len(prompt_keys)),
        "template_collapse_detected": collapse_detected,
    }


def snr_aware_group_filter(
    trajectories: list[TokenizedTrajectory],
    min_std: float = 0.05,
    top_k_ratio: float = 0.75,
) -> tuple[list[TokenizedTrajectory], dict[str, float]]:
    """Filter low-reward-variance prompt groups to preserve high-SNR task gradients (RAGEN-2)."""
    by_task: dict[str, list[TokenizedTrajectory]] = {}
    for traj in trajectories:
        by_task.setdefault(traj.task_id, []).append(traj)

    if not by_task:
        return [], {"snr_keep_ratio": 0.0, "mean_group_std": 0.0, "kept_groups": 0.0, "total_groups": 0.0}

    scored: list[tuple[float, str, list[TokenizedTrajectory]]] = []
    for tid, grp in by_task.items():
        rewards = [t.reward for t in grp]
        std = float(np.std(rewards)) if len(rewards) > 1 else 0.0
        scored.append((std, tid, grp))

    high_snr = [item for item in scored if item[0] >= min_std]
    high_snr.sort(key=lambda x: (-x[0], x[1]))

    if not high_snr:
        return [], {
            "snr_keep_ratio": 0.0,
            "mean_group_std": round(float(np.mean([s[0] for s in scored])), 6),
            "kept_groups": 0.0,
            "total_groups": float(len(scored)),
        }

    if top_k_ratio >= 1.0:
        kept = high_snr
    else:
        keep_n = max(1, math.ceil(len(high_snr) * max(0.0, top_k_ratio)))
        kept = high_snr[:keep_n]

    filtered: list[TokenizedTrajectory] = []
    for _, _, grp in kept:
        filtered.extend(grp)

    return filtered, {
        "snr_keep_ratio": round(len(kept) / len(scored), 6),
        "mean_group_std": round(float(np.mean([s[0] for s in kept])), 6),
        "kept_groups": float(len(kept)),
        "total_groups": float(len(scored)),
    }
