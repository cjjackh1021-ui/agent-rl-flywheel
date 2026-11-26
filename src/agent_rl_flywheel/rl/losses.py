"""Pure-PyTorch post-training losses: Masked SFT, DPO, Token-OPD, GRPO, and DAPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute next-token log probabilities of shape `[B, T-1]` aligned to `token_ids[:, 1:]`."""
    shift_logits = logits[:, :-1, :]
    shift_labels = token_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)


def masked_sft_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    assistant_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute cross-entropy strictly over `assistant_mask == 1` tokens (zero grad on `<obs>`)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = token_ids[:, 1:]
    shift_mask = assistant_mask[:, 1:].to(dtype=logits.dtype)
    token_ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view_as(shift_labels)
    denom = shift_mask.sum().clamp(min=1.0)
    return (token_ce * shift_mask).sum() / denom


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute standard Direct Preference Optimization (DPO) loss on sequence log-probs."""
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    metrics = {
        "chosen_reward": float(chosen_rewards.mean().detach().item()),
        "rejected_reward": float(rejected_rewards.mean().detach().item()),
        "reward_margin": float(logits.mean().detach().item()),
        "pref_accuracy": float((logits > 0).float().mean().detach().item()),
    }
    return loss, metrics


def token_level_opd_loss(
    student_logits: torch.Tensor,
    token_ids: torch.Tensor,
    teacher_token_logprobs: torch.Tensor,
    assistant_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Token-level On-Policy Distillation loss against same-vocab teacher token logprobs."""
    student_lp = compute_per_token_logprobs(student_logits / temperature, token_ids)
    shift_mask = assistant_mask[:, 1:].to(dtype=student_logits.dtype)
    target_teacher_lp = teacher_token_logprobs[:, 1:].to(dtype=student_logits.dtype)
    kl_surrogate = (student_lp - target_teacher_lp) ** 2 - target_teacher_lp
    nll_pull = -student_lp * torch.exp(target_teacher_lp).clamp(max=1.0)
    token_loss = 0.5 * kl_surrogate + 0.5 * nll_pull
    return (token_loss * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_ids: list[str],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute group-normalized advantages `(r - mean_g) / (std_g + eps)` and per-group stds."""
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    group_stds: dict[str, float] = {}
    buckets: dict[str, list[int]] = {}
    for idx, gid in enumerate(group_ids):
        buckets.setdefault(gid, []).append(idx)

    for gid, idxs in buckets.items():
        grp = rewards[idxs].float()
        std = float(grp.std(unbiased=False).item()) if len(idxs) > 1 else 0.0
        group_stds[gid] = std
        if len(idxs) <= 1 or std <= eps:
            advantages[idxs] = 0.0
        else:
            mean = grp.mean()
            advantages[idxs] = (grp - mean) / (std + eps)
    return advantages, group_stds


def grpo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    assistant_mask: torch.Tensor,
    ref_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Standard Group Relative Policy Optimization (sample-normalized clipped objective + KL)."""
    shift_mask = assistant_mask[:, 1:].to(dtype=new_logprobs.dtype)
    shift_old = old_logprobs[:, 1:].to(dtype=new_logprobs.dtype)
    adv = advantages.unsqueeze(-1).to(dtype=new_logprobs.dtype)

    log_ratio = (new_logprobs - shift_old).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_token = -torch.min(ratio * adv, clipped * adv)

    kl_val = 0.0
    if ref_logprobs is not None and kl_beta > 0.0:
        shift_ref = (
            ref_logprobs[:, 1:].to(dtype=new_logprobs.dtype)
            if ref_logprobs.shape[1] == old_logprobs.shape[1]
            else ref_logprobs.to(dtype=new_logprobs.dtype)
        )
        delta = (shift_ref - new_logprobs).clamp(min=-20.0, max=20.0)
        kl_token = torch.exp(delta) - delta - 1.0
        pg_token = pg_token + kl_beta * kl_token
        kl_val = float(((kl_token * shift_mask).sum() / shift_mask.sum().clamp(min=1.0)).detach().item())

    per_sample_denom = shift_mask.sum(dim=-1).clamp(min=1.0)
    per_sample_loss = (pg_token * shift_mask).sum(dim=-1) / per_sample_denom
    has_tokens = (shift_mask.sum(dim=-1) > 0).to(dtype=new_logprobs.dtype)
    loss = (per_sample_loss * has_tokens).sum() / has_tokens.sum().clamp(min=1.0)

    clip_frac = (
        (((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).to(dtype=new_logprobs.dtype) * shift_mask).sum()
        / shift_mask.sum().clamp(min=1.0)
    )
    return loss, {"kl": kl_val, "clip_frac": float(clip_frac.detach().item())}


def dapo_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    assistant_mask: torch.Tensor,
    group_ids: list[str],
    rewards: torch.Tensor,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.28,
    dynamic_sampling: bool = True,
    min_group_std: float = 1e-6,
    kl_beta: float = 0.0,
    ref_logprobs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DAPO: Asymmetric Clip-Higher + Dynamic Sampling (drop zero-variance groups) + Token-Level Loss."""
    _, group_stds = compute_group_advantages(rewards=rewards, group_ids=group_ids)
    if dynamic_sampling:
        valid_rows = [group_stds.get(gid, 0.0) > min_group_std for gid in group_ids]
    else:
        valid_rows = [True] * len(group_ids)

    valid_tensor = torch.tensor(valid_rows, device=new_logprobs.device, dtype=new_logprobs.dtype)
    valid_group_ratio = float(valid_tensor.mean().item()) if len(valid_rows) > 0 else 0.0

    shift_mask = assistant_mask[:, 1:].to(dtype=new_logprobs.dtype) * valid_tensor.unsqueeze(-1)
    total_tokens = shift_mask.sum()
    if float(total_tokens.detach().item()) <= 0.0:
        zero_loss = new_logprobs.sum() * 0.0
        return zero_loss, {
            "kl": 0.0,
            "clip_frac_high": 0.0,
            "clip_frac_low": 0.0,
            "valid_group_ratio": valid_group_ratio,
        }

    shift_old = old_logprobs[:, 1:].to(dtype=new_logprobs.dtype)
    adv = advantages.unsqueeze(-1).to(dtype=new_logprobs.dtype)

    log_ratio = (new_logprobs - shift_old).clamp(min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
    pg_token = -torch.min(ratio * adv, clipped * adv)

    kl_val = 0.0
    if ref_logprobs is not None and kl_beta > 0.0:
        shift_ref = (
            ref_logprobs[:, 1:].to(dtype=new_logprobs.dtype)
            if ref_logprobs.shape[1] == old_logprobs.shape[1]
            else ref_logprobs.to(dtype=new_logprobs.dtype)
        )
        delta = (shift_ref - new_logprobs).clamp(min=-20.0, max=20.0)
        kl_token = torch.exp(delta) - delta - 1.0
        pg_token = pg_token + kl_beta * kl_token
        kl_val = float(((kl_token * shift_mask).sum() / total_tokens).detach().item())

    # Token-level policy gradient normalization across all active tokens in batch
    loss = (pg_token * shift_mask).sum() / total_tokens
    clip_high = (((ratio > 1.0 + clip_eps_high).to(dtype=new_logprobs.dtype) * shift_mask).sum() / total_tokens)
    clip_low = (((ratio < 1.0 - clip_eps_low).to(dtype=new_logprobs.dtype) * shift_mask).sum() / total_tokens)
    return loss, {
        "kl": kl_val,
        "clip_frac_high": float(clip_high.detach().item()),
        "clip_frac_low": float(clip_low.detach().item()),
        "valid_group_ratio": valid_group_ratio,
    }
