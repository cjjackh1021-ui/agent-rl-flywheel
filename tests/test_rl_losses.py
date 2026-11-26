"""Unit tests for pure-PyTorch SFT, DPO, Token-OPD, GRPO, and DAPO losses."""

from __future__ import annotations

import torch

from agent_rl_flywheel.rl.losses import (
    compute_group_advantages,
    dapo_loss,
    dpo_loss,
    grpo_loss,
    masked_sft_loss,
    token_level_opd_loss,
)


def test_masked_sft_ignores_obs_tokens_completely() -> None:
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 5, 6, 7, 8, 9]], dtype=torch.long)
    # Only positions 4 and 5 (shifted indices 3 and 4) are trainable assistant tokens
    mask = torch.tensor([[0, 0, 0, 0, 1, 1]], dtype=torch.long)

    logits_a = torch.randn(1, 6, 16, requires_grad=True)
    loss_a = masked_sft_loss(logits_a, token_ids, mask)
    loss_a.backward()
    assert logits_a.grad is not None
    # Shifted logits positions 0, 1, 2 predict token_ids 1, 2, 3 (all mask==0), so grad must be exact 0
    assert torch.allclose(logits_a.grad[:, :3, :], torch.zeros_like(logits_a.grad[:, :3, :]))
    assert float(logits_a.grad[:, 3:5, :].abs().sum().item()) > 0.0


def test_dpo_and_token_opd_losses() -> None:
    pol_c = torch.tensor([-1.0, -0.8], requires_grad=True)
    pol_r = torch.tensor([-2.5, -2.2], requires_grad=True)
    ref_c = torch.tensor([-1.5, -1.2])
    ref_r = torch.tensor([-1.4, -1.3])
    loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["reward_margin"] > 0.0
    assert metrics["pref_accuracy"] == 1.0

    student_logits = torch.randn(2, 6, 12, requires_grad=True)
    toks = torch.randint(0, 12, (2, 6))
    teacher_lp = torch.full((2, 6), -0.5)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0]])
    opd_l = token_level_opd_loss(student_logits, toks, teacher_lp, mask)
    opd_l.backward()
    assert torch.isfinite(opd_l)
    assert student_logits.grad is not None


def test_dapo_dynamic_sampling_clip_higher_and_empty_guard() -> None:
    # Group g0 has variance ([0.0, 0.9]); group g1 has zero variance ([0.1, 0.1])
    group_ids = ["g0", "g0", "g1", "g1"]
    rewards = torch.tensor([0.0, 0.9, 0.1, 0.1], dtype=torch.float32)
    adv, stds = compute_group_advantages(rewards, group_ids)
    assert stds["g0"] > 0.4 and stds["g1"] == 0.0
    assert abs(float(adv[0] + adv[1])) < 1e-5
    assert float(adv[2]) == 0.0 and float(adv[3]) == 0.0

    # Set ratio = 1.25 (> 1.2, < 1.28) on positive advantage row 1 to verify asymmetric Clip-Higher
    old_lp = torch.full((4, 6), -1.0)
    new_lp = torch.full((4, 5), -1.0, requires_grad=True)
    with torch.no_grad():
        delta_lp = torch.log(torch.tensor(1.25))
    new_lp_mod = new_lp + delta_lp  # keeps grad flow cleanly
    mask = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
        ],
        dtype=torch.long,
    )

    loss_dapo, m_dapo = dapo_loss(
        new_logprobs=new_lp_mod,
        old_logprobs=old_lp,
        advantages=adv,
        assistant_mask=mask,
        group_ids=group_ids,
        rewards=rewards,
        clip_eps_low=0.2,
        clip_eps_high=0.28,
        dynamic_sampling=True,
    )
    assert m_dapo["valid_group_ratio"] == 0.5
    assert m_dapo["clip_frac_high"] == 0.0  # ratio=1.25 is below 1.28 upper bound
    loss_dapo.backward()
    assert new_lp.grad is not None
    # Zero-variance group g1 rows (indices 2, 3) must receive exact zero gradient
    assert torch.allclose(new_lp.grad[2:], torch.zeros_like(new_lp.grad[2:]))

    # Compare with symmetric GRPO clip_eps=0.2 where ratio=1.25 IS clipped
    loss_grpo, m_grpo = grpo_loss(new_lp_mod.detach().requires_grad_(True), old_lp, adv, mask, clip_eps=0.2, kl_beta=0.0)
    assert torch.isfinite(loss_grpo)
    assert m_grpo["clip_frac"] > 0.0

    # When ALL groups have zero std, DAPO returns safe 0 loss without NaN
    zero_rewards = torch.tensor([0.5, 0.5, 0.2, 0.2], dtype=torch.float32)
    zero_adv, _ = compute_group_advantages(zero_rewards, group_ids)
    safe_loss, safe_m = dapo_loss(
        new_logprobs=new_lp,
        old_logprobs=old_lp,
        advantages=zero_adv,
        assistant_mask=mask,
        group_ids=group_ids,
        rewards=zero_rewards,
        dynamic_sampling=True,
    )
    assert float(safe_loss.item()) == 0.0
    assert safe_m["valid_group_ratio"] == 0.0
