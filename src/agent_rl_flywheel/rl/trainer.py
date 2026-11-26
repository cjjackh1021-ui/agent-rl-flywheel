"""Unified trainer orchestrating Masked SFT, On-Policy DPO, Token-OPD, GRPO, and DAPO+RAGEN-2."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.data.teacher_deepseek import DPOPair
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness
from agent_rl_flywheel.env.proxy import AgentProxyGateway
from agent_rl_flywheel.protocol import TokenizedTrajectory
from agent_rl_flywheel.rl.losses import (
    compute_group_advantages,
    compute_per_token_logprobs,
    dapo_loss,
    dpo_loss,
    grpo_loss,
    masked_sft_loss,
    token_level_opd_loss,
)
from agent_rl_flywheel.rl.ragen2 import (
    compute_online_mi_proxy,
    compute_token_entropy,
    snr_aware_group_filter,
)


def _collate_sequences(
    seqs: list[list[int]],
    masks: list[list[int]],
    pad_id: int,
    float_seqs: list[list[float]] | None = None,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if max_seq_len is not None:
        seqs = [s[-max_seq_len:] for s in seqs]
        masks = [m[-max_seq_len:] for m in masks]
        if float_seqs is not None:
            float_seqs = [fs[-max_seq_len:] for fs in float_seqs]
    max_len = max(len(s) for s in seqs)
    b = len(seqs)
    ids_t = torch.full((b, max_len), pad_id, dtype=torch.long)
    mask_t = torch.zeros((b, max_len), dtype=torch.long)
    flt_t = torch.zeros((b, max_len), dtype=torch.float32) if float_seqs is not None else None
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask_t[i, : len(m)] = torch.tensor(m, dtype=torch.long)
        if flt_t is not None and float_seqs is not None:
            fs = float_seqs[i]
            flt_t[i, : len(fs)] = torch.tensor(fs, dtype=torch.float32)
    return ids_t, mask_t, flt_t


def _sequence_masked_logprob_sum(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_tok = compute_per_token_logprobs(logits, token_ids)
    shift_mask = mask[:, 1:].to(dtype=per_tok.dtype)
    return (per_tok * shift_mask).sum(dim=-1)


class AgentRLTrainer:
    """Runs SFT, On-Policy DPO, Token-OPD, GRPO, and DAPO+RAGEN-2 on either backend."""

    def __init__(
        self,
        backend: Any,
        harness: MultiTurnAgentHarness,
        out_dir: str | Path,
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.harness = harness
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.log_path = self.out_dir / "train_metrics.jsonl"

    def _append_log(self, row: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def train_sft(
        self,
        trajectories: list[TokenizedTrajectory],
        epochs: int = 1,
        batch_size: int = 4,
        lr: float = 1e-4,
        max_grad_norm: float = 1.0,
        stage_name: str = "sft",
    ) -> dict[str, float]:
        """Train with cross-entropy strictly masked to `assistant_mask == 1`."""
        if not trajectories:
            return {"loss": 0.0, "steps": 0.0}
        self.backend.model.train()
        opt = torch.optim.AdamW(self.backend.trainable_parameters(), lr=lr)
        losses: list[float] = []
        step = 0
        for epoch in range(epochs):
            pool = list(trajectories)
            self.rng.shuffle(pool)
            for i in range(0, len(pool), batch_size):
                batch = pool[i : i + batch_size]
                ids_t, mask_t, _ = _collate_sequences(
                    [t.token_ids for t in batch],
                    [t.assistant_mask for t in batch],
                    pad_id=self.backend.pad_token_id,
                    max_seq_len=getattr(self.backend, "max_seq_len", None),
                )
                ids_t = ids_t.to(self.backend.device)
                mask_t = mask_t.to(self.backend.device)
                opt.zero_grad(set_to_none=True)
                logits = self.backend.forward_logits(ids_t)
                loss = masked_sft_loss(logits=logits, token_ids=ids_t, assistant_mask=mask_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backend.trainable_parameters(), max_grad_norm)
                opt.step()
                val = float(loss.detach().item())
                losses.append(val)
                step += 1
                self._append_log({"stage": stage_name, "epoch": epoch, "step": step, "loss": val})
        return {"loss": round(sum(losses) / max(1, len(losses)), 6), "steps": float(step)}

    def train_dpo(
        self,
        pairs: list[DPOPair],
        epochs: int = 1,
        batch_size: int = 2,
        lr: float = 5e-5,
        beta: float = 0.1,
        max_grad_norm: float = 1.0,
        stage_name: str = "on_policy_dpo",
    ) -> dict[str, float]:
        """Train with DPO on prefix-preserving student-vs-teacher preference pairs."""
        if not pairs:
            return {"loss": 0.0, "reward_margin": 0.0, "steps": 0.0}
        self.backend.sync_reference()
        self.backend.model.train()
        opt = torch.optim.AdamW(self.backend.trainable_parameters(), lr=lr)
        losses: list[float] = []
        margins: list[float] = []
        step = 0
        for epoch in range(epochs):
            pool = list(pairs)
            self.rng.shuffle(pool)
            for i in range(0, len(pool), batch_size):
                batch = pool[i : i + batch_size]
                c_ids, c_mask, _ = _collate_sequences(
                    [p.chosen_token_ids for p in batch],
                    [p.chosen_mask for p in batch],
                    pad_id=self.backend.pad_token_id,
                    max_seq_len=getattr(self.backend, "max_seq_len", None),
                )
                r_ids, r_mask, _ = _collate_sequences(
                    [p.rejected_token_ids for p in batch],
                    [p.rejected_mask for p in batch],
                    pad_id=self.backend.pad_token_id,
                    max_seq_len=getattr(self.backend, "max_seq_len", None),
                )
                c_ids, c_mask = c_ids.to(self.backend.device), c_mask.to(self.backend.device)
                r_ids, r_mask = r_ids.to(self.backend.device), r_mask.to(self.backend.device)

                opt.zero_grad(set_to_none=True)
                pol_c = _sequence_masked_logprob_sum(self.backend.forward_logits(c_ids), c_ids, c_mask)
                pol_r = _sequence_masked_logprob_sum(self.backend.forward_logits(r_ids), r_ids, r_mask)
                with torch.no_grad():
                    ref_c = _sequence_masked_logprob_sum(self.backend.forward_logits(c_ids, use_ref=True), c_ids, c_mask)
                    ref_r = _sequence_masked_logprob_sum(self.backend.forward_logits(r_ids, use_ref=True), r_ids, r_mask)

                loss, metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backend.trainable_parameters(), max_grad_norm)
                opt.step()
                val = float(loss.detach().item())
                losses.append(val)
                margins.append(metrics["reward_margin"])
                step += 1
                self._append_log({"stage": stage_name, "epoch": epoch, "step": step, "loss": val, **metrics})
        return {
            "loss": round(sum(losses) / max(1, len(losses)), 6),
            "reward_margin": round(sum(margins) / max(1, len(margins)), 6),
            "steps": float(step),
        }

    def train_token_opd(
        self,
        trajectories: list[TokenizedTrajectory],
        teacher_token_logprobs: list[list[float]],
        epochs: int = 1,
        batch_size: int = 2,
        lr: float = 1e-4,
        temperature: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        """Same-vocab token-level On-Policy Distillation using cached teacher logprobs."""
        if not trajectories:
            return {"loss": 0.0, "steps": 0.0}
        self.backend.model.train()
        opt = torch.optim.AdamW(self.backend.trainable_parameters(), lr=lr)
        losses: list[float] = []
        step = 0
        paired = list(zip(trajectories, teacher_token_logprobs))
        for epoch in range(epochs):
            self.rng.shuffle(paired)
            for i in range(0, len(paired), batch_size):
                chunk = paired[i : i + batch_size]
                ids_t, mask_t, t_lp = _collate_sequences(
                    [t.token_ids for t, _ in chunk],
                    [t.assistant_mask for t, _ in chunk],
                    pad_id=self.backend.pad_token_id,
                    float_seqs=[lp for _, lp in chunk],
                    max_seq_len=getattr(self.backend, "max_seq_len", None),
                )
                assert t_lp is not None
                ids_t = ids_t.to(self.backend.device)
                mask_t = mask_t.to(self.backend.device)
                t_lp = t_lp.to(self.backend.device)

                opt.zero_grad(set_to_none=True)
                logits = self.backend.forward_logits(ids_t)
                loss = token_level_opd_loss(
                    student_logits=logits,
                    token_ids=ids_t,
                    teacher_token_logprobs=t_lp,
                    assistant_mask=mask_t,
                    temperature=temperature,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backend.trainable_parameters(), max_grad_norm)
                opt.step()
                val = float(loss.detach().item())
                losses.append(val)
                step += 1
                self._append_log({"stage": "token_opd", "epoch": epoch, "step": step, "loss": val})
        return {"loss": round(sum(losses) / max(1, len(losses)), 6), "steps": float(step)}

    def train_rl(
        self,
        tasks: list[TaskItem],
        algo: str = "dapo",
        use_ragen2: bool = True,
        steps: int = 4,
        prompts_per_step: int = 2,
        group_size: int = 4,
        lr: float = 2e-5,
        clip_eps: float = 0.2,
        clip_eps_high: float = 0.28,
        kl_beta: float = 0.02,
        dynamic_sampling: bool = True,
        snr_min_reward_std: float = 0.05,
        snr_top_k_ratio: float = 0.75,
        mi_collapse_threshold: float = 0.15,
        max_grad_norm: float = 1.0,
        temperature: float = 0.8,
    ) -> list[dict[str, Any]]:
        """Run multi-turn RL (`grpo` or `dapo`) with optional RAGEN-2 MI/SNR anti-collapse."""
        if not tasks:
            return []
        self.backend.sync_reference()
        opt = torch.optim.AdamW(self.backend.trainable_parameters(), lr=lr)
        gateway = AgentProxyGateway(self.backend)
        history: list[dict[str, Any]] = []

        for step_idx in range(1, steps + 1):
            batch_tasks = [self.rng.choice(tasks) for _ in range(prompts_per_step)]
            rollouts: list[TokenizedTrajectory] = []
            for task in batch_tasks:
                for _ in range(group_size):
                    traj = self.harness.run_rollout(task=task, gateway=gateway, temperature=temperature)
                    rollouts.append(traj)

            all_rewards = [r.reward for r in rollouts]
            acc_rate = sum(r.reward_breakdown.get("acc", 0.0) for r in rollouts) / len(rollouts)
            tool_rate = sum(1.0 for r in rollouts if r.tool_calls) / len(rollouts)
            fmt_rate = sum(r.reward_breakdown.get("format", 0.0) for r in rollouts) / len(rollouts)
            avg_resp = sum(r.response_len for r in rollouts) / len(rollouts)

            active_rollouts = rollouts
            snr_metrics = {"snr_keep_ratio": 1.0, "mean_group_std": 0.0}
            if use_ragen2:
                filtered, snr_metrics = snr_aware_group_filter(
                    rollouts,
                    min_std=snr_min_reward_std,
                    top_k_ratio=snr_top_k_ratio,
                )
                if filtered:
                    active_rollouts = filtered

            ids_t, mask_t, old_lp_t = _collate_sequences(
                [r.token_ids for r in active_rollouts],
                [r.assistant_mask for r in active_rollouts],
                pad_id=self.backend.pad_token_id,
                float_seqs=[r.old_logprobs for r in active_rollouts],
                max_seq_len=getattr(self.backend, "max_seq_len", None),
            )
            assert old_lp_t is not None
            ids_t = ids_t.to(self.backend.device)
            mask_t = mask_t.to(self.backend.device)
            old_lp_t = old_lp_t.to(self.backend.device)
            rewards_t = torch.tensor([r.reward for r in active_rollouts], dtype=torch.float32, device=self.backend.device)
            gids = [r.task_id for r in active_rollouts]
            adv_t, _ = compute_group_advantages(rewards=rewards_t, group_ids=gids)

            self.backend.model.train()
            opt.zero_grad(set_to_none=True)
            logits = self.backend.forward_logits(ids_t)
            new_lp = compute_per_token_logprobs(logits, ids_t)
            entropy_val = compute_token_entropy(logits, mask_t)
            mi_info = compute_online_mi_proxy(
                rollouts,
                collapse_threshold=mi_collapse_threshold,
                observed_entropy=entropy_val,
            )

            ref_lp = None
            if kl_beta > 0.0:
                with torch.no_grad():
                    ref_logits = self.backend.forward_logits(ids_t, use_ref=True)
                    ref_lp = compute_per_token_logprobs(ref_logits, ids_t)

            if algo.lower() == "grpo":
                loss, algo_metrics = grpo_loss(
                    new_logprobs=new_lp,
                    old_logprobs=old_lp_t,
                    advantages=adv_t,
                    assistant_mask=mask_t,
                    ref_logprobs=ref_lp,
                    clip_eps=clip_eps,
                    kl_beta=kl_beta,
                )
            elif algo.lower() == "dapo":
                loss, algo_metrics = dapo_loss(
                    new_logprobs=new_lp,
                    old_logprobs=old_lp_t,
                    advantages=adv_t,
                    assistant_mask=mask_t,
                    group_ids=gids,
                    rewards=rewards_t,
                    clip_eps_low=clip_eps,
                    clip_eps_high=clip_eps_high,
                    dynamic_sampling=dynamic_sampling,
                    min_group_std=1e-6,
                    kl_beta=kl_beta,
                    ref_logprobs=ref_lp,
                )
            else:
                raise ValueError(f"Unsupported RL algorithm: {algo}")

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backend.trainable_parameters(), max_grad_norm)
                opt.step()

            step_row: dict[str, Any] = {
                "stage": f"{algo.lower()}{'_ragen2' if use_ragen2 else ''}",
                "step": step_idx,
                "loss": round(float(loss.detach().item()), 6),
                "mean_reward": round(sum(all_rewards) / len(all_rewards), 6),
                "acc_rate": round(acc_rate, 4),
                "tool_rate": round(tool_rate, 4),
                "format_rate": round(fmt_rate, 4),
                "avg_response_tokens": round(avg_resp, 2),
                "token_entropy": round(entropy_val, 6),
                "mi_proxy": mi_info["mi_proxy"],
                "template_collapse_detected": mi_info["template_collapse_detected"],
                "snr_keep_ratio": snr_metrics.get("snr_keep_ratio", 1.0),
                "retokenization_events": gateway.retokenization_events,
                **algo_metrics,
            }
            history.append(step_row)
            self._append_log(step_row)
        return history
