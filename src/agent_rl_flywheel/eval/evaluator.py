"""Evaluation suite across locked ID and OOD splits with RAGEN-2 MI diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.env.harness import MultiTurnAgentHarness
from agent_rl_flywheel.env.proxy import AgentProxyGateway
from agent_rl_flywheel.rl.ragen2 import compute_online_mi_proxy


@dataclass(frozen=True)
class EvalResult:
    """Aggregated evaluation metrics for one experiment stage and split."""

    stage_name: str
    split_name: str
    num_tasks: int
    pass_at_1: float
    tool_use_rate: float
    format_rate: float
    mean_reward: float
    avg_turns: float
    avg_response_tokens: float
    mi_proxy: float
    template_collapse_detected: bool


def evaluate_policy(
    stage_name: str,
    split_name: str,
    tasks: list[TaskItem],
    backend: Any,
    harness: MultiTurnAgentHarness,
    temperature: float = 0.0,
) -> EvalResult:
    """Evaluate policy deterministically on a locked split without mutating weights."""
    if not tasks:
        return EvalResult(
            stage_name=stage_name,
            split_name=split_name,
            num_tasks=0,
            pass_at_1=0.0,
            tool_use_rate=0.0,
            format_rate=0.0,
            mean_reward=0.0,
            avg_turns=0.0,
            avg_response_tokens=0.0,
            mi_proxy=0.0,
            template_collapse_detected=False,
        )

    gateway = AgentProxyGateway(backend)
    trajectories = [
        harness.run_rollout(task=t, gateway=gateway, temperature=temperature) for t in tasks
    ]
    n = len(trajectories)
    pass_at_1 = sum(r.reward_breakdown.get("acc", 0.0) for r in trajectories) / n
    tool_rate = sum(1.0 for r in trajectories if r.tool_calls) / n
    fmt_rate = sum(r.reward_breakdown.get("format", 0.0) for r in trajectories) / n
    mean_reward = sum(r.reward for r in trajectories) / n
    avg_turns = sum(r.turn_count for r in trajectories) / n
    avg_resp = sum(r.response_len for r in trajectories) / n
    mi_info = compute_online_mi_proxy(trajectories)

    return EvalResult(
        stage_name=stage_name,
        split_name=split_name,
        num_tasks=n,
        pass_at_1=round(pass_at_1, 4),
        tool_use_rate=round(tool_rate, 4),
        format_rate=round(fmt_rate, 4),
        mean_reward=round(mean_reward, 4),
        avg_turns=round(avg_turns, 2),
        avg_response_tokens=round(avg_resp, 1),
        mi_proxy=float(mi_info["mi_proxy"]),
        template_collapse_detected=bool(mi_info["template_collapse_detected"]),
    )


def render_ablation_markdown(results: list[EvalResult]) -> str:
    """Render evaluation rows into a concise GitHub Markdown ablation table."""
    lines = [
        "# Ablation Report (Locked ID & OOD Evaluation)",
        "",
        "| Stage | Split | N | Pass@1 | Tool Rate | Format Rate | Mean Reward | Avg Turns | Avg Resp Tokens | MI Proxy | Collapse? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.stage_name} | {r.split_name} | {r.num_tasks} | {r.pass_at_1:.4f} | "
            f"{r.tool_use_rate:.4f} | {r.format_rate:.4f} | {r.mean_reward:.4f} | "
            f"{r.avg_turns:.2f} | {r.avg_response_tokens:.1f} | {r.mi_proxy:.4f} | "
            f"{'YES' if r.template_collapse_detected else 'NO'} |"
        )
    return "\n".join(lines) + "\n"
