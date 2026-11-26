"""RLVR verifiable composite reward function with anti-reward-hacking gates."""

from __future__ import annotations

from dataclasses import dataclass, field

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.protocol import TokenizedTrajectory, extract_doc_ids
from agent_rl_flywheel.tools.sympy_tool import SymPyTool


@dataclass(frozen=True)
class RewardResult:
    """Breakdown of verifiable reward components for a trajectory."""

    total: float
    acc: float
    tool: float
    fmt: float
    penalty: float
    hacking_flags: list[str] = field(default_factory=list)


class RLVRVerifier:
    """Verifiable reward scorer with strict anti-reward-hacking protections."""

    def __init__(
        self,
        sympy_tool: SymPyTool | None = None,
        soft_max_tokens: int = 1400,
        hard_max_tokens: int = 2048,
        overlong_penalty_max: float = 0.30,
    ) -> None:
        self.sympy_tool = sympy_tool or SymPyTool()
        self.soft_max_tokens = soft_max_tokens
        self.hard_max_tokens = hard_max_tokens
        self.overlong_penalty_max = overlong_penalty_max

    def _check_accuracy(self, task: TaskItem, trajectory: TokenizedTrajectory) -> float:
        ans = trajectory.final_answer
        if not ans:
            return 0.0
        answer_ok = self.sympy_tool.are_equivalent(ans, task.gold_answer)
        if not answer_ok:
            # Normalized string fallback for non-numeric answers
            norm_pred = ans.strip().lower()
            norm_gold = task.gold_answer.strip().lower()
            answer_ok = norm_gold in norm_pred
        if not answer_ok:
            return 0.0

        if task.required_doc_ids:
            cited = set(extract_doc_ids(ans))
            # Also accept doc_ids grounded via hybrid_search observations and cited in answer/final think
            if not cited and trajectory.turns:
                cited = set(extract_doc_ids(trajectory.turns[-1].assistant_text))
            if not any(req in cited for req in task.required_doc_ids):
                return 0.0
        return 1.0

    def score(self, task: TaskItem, trajectory: TokenizedTrajectory) -> RewardResult:
        """Compute `R = 0.6 * R_acc + 0.2 * R_tool + 0.1 * R_format - penalty`."""
        flags: list[str] = []
        acc = self._check_accuracy(task=task, trajectory=trajectory)

        # Format compliance: every turn valid + terminal <answer>
        all_turns_valid = bool(trajectory.turns) and all(t.parsed.format_valid for t in trajectory.turns)
        has_terminal_answer = trajectory.final_answer is not None
        fmt = 1.0 if (all_turns_valid and has_terminal_answer) else 0.0

        # Anti-reward-hacking tool score
        successful_calls = sum(1 for t in trajectory.turns if t.tool_success is True)
        duplicate_calls = sum(1 for t in trajectory.turns if t.is_duplicate_call)
        if duplicate_calls > 0:
            flags.append(f"redundant_tool_calls_{duplicate_calls}")

        if not task.requires_tool:
            tool = 1.0 if duplicate_calls == 0 else 0.5
        else:
            if successful_calls == 0:
                tool = 0.0
                if acc > 0.0:
                    flags.append("guessed_without_tool")
            elif acc < 1.0:
                # Cap tool reward when answer is wrong to block blind tool-spinning
                tool = max(0.0, 0.25 - 0.15 * duplicate_calls)
                flags.append("tool_spin_without_acc")
            else:
                tool = 1.0 if duplicate_calls == 0 else 0.5

        # Overlong + duplicate loop penalty (DAPO Overlong Reward Shaping)
        resp_tokens = trajectory.response_len
        overlong_pen = 0.0
        if resp_tokens > self.soft_max_tokens:
            span = max(1, self.hard_max_tokens - self.soft_max_tokens)
            ratio = min(1.0, (resp_tokens - self.soft_max_tokens) / span)
            overlong_pen = ratio * self.overlong_penalty_max
            flags.append("overlong_response")

        loop_pen = 0.15 * duplicate_calls
        penalty = round(overlong_pen + loop_pen, 6)

        total = round(0.6 * acc + 0.2 * tool + 0.1 * fmt - penalty, 6)
        return RewardResult(
            total=total,
            acc=acc,
            tool=tool,
            fmt=fmt,
            penalty=penalty,
            hacking_flags=flags,
        )
