"""DeepSeek-V4.1-Flash teacher client, offline coldstart synthesis, and prefix-preserving OPD/DPO."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.env.harness import (
    DEFAULT_SYSTEM_PROMPT,
    ExactTokenBuffer,
    MultiTurnAgentHarness,
    build_prompt_and_gold_trajectory,
    format_task_prompt,
)
from agent_rl_flywheel.protocol import TokenizedTrajectory, parse_assistant_turn


class ChatCompletionClient(Protocol):
    """Minimal chat interface supporting both `openai.OpenAI` and offline mocks."""

    def complete(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class DPOPair:
    """On-policy DPO preference pair sharing an exact student rollout prefix."""

    task_id: str
    group_id: str
    chosen_token_ids: list[int]
    chosen_mask: list[int]
    rejected_token_ids: list[int]
    rejected_mask: list[int]
    chosen_text: str
    rejected_text: str


class DeepSeekTeacherClient:
    """OpenAI-compatible client targeting `deepseek-flash` (DeepSeek-V4.1-Flash)."""

    def __init__(
        self,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        api_key: str | None = None,
        api_key_env: str = "DEEPSEEK_API_KEY",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        mock_fn: Callable[[list[dict[str, str]], float, int], str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env, "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._mock_fn = mock_fn
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("DEEPSEEK_API_KEY is missing for remote DeepSeek teacher calls.")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Query DeepSeek-V4.1-Flash (or injected mock) with bounded retries."""
        if self._mock_fn is not None:
            return self._mock_fn(messages, temperature, max_tokens)

        client = self._get_client()
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return str(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"DeepSeekTeacherClient failed after {self.max_retries} attempts: {last_exc}")


def _run_teacher_continuation(
    task: TaskItem,
    prefix_turns: list[dict[str, str]],
    client: ChatCompletionClient,
    harness: MultiTurnAgentHarness,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> list[dict[str, str]]:
    """Run DeepSeek teacher from an arbitrary multi-turn prefix until `<answer>` or max_turns."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]
    full_turns: list[dict[str, str]] = []
    for rec in prefix_turns:
        asst = rec["assistant_text"]
        obs = rec.get("obs_text")
        messages.append({"role": "assistant", "content": asst})
        if obs:
            messages.append({"role": "user", "content": obs})
        full_turns.append({"assistant_text": asst, "obs_text": obs or ""})

    for _ in range(len(prefix_turns), harness.max_turns):
        reply = client.complete(messages=messages, temperature=temperature, max_tokens=max_tokens)
        parsed = parse_assistant_turn(reply)
        messages.append({"role": "assistant", "content": reply})
        if parsed.answer is not None:
            full_turns.append({"assistant_text": reply, "obs_text": ""})
            break
        if parsed.tool_call is not None:
            obs_json, _ = harness.execute_tool(parsed.tool_call)
            obs_text = f"<obs>{obs_json}</obs>"
            messages.append({"role": "user", "content": obs_text})
            full_turns.append({"assistant_text": reply, "obs_text": obs_text})
        else:
            full_turns.append({"assistant_text": reply, "obs_text": ""})
            break
    return full_turns


def synthesize_coldstart_sft(
    tasks: list[TaskItem],
    client: ChatCompletionClient,
    harness: MultiTurnAgentHarness,
    encode_fn: Callable[[str], list[int]],
    out_path: str | Path | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> list[TokenizedTrajectory]:
    """Synthesize coldstart SFT trajectories via DeepSeek teacher, retaining verified passes only."""
    accepted: list[TokenizedTrajectory] = []
    done_ids: set[str] = set()
    writer = None
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        done_ids.add(row["task_id"])
                        task_obj = TaskItem.from_dict(row["task"])
                        accepted.append(
                            build_prompt_and_gold_trajectory(
                                task=task_obj,
                                turns_spec=row["turns_spec"],
                                encode_fn=encode_fn,
                                harness=harness,
                            )
                        )
        writer = p.open("a", encoding="utf-8")

    try:
        for task in tasks:
            if task.task_id in done_ids:
                continue
            turns_spec = _run_teacher_continuation(
                task=task,
                prefix_turns=[],
                client=client,
                harness=harness,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            traj = build_prompt_and_gold_trajectory(
                task=task,
                turns_spec=turns_spec,
                encode_fn=encode_fn,
                harness=harness,
            )
            if traj.reward_breakdown.get("acc") == 1.0 and traj.reward_breakdown.get("format") == 1.0:
                accepted.append(traj)
                if writer is not None:
                    writer.write(
                        json.dumps(
                            {
                                "task_id": task.task_id,
                                "task": task.to_dict(),
                                "turns_spec": turns_spec,
                                "reward": traj.reward,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    writer.flush()
    finally:
        if writer is not None:
            writer.close()
    return accepted


def _find_first_failure_turn(traj: TokenizedTrajectory) -> int:
    """Locate the earliest assistant turn index where format, tool execution, or answer failed."""
    for i, turn in enumerate(traj.turns):
        if not turn.parsed.format_valid or turn.is_duplicate_call or turn.tool_success is False:
            return i
    return max(0, len(traj.turns) - 1)


def build_prefix_preserving_opd_and_dpo(
    tasks_by_id: dict[str, TaskItem],
    failed_trajectories: list[TokenizedTrajectory],
    client: ChatCompletionClient,
    harness: MultiTurnAgentHarness,
    encode_fn: Callable[[str], list[int]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict[str, list[Any]]:
    """Cross-family sequence-level on-policy correction + DPO pairs from student prefixes."""
    opd_trajectories: list[TokenizedTrajectory] = []
    dpo_pairs: list[DPOPair] = []

    for failed in failed_trajectories:
        if failed.reward_breakdown.get("acc", 0.0) >= 1.0 and failed.reward_breakdown.get("format", 0.0) >= 1.0:
            continue
        task = tasks_by_id.get(failed.task_id)
        if task is None:
            continue

        k = _find_first_failure_turn(failed)
        prefix_turns: list[dict[str, str]] = []
        for turn in failed.turns[:k]:
            prefix_turns.append(
                {
                    "assistant_text": turn.assistant_text,
                    "obs_text": (turn.tool_observation or "").replace("\n<assistant>\n", "").strip(),
                }
            )

        corrected_turns = _run_teacher_continuation(
            task=task,
            prefix_turns=prefix_turns,
            client=client,
            harness=harness,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        candidate_traj = build_prompt_and_gold_trajectory(
            task=task,
            turns_spec=corrected_turns,
            encode_fn=encode_fn,
            harness=harness,
        )
        if candidate_traj.reward_breakdown.get("acc") != 1.0 or candidate_traj.reward_breakdown.get("format") != 1.0:
            continue

        # Build prefix-preserving OPD trajectory: mask=0 on student prefix [0..k-1], mask=1 on teacher suffix [k..end]
        buf_opd = ExactTokenBuffer()
        prompt_text = format_task_prompt(task)
        prompt_ids = encode_fn(prompt_text)
        buf_opd.append_non_trainable(prompt_ids)

        for idx, step in enumerate(corrected_turns):
            asst_ids = encode_fn(step["assistant_text"])
            if idx < k:
                buf_opd.append_non_trainable(asst_ids)
            else:
                buf_opd.append_trainable(asst_ids, [0.0] * len(asst_ids))
            if step.get("obs_text"):
                obs_wrapped = f"\n{step['obs_text']}\n<assistant>\n"
                buf_opd.append_non_trainable(encode_fn(obs_wrapped))

        opd_trajectories.append(
            TokenizedTrajectory(
                task_id=task.task_id,
                group_id=task.group_id,
                token_ids=buf_opd.token_ids,
                assistant_mask=buf_opd.assistant_mask,
                old_logprobs=buf_opd.old_logprobs,
                prompt_len=len(prompt_ids),
                turn_count=len(corrected_turns),
                final_answer=candidate_traj.final_answer,
                raw_text=candidate_traj.raw_text,
                reward=candidate_traj.reward,
                reward_breakdown=dict(candidate_traj.reward_breakdown),
            )
        )

        # Build rejected suffix with mask=1 on failed student suffix [k..end]
        buf_rej = ExactTokenBuffer()
        buf_rej.append_non_trainable(prompt_ids)
        for idx, turn in enumerate(failed.turns):
            asst_ids = encode_fn(turn.assistant_text)
            if idx < k:
                buf_rej.append_non_trainable(asst_ids)
            else:
                buf_rej.append_trainable(asst_ids, [0.0] * len(asst_ids))
            if turn.tool_observation:
                buf_rej.append_non_trainable(encode_fn(turn.tool_observation))

        dpo_pairs.append(
            DPOPair(
                task_id=task.task_id,
                group_id=task.group_id,
                chosen_token_ids=buf_opd.token_ids,
                chosen_mask=buf_opd.assistant_mask,
                rejected_token_ids=buf_rej.token_ids,
                rejected_mask=buf_rej.assistant_mask,
                chosen_text="".join(s["assistant_text"] for s in corrected_turns[k:]),
                rejected_text="".join(t.assistant_text for t in failed.turns[k:]),
            )
        )

    return {"opd_trajectories": opd_trajectories, "dpo_pairs": dpo_pairs}
