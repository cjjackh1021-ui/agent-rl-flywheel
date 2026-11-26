"""Structured multi-turn Agent protocol, parsing, and trajectory schemas."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
DOC_ID_RE = re.compile(r"\bdoc_[a-zA-Z0-9_-]+\b")


@dataclass(frozen=True)
class ToolCall:
    """Single structured tool invocation parsed from an assistant turn."""

    name: str
    arguments: dict[str, Any]
    raw_json: str = ""

    def canonical_key(self) -> tuple[str, str]:
        """Return canonical (name, sorted-json) key for duplicate-call detection."""
        return (self.name.strip(), json.dumps(self.arguments, sort_keys=True, ensure_ascii=False))


@dataclass
class ParsedTurn:
    """Result of parsing a single assistant turn under the `<think>/<tool_call>/<answer>` protocol."""

    raw_text: str
    think: str | None = None
    tool_call: ToolCall | None = None
    answer: str | None = None
    format_valid: bool = False
    format_errors: list[str] = field(default_factory=list)


@dataclass
class TurnRecord:
    """Recorded turn metadata inside a trajectory."""

    turn_idx: int
    assistant_text: str
    parsed: ParsedTurn
    tool_observation: str | None = None
    tool_success: bool | None = None
    is_duplicate_call: bool = False
    assistant_token_count: int = 0
    obs_token_count: int = 0


@dataclass
class TokenizedTrajectory:
    """Complete multi-turn rollout trajectory preserving exact token IDs and `<obs>` masks."""

    task_id: str
    group_id: str
    token_ids: list[int]
    assistant_mask: list[int]
    old_logprobs: list[float]
    prompt_len: int
    turn_count: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    final_answer: str | None = None
    raw_text: str = ""
    reward: float = 0.0
    reward_breakdown: dict[str, Any] = field(default_factory=dict)
    retokenization_events: int = 0

    @property
    def response_len(self) -> int:
        """Number of tokens generated after the initial prompt."""
        return max(0, len(self.token_ids) - self.prompt_len)

    @property
    def trainable_token_count(self) -> int:
        """Number of assistant tokens contributing gradients (`assistant_mask == 1`)."""
        return sum(self.assistant_mask)


def parse_assistant_turn(text: str) -> ParsedTurn:
    """Parse one assistant response expecting `<think>` and either `<tool_call>` or `<answer>`."""
    errors: list[str] = []
    thinks = THINK_RE.findall(text)
    tool_matches = TOOL_CALL_RE.findall(text)
    answers = ANSWER_RE.findall(text)

    if len(thinks) != 1:
        errors.append(f"expected_1_think_got_{len(thinks)}")
    think = thinks[0].strip() if len(thinks) == 1 else (thinks[-1].strip() if thinks else None)
    if think is not None and not think:
        errors.append("empty_think")

    if len(tool_matches) > 1:
        errors.append(f"multiple_tool_calls_{len(tool_matches)}")
    if len(answers) > 1:
        errors.append(f"multiple_answers_{len(answers)}")
    if tool_matches and answers:
        errors.append("both_tool_call_and_answer")
    if not tool_matches and not answers:
        errors.append("missing_tool_call_or_answer")

    parsed_tool: ToolCall | None = None
    if tool_matches:
        raw_tc = tool_matches[-1].strip()
        try:
            payload = json.loads(raw_tc)
            if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
                errors.append("tool_call_missing_name")
            else:
                args = payload.get("arguments", {})
                if not isinstance(args, dict):
                    errors.append("tool_call_arguments_not_dict")
                else:
                    parsed_tool = ToolCall(name=payload["name"].strip(), arguments=args, raw_json=raw_tc)
        except json.JSONDecodeError:
            errors.append("tool_call_invalid_json")

    parsed_answer: str | None = None
    if answers:
        parsed_answer = answers[-1].strip()
        if not parsed_answer:
            errors.append("empty_answer")

    return ParsedTurn(
        raw_text=text,
        think=think,
        tool_call=parsed_tool,
        answer=parsed_answer,
        format_valid=len(errors) == 0,
        format_errors=errors,
    )


def extract_doc_ids(text: str | None) -> list[str]:
    """Extract referenced document IDs (`doc_...`) while preserving first-seen order."""
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in DOC_ID_RE.findall(text):
        if match not in seen:
            seen.add(match)
            ordered.append(match)
    return ordered
