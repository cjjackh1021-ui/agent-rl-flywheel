"""Multi-turn Agent Harness with exact token buffering and strict `<obs>` loss masking."""

from __future__ import annotations

import json
from typing import Any

from agent_rl_flywheel.data.schema import TaskItem
from agent_rl_flywheel.env.proxy import AgentProxyGateway
from agent_rl_flywheel.env.verifier import RLVRVerifier
from agent_rl_flywheel.protocol import (
    TokenizedTrajectory,
    ToolCall,
    TurnRecord,
    parse_assistant_turn,
)
from agent_rl_flywheel.tools.hybrid_search import HybridRetriever
from agent_rl_flywheel.tools.sympy_tool import SymPyTool

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise reasoning agent with two tools: `sympy_eval(expr)` and "
    "`hybrid_search(query, top_k)`. On each turn, think inside <think>...</think>, "
    "then emit either <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> "
    "or your final conclusion inside <answer>...</answer> (cite doc_id when using search)."
)


class ExactTokenBuffer:
    """Append-only token buffer ensuring zero retokenization drift and strict `<obs>` masking."""

    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.assistant_mask: list[int] = []
        self.old_logprobs: list[float] = []

    def append_non_trainable(self, ids: list[int]) -> None:
        """Append prompt or `<obs>` observation tokens with mask=0 and logprob=0.0."""
        self.token_ids.extend(ids)
        self.assistant_mask.extend([0] * len(ids))
        self.old_logprobs.extend([0.0] * len(ids))

    def append_trainable(self, ids: list[int], logprobs: list[float]) -> None:
        """Append model-generated assistant tokens with mask=1 and sampled logprobs."""
        if len(ids) != len(logprobs):
            raise ValueError("Trainable token_ids and logprobs length mismatch")
        self.token_ids.extend(ids)
        self.assistant_mask.extend([1] * len(ids))
        self.old_logprobs.extend(logprobs)


def format_task_prompt(task: TaskItem, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Format fixed system + user prompt header for the multi-turn agent harness."""
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{task.prompt}\n</user>\n<assistant>\n"


class MultiTurnAgentHarness:
    """Executes multi-turn tool-calling rollouts and computes verifiable RLVR rewards."""

    def __init__(
        self,
        sympy_tool: SymPyTool,
        retriever: HybridRetriever,
        verifier: RLVRVerifier | None = None,
        max_turns: int = 4,
        max_new_tokens_per_turn: int = 256,
        max_seq_len: int = 2048,
    ) -> None:
        self.sympy_tool = sympy_tool
        self.retriever = retriever
        self.verifier = verifier or RLVRVerifier(sympy_tool=sympy_tool)
        self.max_turns = max_turns
        self.max_new_tokens_per_turn = max_new_tokens_per_turn
        self.max_seq_len = max_seq_len

    def execute_tool(self, call: ToolCall) -> tuple[str, bool]:
        """Run a sandboxed tool call and return `(json_observation, ok)`."""
        if call.name == "sympy_eval":
            expr = call.arguments.get("expr")
            if not isinstance(expr, str):
                return json.dumps({"ok": False, "error": "expr must be string"}, ensure_ascii=False), False
            out = self.sympy_tool.evaluate(expr)
            return json.dumps(out, ensure_ascii=False), bool(out.get("ok"))
        if call.name == "hybrid_search":
            query = call.arguments.get("query")
            if not isinstance(query, str):
                return json.dumps({"ok": False, "error": "query must be string"}, ensure_ascii=False), False
            top_k = call.arguments.get("top_k", 3)
            if not isinstance(top_k, int) or top_k <= 0:
                top_k = 3
            hits = self.retriever.search(query, top_k=min(top_k, 5))
            return json.dumps({"ok": bool(hits), "hits": hits}, ensure_ascii=False), bool(hits)
        return json.dumps({"ok": False, "error": f"unknown tool: {call.name}"}, ensure_ascii=False), False

    def run_rollout(
        self,
        task: TaskItem,
        gateway: AgentProxyGateway,
        temperature: float = 0.7,
    ) -> TokenizedTrajectory:
        """Run a multi-turn rollout via `gateway`, preserving exact token IDs and `<obs>` masks."""
        buffer = ExactTokenBuffer()
        prompt_text = format_task_prompt(task)
        prompt_ids = gateway.encode_non_trainable(prompt_text)
        buffer.append_non_trainable(prompt_ids)
        prompt_len = len(prompt_ids)

        turns: list[TurnRecord] = []
        tool_calls: list[ToolCall] = []
        seen_calls: set[tuple[str, str]] = set()
        raw_segments: list[str] = [prompt_text]
        final_answer: str | None = None

        for turn_idx in range(self.max_turns):
            remaining = self.max_seq_len - len(buffer.token_ids)
            if remaining <= 8:
                break
            budget = min(self.max_new_tokens_per_turn, remaining)
            sample = gateway.sample_turn(
                prefix_token_ids=list(buffer.token_ids),
                max_new_tokens=budget,
                temperature=temperature,
            )
            buffer.append_trainable(sample.token_ids, sample.logprobs)
            raw_segments.append(sample.text)

            parsed = parse_assistant_turn(sample.text)
            record = TurnRecord(
                turn_idx=turn_idx,
                assistant_text=sample.text,
                parsed=parsed,
                assistant_token_count=len(sample.token_ids),
            )

            if parsed.answer is not None:
                final_answer = parsed.answer
                turns.append(record)
                break

            if parsed.tool_call is not None:
                key = parsed.tool_call.canonical_key()
                is_dup = key in seen_calls
                seen_calls.add(key)
                tool_calls.append(parsed.tool_call)
                obs_json, ok = self.execute_tool(parsed.tool_call)
                obs_text = f"\n<obs>{obs_json}</obs>\n<assistant>\n"
                obs_ids = gateway.encode_non_trainable(obs_text)
                if len(buffer.token_ids) + len(obs_ids) <= self.max_seq_len:
                    buffer.append_non_trainable(obs_ids)
                    raw_segments.append(obs_text)
                record.tool_observation = obs_text
                record.tool_success = ok
                record.is_duplicate_call = is_dup
                record.obs_token_count = len(obs_ids)
                turns.append(record)
            else:
                # Format error without tool call or answer: append short corrective hint as non-trainable
                turns.append(record)
                if turn_idx + 1 < self.max_turns:
                    hint = "\n<obs>{\"ok\": false, \"error\": \"Emit <tool_call> or <answer>.\"}</obs>\n<assistant>\n"
                    hint_ids = gateway.encode_non_trainable(hint)
                    if len(buffer.token_ids) + len(hint_ids) <= self.max_seq_len:
                        buffer.append_non_trainable(hint_ids)
                        raw_segments.append(hint)
                else:
                    break

        traj = TokenizedTrajectory(
            task_id=task.task_id,
            group_id=task.group_id,
            token_ids=buffer.token_ids,
            assistant_mask=buffer.assistant_mask,
            old_logprobs=buffer.old_logprobs,
            prompt_len=prompt_len,
            turn_count=len(turns),
            tool_calls=tool_calls,
            turns=turns,
            final_answer=final_answer,
            raw_text="".join(raw_segments),
            retokenization_events=gateway.retokenization_events,
        )
        reward_res = self.verifier.score(task=task, trajectory=traj)
        traj.reward = reward_res.total
        traj.reward_breakdown = {
            "acc": reward_res.acc,
            "tool": reward_res.tool,
            "format": reward_res.fmt,
            "penalty": reward_res.penalty,
            "hacking_flags": list(reward_res.hacking_flags),
        }
        return traj


def build_prompt_and_gold_trajectory(
    task: TaskItem,
    turns_spec: list[dict[str, Any]],
    encode_fn: Any,
    harness: MultiTurnAgentHarness,
) -> TokenizedTrajectory:
    """Construct a ground-truth or teacher-generated tokenized trajectory with strict `<obs>` masks."""
    buf = ExactTokenBuffer()
    prompt_text = format_task_prompt(task)
    prompt_ids = encode_fn(prompt_text)
    buf.append_non_trainable(prompt_ids)
    raw_segments = [prompt_text]
    turns: list[TurnRecord] = []
    tool_calls: list[ToolCall] = []
    seen_calls: set[tuple[str, str]] = set()
    final_answer: str | None = None

    for idx, step in enumerate(turns_spec):
        asst_text = step["assistant_text"]
        asst_ids = encode_fn(asst_text)
        buf.append_trainable(asst_ids, [0.0] * len(asst_ids))
        raw_segments.append(asst_text)
        parsed = parse_assistant_turn(asst_text)
        rec = TurnRecord(
            turn_idx=idx,
            assistant_text=asst_text,
            parsed=parsed,
            assistant_token_count=len(asst_ids),
        )
        if parsed.answer is not None:
            final_answer = parsed.answer
            turns.append(rec)
            break
        if parsed.tool_call is not None:
            key = parsed.tool_call.canonical_key()
            rec.is_duplicate_call = key in seen_calls
            seen_calls.add(key)
            tool_calls.append(parsed.tool_call)
            obs_json, ok = harness.execute_tool(parsed.tool_call)
            obs_text = f"\n<obs>{obs_json}</obs>\n<assistant>\n"
            obs_ids = encode_fn(obs_text)
            buf.append_non_trainable(obs_ids)
            raw_segments.append(obs_text)
            rec.tool_observation = obs_text
            rec.tool_success = ok
            rec.obs_token_count = len(obs_ids)
        turns.append(rec)

    traj = TokenizedTrajectory(
        task_id=task.task_id,
        group_id=task.group_id,
        token_ids=buf.token_ids,
        assistant_mask=buf.assistant_mask,
        old_logprobs=buf.old_logprobs,
        prompt_len=len(prompt_ids),
        turn_count=len(turns),
        tool_calls=tool_calls,
        turns=turns,
        final_answer=final_answer,
        raw_text="".join(raw_segments),
    )
    res = harness.verifier.score(task=task, trajectory=traj)
    traj.reward = res.total
    traj.reward_breakdown = {
        "acc": res.acc,
        "tool": res.tool,
        "format": res.fmt,
        "penalty": res.penalty,
        "hacking_flags": list(res.hacking_flags),
    }
    return traj
