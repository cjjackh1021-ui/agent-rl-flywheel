"""Agent-Lightning style API gateway / proxy abstraction preserving exact token IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TurnSample:
    """Single generated turn carrying exact token IDs and rollout logprobs."""

    text: str
    token_ids: list[int]
    logprobs: list[float]


class TokenPolicyBackend(Protocol):
    """Protocol implemented by both `TinyCausalLMBackend` and `HFQLoRABackend`."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...

    def generate_with_logprobs(
        self,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.7,
    ) -> TurnSample: ...


class AgentProxyGateway:
    """Tracks rollout turns without re-tokenizing assistant responses."""

    def __init__(self, backend: TokenPolicyBackend) -> None:
        self.backend = backend
        self.total_calls: int = 0
        self.retokenization_events: int = 0

    def encode_non_trainable(self, text: str) -> list[int]:
        """Encode fixed system, user, or `<obs>` strings appended with mask=0."""
        return self.backend.encode(text)

    def sample_turn(
        self,
        prefix_token_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> TurnSample:
        """Generate assistant tokens directly from `prefix_token_ids`, returning exact token IDs."""
        self.total_calls += 1
        sample = self.backend.generate_with_logprobs(
            prompt_token_ids=prefix_token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        if len(sample.token_ids) != len(sample.logprobs):
            raise ValueError(
                f"Backend returned mismatched token_ids ({len(sample.token_ids)}) "
                f"and logprobs ({len(sample.logprobs)})"
            )
        return sample
