"""Core dataset dataclasses for verifiable multi-turn tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agent_rl_flywheel.tools.hybrid_search import DocumentChunk


@dataclass(frozen=True)
class TaskItem:
    """Single verifiable multi-turn task with group isolation ID."""

    task_id: str
    group_id: str
    task_type: str  # "math" | "rag" | "hybrid_rag_math"
    split: str  # "train_coldstart" | "train_rl" | "val" | "locked_test_id" | "locked_test_ood"
    prompt: str
    gold_answer: str
    required_doc_ids: list[str] = field(default_factory=list)
    requires_tool: bool = True
    gold_turns: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize task into a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TaskItem:
        """Deserialize a `TaskItem` from dictionary."""
        return cls(**raw)


@dataclass
class DatasetBundle:
    """Complete dataset bundle with split tasks and document knowledge base."""

    splits: dict[str, list[TaskItem]]
    corpus: list[DocumentChunk]
    metadata: dict[str, Any] = field(default_factory=dict)
