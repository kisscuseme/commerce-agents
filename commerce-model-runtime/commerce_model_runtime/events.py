from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias

from .types import ModelResponse, ModelUsage


@dataclass(frozen=True)
class TextDelta:
    type: ClassVar[str] = "text_delta"
    text: str


@dataclass(frozen=True)
class ToolCallStarted:
    type: ClassVar[str] = "tool_call_started"
    id: str
    name: str
    provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolArgumentsDelta:
    type: ClassVar[str] = "tool_arguments_delta"
    id: str
    delta: str


@dataclass(frozen=True)
class ToolCallCompleted:
    type: ClassVar[str] = "tool_call_completed"
    id: str
    name: str
    arguments: dict[str, Any]
    provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCallFailed:
    type: ClassVar[str] = "tool_call_failed"
    id: str
    name: str
    reason: str
    provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class UsageUpdated:
    type: ClassVar[str] = "usage_updated"
    usage: ModelUsage


@dataclass(frozen=True)
class ResponseCompleted:
    type: ClassVar[str] = "response_completed"
    response: ModelResponse


ModelEvent: TypeAlias = (
    TextDelta
    | ToolCallStarted
    | ToolArgumentsDelta
    | ToolCallCompleted
    | ToolCallFailed
    | UsageUpdated
    | ResponseCompleted
)
