from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias


class SegmentStability(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class SystemSegment:
    text: str
    stability: SegmentStability = SegmentStability.DYNAMIC


class ToolChoiceMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    SPECIFIC = "specific"


@dataclass(frozen=True)
class ToolChoice:
    mode: ToolChoiceMode
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode is ToolChoiceMode.SPECIFIC:
            if not self.name or not self.name.strip():
                raise ValueError("specific tool choice requires a non-empty tool name")
        elif self.name is not None:
            raise ValueError("tool name is only valid for specific tool choice")

    @classmethod
    def auto(cls) -> "ToolChoice":
        return cls(ToolChoiceMode.AUTO)

    @classmethod
    def none(cls) -> "ToolChoice":
        return cls(ToolChoiceMode.NONE)

    @classmethod
    def specific(cls, name: str) -> "ToolChoice":
        return cls(ToolChoiceMode.SPECIFIC, name=name)


class ReasoningEffort(str, Enum):
    OFF = "off"
    DEFAULT = "default"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


@dataclass(frozen=True)
class ReasoningConfig:
    effort: ReasoningEffort = ReasoningEffort.DEFAULT


@dataclass(frozen=True)
class CachePolicy:
    enabled: bool = True
    rolling_conversation: bool = False


@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be non-empty")
        if not self.model.strip():
            raise ValueError("model must be non-empty")


@dataclass(frozen=True)
class ProviderState:
    provider: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            json.dumps(self.data)
        except (TypeError, ValueError) as exc:
            raise TypeError("provider state data must be JSON-serializable") from exc


@dataclass(frozen=True)
class TextContent:
    text: str


@dataclass(frozen=True)
class ToolCallContent:
    id: str
    name: str
    arguments: dict[str, Any]
    provider_tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolResultContent:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ProviderOpaqueContent:
    provider: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        try:
            json.dumps(self.data)
        except (TypeError, ValueError) as exc:
            raise TypeError("provider opaque content data must be JSON-serializable") from exc


ModelContent: TypeAlias = TextContent | ToolCallContent | ToolResultContent | ProviderOpaqueContent


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["user", "assistant"]
    content: list[ModelContent]


@dataclass(frozen=True)
class FunctionToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    progressive: bool = False


@dataclass(frozen=True)
class BuiltinToolSpec:
    kind: str
    options: dict[str, Any] = field(default_factory=dict)


ToolSpec: TypeAlias = FunctionToolSpec | BuiltinToolSpec


@dataclass(frozen=True)
class ModelRequestMetadata:
    operation: str | None = None
    data_classification: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRequest:
    target: ModelTarget
    system: list[SystemSegment]
    messages: list[ModelMessage]
    tools: list[ToolSpec]
    tool_choice: ToolChoice
    max_tokens: int
    reasoning: ReasoningConfig | None = None
    cache: CachePolicy | None = None
    provider_state: ProviderState | None = None
    metadata: ModelRequestMetadata | None = None

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    provider_details: dict[str, Any] = field(default_factory=dict)


class StopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    CONTENT_FILTER = "content_filter"
    PAUSE = "pause"
    ABANDONED = "abandoned"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelResponse:
    message: ModelMessage | None
    stop_reason: StopReason
    usage: ModelUsage = field(default_factory=ModelUsage)
    provider_state: ProviderState | None = None
    provider_request_id: str | None = None
