from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from commerce_model_runtime import (
    ModelEvent,
    ModelMessage,
    ModelUsage,
    ProviderState,
    ResponseCompleted,
    StopReason,
    TextContent,
    TextDelta,
    ToolArgumentsDelta,
    ToolCallCompleted,
    ToolCallContent,
    ToolCallFailed,
    ToolCallStarted,
    UsageUpdated,
)

from .presentation import PresentationComponent, enrich_partial
from .streaming import AgentEvent, ToolOutcome, parse_partial_json
from .turn import EagerDispatcher, UNREADABLE_INPUT_TEXT


@dataclass
class _StreamingTool:
    id: str
    name: str
    buffer: str = ""
    signature: str | None = None


@dataclass(frozen=True)
class ModelRoundResult:
    message: ModelMessage | None
    tool_calls: tuple[ToolCallContent, ...]
    stop_reason: StopReason
    usage: ModelUsage
    provider_state: ProviderState | None
    malformed_call_ids: frozenset[str]


@dataclass
class ModelRoundRunner:
    specs: Mapping[str, PresentationComponent] = field(default_factory=dict)
    partial_tools: Collection[str] = ()
    state: Any = None
    eager_frames: bool = False
    _tools: dict[str, _StreamingTool] = field(default_factory=dict, init=False)
    _tool_calls: list[ToolCallContent] = field(default_factory=list, init=False)
    _malformed: set[str] = field(default_factory=set, init=False)
    _message: ModelMessage | None = field(default=None, init=False)
    _text_parts: list[str] = field(default_factory=list, init=False)
    _stop_reason: StopReason = field(default=StopReason.UNKNOWN, init=False)
    _usage: ModelUsage = field(default_factory=ModelUsage, init=False)
    _provider_state: ProviderState | None = field(default=None, init=False)

    @property
    def result(self) -> ModelRoundResult:
        message = self._message
        if message is None:
            content = []
            if self._text_parts:
                content.append(TextContent("".join(self._text_parts)))
            content.extend(self._tool_calls)
            if content:
                message = ModelMessage(role="assistant", content=content)
        return ModelRoundResult(
            message=message,
            tool_calls=tuple(self._tool_calls),
            stop_reason=self._stop_reason,
            usage=self._usage,
            provider_state=self._provider_state,
            malformed_call_ids=frozenset(self._malformed),
        )

    def _partial_frame(self, tool: _StreamingTool) -> AgentEvent | None:
        if tool.name not in self.partial_tools or tool.name not in self.specs:
            return None
        parsed = parse_partial_json(tool.buffer, settle_strings=not self.eager_frames)
        partial = enrich_partial(self.specs[tool.name], parsed, self.state) if parsed else None
        if partial is None:
            return None
        component, payload, signature = partial
        key = json.dumps(payload if self.eager_frames else signature, sort_keys=True, default=str)
        if key == tool.signature:
            return None
        tool.signature = key
        return AgentEvent.ui_partial(component, payload, tool.id)

    async def relay(
        self,
        events: AsyncIterator[ModelEvent],
        dispatcher: EagerDispatcher,
        announce: Callable[[str, str, dict[str, Any]], AgentEvent],
    ) -> AsyncIterator[AgentEvent]:
        async for event in events:
            if isinstance(event, TextDelta):
                if event.text:
                    self._text_parts.append(event.text)
                    yield AgentEvent.text_delta(event.text)
                continue

            if isinstance(event, ToolCallStarted):
                self._tools[event.id] = _StreamingTool(event.id, event.name)
                continue

            if isinstance(event, ToolArgumentsDelta):
                tool = self._tools.get(event.id)
                if tool is None:
                    continue
                tool.buffer += event.delta
                frame = self._partial_frame(tool)
                if frame is not None:
                    yield frame
                continue

            if isinstance(event, ToolCallCompleted):
                call = ToolCallContent(
                    id=event.id,
                    name=event.name,
                    arguments=dict(event.arguments),
                    provider_tool_call_id=event.provider_tool_call_id,
                )
                self._tool_calls.append(call)
                if dispatcher.dispatch(event.name, event.id, dict(event.arguments)):
                    yield announce(event.name, event.id, dict(event.arguments))
                continue

            if isinstance(event, ToolCallFailed):
                self._malformed.add(event.id)
                call = ToolCallContent(
                    id=event.id,
                    name=event.name,
                    arguments={},
                    provider_tool_call_id=event.provider_tool_call_id,
                )
                self._tool_calls.append(call)
                dispatcher.settle(event.id, ToolOutcome.error(UNREADABLE_INPUT_TEXT))
                continue

            if isinstance(event, UsageUpdated):
                self._usage = event.usage
                continue

            if isinstance(event, ResponseCompleted):
                response = event.response
                self._message = response.message
                self._stop_reason = response.stop_reason
                self._usage = response.usage
                self._provider_state = response.provider_state


def host_usage_totals() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def accumulate_model_usage(totals: dict[str, int], usage: ModelUsage) -> None:
    totals["input_tokens"] += usage.input_tokens or 0
    totals["output_tokens"] += usage.output_tokens or 0
    totals["cache_read_input_tokens"] += usage.cached_input_tokens or 0
    totals["cache_creation_input_tokens"] += int(
        usage.provider_details.get("cache_creation_input_tokens") or 0
    )


def model_prompt_tokens(usage: ModelUsage) -> int:
    return (
        (usage.input_tokens or 0)
        + (usage.cached_input_tokens or 0)
        + int(usage.provider_details.get("cache_creation_input_tokens") or 0)
    )
