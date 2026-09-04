from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .capabilities import ModelCapabilities
from .events import ModelEvent, ResponseCompleted, TextDelta, ToolCallCompleted, ToolCallStarted
from .types import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelTarget,
    ModelUsage,
    StopReason,
    TextContent,
    ToolCallContent,
)


@dataclass(frozen=True)
class ScriptedRound:
    events: tuple[ModelEvent, ...]
    response: ModelResponse


class FakeModelRuntime:
    provider = "fake"

    def __init__(self, rounds: Iterable[ScriptedRound]) -> None:
        self._rounds = list(rounds)
        self.calls: list[ModelRequest] = []
        self.stream_calls: list[ModelRequest] = []
        self.complete_calls: list[ModelRequest] = []

    def capabilities_for(self, target: ModelTarget) -> ModelCapabilities:
        return ModelCapabilities.full()

    def _next_round(self) -> ScriptedRound:
        if not self._rounds:
            raise AssertionError("fake model runtime script exhausted")
        return self._rounds.pop(0)

    async def stream(self, request: ModelRequest):
        self.calls.append(request)
        self.stream_calls.append(request)
        scripted = self._next_round()
        for event in scripted.events:
            yield event

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        self.complete_calls.append(request)
        return self._next_round().response


def text_round(text: str, *, usage: ModelUsage | None = None) -> ScriptedRound:
    response = ModelResponse(
        message=ModelMessage(role="assistant", content=[TextContent(text)]),
        stop_reason=StopReason.END_TURN,
        usage=usage or ModelUsage(),
    )
    return ScriptedRound((TextDelta(text), ResponseCompleted(response)), response)


def tool_round(
    name: str,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    usage: ModelUsage | None = None,
) -> ScriptedRound:
    call = ToolCallContent(id=call_id, name=name, arguments=dict(arguments))
    response = ModelResponse(
        message=ModelMessage(role="assistant", content=[call]),
        stop_reason=StopReason.TOOL_USE,
        usage=usage or ModelUsage(),
    )
    return ScriptedRound(
        (
            ToolCallStarted(id=call_id, name=name),
            ToolCallCompleted(id=call_id, name=name, arguments=dict(arguments)),
            ResponseCompleted(response),
        ),
        response,
    )


def multi_tool_round(
    calls: Iterable[tuple[str, dict[str, Any], str]],
    *,
    usage: ModelUsage | None = None,
) -> ScriptedRound:
    normalized = [(name, dict(arguments), call_id) for name, arguments, call_id in calls]
    content = [
        ToolCallContent(id=call_id, name=name, arguments=arguments)
        for name, arguments, call_id in normalized
    ]
    response = ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        stop_reason=StopReason.TOOL_USE,
        usage=usage or ModelUsage(),
    )
    events: list[ModelEvent] = []
    for name, arguments, call_id in normalized:
        events.append(ToolCallStarted(id=call_id, name=name))
        events.append(ToolCallCompleted(id=call_id, name=name, arguments=arguments))
    events.append(ResponseCompleted(response))
    return ScriptedRound(tuple(events), response)
