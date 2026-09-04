import asyncio

import pytest

from commerce_model_runtime import (
    ModelMessage,
    ModelResponse,
    ModelUsage,
    ProviderState,
    ResponseCompleted,
    StopReason,
    TextContent,
    TextDelta,
    ToolArgumentsDelta,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    UsageUpdated,
)
from commerce_common.model_round import ModelRoundRunner
from commerce_common.presentation import PresentationComponent
from commerce_common.streaming import AgentEvent, ToolOutcome
from commerce_common.turn import EagerDispatcher, UNREADABLE_INPUT_TEXT


async def collect(runner, events, dispatcher):
    async def source():
        for event in events:
            yield event

    return [
        event
        async for event in runner.relay(
            source(), dispatcher, lambda n, i, a: AgentEvent.tool_call(n, i, a)
        )
    ]


@pytest.mark.asyncio
async def test_text_and_usage_and_response_are_collected():
    response = ModelResponse(
        ModelMessage("assistant", [TextContent("hello")]),
        StopReason.END_TURN,
        ModelUsage(input_tokens=2, output_tokens=1),
        ProviderState("fake", {"id": "r"}),
    )
    runner = ModelRoundRunner()

    async def unused(name, args):
        return ToolOutcome("unused")

    dispatcher = EagerDispatcher(unused, enabled=False)
    host = await collect(
        runner,
        [
            TextDelta("he"),
            TextDelta("llo"),
            UsageUpdated(ModelUsage(input_tokens=2)),
            ResponseCompleted(response),
        ],
        dispatcher,
    )
    assert [e.data["text"] for e in host] == ["he", "llo"]
    result = runner.result
    assert result.message == response.message
    assert result.stop_reason is StopReason.END_TURN
    assert result.usage.output_tokens == 1
    assert result.provider_state == ProviderState("fake", {"id": "r"})


@pytest.mark.asyncio
async def test_dispatch_starts_only_when_tool_call_is_completed():
    started = asyncio.Event()
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        started.set()
        return ToolOutcome("ok")

    runner = ModelRoundRunner()
    dispatcher = EagerDispatcher(execute, enabled=True)
    events = [
        ToolCallStarted("c1", "search"),
        ToolArgumentsDelta("c1", '{"q":"x"}'),
        ToolCallCompleted("c1", "search", {"q": "x"}),
    ]
    host = await collect(runner, events, dispatcher)
    await asyncio.wait_for(started.wait(), 0.2)
    assert calls == [("search", {"q": "x"})]
    assert [e.type for e in host] == ["tool_call"]
    assert runner.result.tool_calls[0].arguments == {"q": "x"}
    dispatcher.cancel()


@pytest.mark.asyncio
async def test_failed_tool_call_is_settled_without_execution():
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return ToolOutcome("ran")

    runner = ModelRoundRunner()
    dispatcher = EagerDispatcher(execute, enabled=True)
    await collect(
        runner,
        [ToolCallStarted("bad", "search"), ToolCallFailed("bad", "search", "bad json")],
        dispatcher,
    )
    outcomes = await dispatcher.collect(runner.result.tool_calls)
    assert calls == []
    assert outcomes[0].is_error and outcomes[0].result_text == UNREADABLE_INPUT_TEXT
    assert runner.result.malformed_call_ids == frozenset({"bad"})


@pytest.mark.asyncio
async def test_partial_ui_uses_argument_deltas_but_execution_waits_for_completion():
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return ToolOutcome("ok")

    spec = PresentationComponent(
        name="present_products",
        component="products",
        payload_model=dict,
        enrich_partial=lambda data, state: data if data.get("items") else None,
    )
    runner = ModelRoundRunner(
        specs={"present_products": spec}, partial_tools={"present_products"}
    )
    dispatcher = EagerDispatcher(execute, enabled=True)
    events = [
        ToolCallStarted("c1", "present_products"),
        ToolArgumentsDelta("c1", '{"items":[{"id":"p1"}]}'),
    ]
    host = await collect(runner, events, dispatcher)
    assert calls == []
    assert host[0].type == "ui_partial" and host[0].data["stream_id"] == "c1"
