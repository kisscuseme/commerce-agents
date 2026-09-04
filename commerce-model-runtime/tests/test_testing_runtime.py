import pytest

from commerce_model_runtime import (
    FunctionToolSpec,
    ModelMessage,
    ModelRequest,
    ModelTarget,
    StopReason,
    SystemSegment,
    TextContent,
    ToolCallContent,
    ToolChoice,
)
from commerce_model_runtime.testing import (
    FakeModelRuntime,
    multi_tool_round,
    text_round,
    tool_round,
)


def minimal_request():
    return ModelRequest(
        target=ModelTarget(provider="fake", model="fake-model"),
        system=[SystemSegment("system")],
        messages=[ModelMessage(role="user", content=[TextContent("hello")])],
        tools=[
            FunctionToolSpec(
                name="search_products",
                description="search",
                input_schema={"type": "object", "properties": {}},
            )
        ],
        tool_choice=ToolChoice.auto(),
        max_tokens=128,
    )


@pytest.mark.asyncio
async def test_fake_runtime_records_request_and_replays_events():
    runtime = FakeModelRuntime([text_round("hello")])
    events = [event async for event in runtime.stream(minimal_request())]
    assert len(runtime.calls) == 1
    assert runtime.calls[0] == minimal_request()
    assert [type(e).__name__ for e in events] == ["TextDelta", "ResponseCompleted"]
    assert events[-1].response.stop_reason is StopReason.END_TURN


@pytest.mark.asyncio
async def test_tool_round_emits_complete_arguments_and_response_message():
    runtime = FakeModelRuntime([tool_round("search_products", {"query": "tent"}, call_id="c1")])
    events = [event async for event in runtime.stream(minimal_request())]
    completed = [e for e in events if type(e).__name__ == "ToolCallCompleted"]
    assert completed[0].arguments == {"query": "tent"}
    response = events[-1].response
    assert response.stop_reason is StopReason.TOOL_USE
    assert response.message.content == [
        ToolCallContent(id="c1", name="search_products", arguments={"query": "tent"})
    ]


@pytest.mark.asyncio
async def test_multi_tool_round_preserves_call_order():
    runtime = FakeModelRuntime([
        multi_tool_round([
            ("search_products", {"query": "tent"}, "c1"),
            ("search_products", {"query": "bag"}, "c2"),
        ])
    ])
    events = [event async for event in runtime.stream(minimal_request())]
    assert [e.id for e in events if type(e).__name__ == "ToolCallCompleted"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_complete_records_request_and_returns_scripted_response():
    runtime = FakeModelRuntime([text_round("memory result")])
    response = await runtime.complete(minimal_request())
    assert response.message.content == [TextContent("memory result")]
    assert runtime.complete_calls == [minimal_request()]


@pytest.mark.asyncio
async def test_fake_runtime_fails_when_script_is_exhausted():
    runtime = FakeModelRuntime([])
    with pytest.raises(AssertionError, match="script exhausted"):
        await runtime.complete(minimal_request())
