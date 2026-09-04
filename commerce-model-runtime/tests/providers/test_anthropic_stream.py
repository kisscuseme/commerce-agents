from types import SimpleNamespace as NS

import pytest

from commerce_model_runtime import (
    ModelMessage,
    ModelRequest,
    ModelTarget,
    StopReason,
    StreamInterruptedError,
    SystemSegment,
    TextContent,
    ToolChoice,
)
from commerce_model_runtime.providers.anthropic import AnthropicRuntime


def request():
    return ModelRequest(
        target=ModelTarget("anthropic", "claude-sonnet-5"),
        system=[SystemSegment("sys")],
        messages=[ModelMessage(role="user", content=[TextContent("hello")])],
        tools=[],
        tool_choice=ToolChoice.auto(),
        max_tokens=128,
    )


class FakeStream:
    def __init__(self, events, final):
        self.events = events
        self.final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def __aiter__(self):
        self._it = iter(self.events)
        return self

    async def __anext__(self):
        try:
            event = next(self._it)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(event, BaseException):
            raise event
        return event

    async def get_final_message(self):
        return self.final


class FailingStream(FakeStream):
    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise RuntimeError("connection dropped")


class FakeMessages:
    def __init__(self, stream):
        self._stream = stream
        self.calls = []

    def stream(self, **body):
        self.calls.append(body)
        return self._stream


class FakeClient:
    def __init__(self, stream):
        self.messages = FakeMessages(stream)


@pytest.mark.asyncio
async def test_stream_normalizes_text_tool_argument_and_completion_events():
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=10, output_tokens=0))),
        NS(type="content_block_start", index=0, content_block=NS(type="text", text="")),
        NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Hi ")),
        NS(
            type="content_block_start",
            index=1,
            content_block=NS(type="tool_use", id="toolu_1", name="search_products", input={}),
        ),
        NS(
            type="content_block_delta",
            index=1,
            delta=NS(type="input_json_delta", partial_json='{"query":"tent"}'),
        ),
        NS(type="content_block_stop", index=1),
        NS(type="message_delta", delta=NS(stop_reason="tool_use"), usage=NS(output_tokens=7)),
    ]
    final = NS(
        id="msg_1",
        stop_reason="tool_use",
        content=[
            NS(type="text", text="Hi "),
            NS(type="tool_use", id="toolu_1", name="search_products", input={"query": "tent"}),
        ],
        usage=NS(input_tokens=10, output_tokens=7, cache_read_input_tokens=3),
    )
    runtime = AnthropicRuntime(client=FakeClient(FakeStream(events, final)))
    normalized = [event async for event in runtime.stream(request())]
    assert [e.type for e in normalized] == [
        "usage_updated",
        "text_delta",
        "tool_call_started",
        "tool_arguments_delta",
        "tool_call_completed",
        "usage_updated",
        "response_completed",
    ]
    assert normalized[1].text == "Hi "
    assert normalized[4].arguments == {"query": "tent"}
    assert normalized[-1].response.stop_reason is StopReason.TOOL_USE
    tool = normalized[-1].response.message.content[1]
    assert tool.id == "toolu_1" and tool.provider_tool_call_id == "toolu_1"
    assert normalized[-1].response.usage.cached_input_tokens == 3


@pytest.mark.asyncio
async def test_stream_turns_invalid_json_into_tool_call_failed_without_throwing():
    events = [
        NS(
            type="content_block_start",
            index=0,
            content_block=NS(type="tool_use", id="toolu_bad", name="search_products", input={}),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="input_json_delta", partial_json='{"query":'),
        ),
        NS(type="content_block_stop", index=0),
    ]
    final = NS(id="msg_bad", stop_reason="tool_use", content=[], usage=NS())
    runtime = AnthropicRuntime(client=FakeClient(FakeStream(events, final)))
    normalized = [event async for event in runtime.stream(request())]
    failed = [e for e in normalized if e.type == "tool_call_failed"]
    assert len(failed) == 1
    assert failed[0].id == "toolu_bad"


@pytest.mark.asyncio
async def test_open_tool_value_error_is_salvaged_as_failed_call_not_stream_failure():
    events = [
        NS(type="message_start", message=NS(usage=NS(input_tokens=4, output_tokens=1))),
        NS(
            type="content_block_start",
            index=0,
            content_block=NS(type="tool_use", id="toolu_bad", name="present_products", input={}),
        ),
        NS(
            type="content_block_delta",
            index=0,
            delta=NS(type="input_json_delta", partial_json='{"picks":'),
        ),
        ValueError("Anthropic accumulator rejected incomplete tool JSON"),
    ]
    final = NS(id="never", stop_reason="tool_use", content=[], usage=NS())
    runtime = AnthropicRuntime(client=FakeClient(FakeStream(events, final)))

    normalized = [event async for event in runtime.stream(request())]

    assert [event.type for event in normalized][-1] == "tool_call_failed"
    failed = normalized[-1]
    assert failed.id == "toolu_bad" and failed.name == "present_products"
    assert not [event for event in normalized if event.type == "response_completed"]


@pytest.mark.asyncio
async def test_stream_error_after_visible_output_is_stream_interrupted():
    events = [NS(type="content_block_delta", index=0, delta=NS(type="text_delta", text="Hi"))]
    final = NS(id="never", stop_reason="end_turn", content=[], usage=NS())
    runtime = AnthropicRuntime(client=FakeClient(FailingStream(events, final)))
    with pytest.raises(StreamInterruptedError, match="connection dropped"):
        _ = [event async for event in runtime.stream(request())]
