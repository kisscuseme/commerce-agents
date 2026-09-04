from collections.abc import AsyncIterator

from commerce_model_runtime import (
    ModelMessage,
    ModelResponse,
    ModelRuntimeError,
    ModelTarget,
    ModelUsage,
    ProviderProtocolError,
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


def test_canonical_events_carry_only_provider_neutral_data():
    usage = ModelUsage(input_tokens=10, output_tokens=4)
    response = ModelResponse(
        message=ModelMessage(role="assistant", content=[TextContent("done")]),
        stop_reason=StopReason.END_TURN,
        usage=usage,
    )
    events = [
        TextDelta("hi"),
        ToolCallStarted(id="call_1", name="search_products"),
        ToolArgumentsDelta(id="call_1", delta='{"query":"tent"}'),
        ToolCallCompleted(id="call_1", name="search_products", arguments={"query": "tent"}),
        ToolCallFailed(id="call_2", name="search_products", reason="bad json"),
        UsageUpdated(usage),
        ResponseCompleted(response),
    ]
    assert [event.type for event in events] == [
        "text_delta",
        "tool_call_started",
        "tool_arguments_delta",
        "tool_call_completed",
        "tool_call_failed",
        "usage_updated",
        "response_completed",
    ]


def test_runtime_error_keeps_provider_context_without_sdk_exception_type():
    error = ProviderProtocolError(
        "invalid stream",
        provider="anthropic",
        model="claude-sonnet-5",
        provider_request_id="req_123",
    )
    assert isinstance(error, ModelRuntimeError)
    assert error.provider == "anthropic"
    assert error.model == "claude-sonnet-5"
    assert error.provider_request_id == "req_123"


def test_model_runtime_protocol_is_runtime_checkable():
    from commerce_model_runtime import ModelRuntime

    class Runtime:
        provider = "fake"

        def capabilities_for(self, target: ModelTarget):
            return object()

        async def stream(self, request) -> AsyncIterator[object]:
            if False:
                yield None

        async def complete(self, request):
            return ModelResponse(None, StopReason.END_TURN)

    assert isinstance(Runtime(), ModelRuntime)
