from types import SimpleNamespace as NS

import pytest

from commerce_model_runtime import (
    AuthenticationError,
    ModelMessage,
    ModelRequest,
    ModelTarget,
    RateLimitError,
    StopReason,
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


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def create(self, **body):
        self.calls.append(body)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, messages):
        self.messages = messages


@pytest.mark.asyncio
async def test_complete_normalizes_response_and_usage():
    response = NS(
        id="msg_1",
        stop_reason="end_turn",
        content=[NS(type="text", text="done")],
        usage=NS(input_tokens=4, output_tokens=2, cache_creation_input_tokens=1),
    )
    messages = FakeMessages(response=response)
    runtime = AnthropicRuntime(client=FakeClient(messages))
    result = await runtime.complete(request())
    assert result.stop_reason is StopReason.END_TURN
    assert result.message.content == [TextContent("done")]
    assert result.usage.input_tokens == 4
    assert result.usage.provider_details["cache_creation_input_tokens"] == 1
    assert messages.calls[0]["model"] == "claude-sonnet-5"


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error_type", [(401, AuthenticationError), (429, RateLimitError)])
async def test_complete_normalizes_provider_errors(status, error_type):
    runtime = AnthropicRuntime(client=FakeClient(FakeMessages(error=FakeHTTPError(status))))
    with pytest.raises(error_type):
        await runtime.complete(request())
