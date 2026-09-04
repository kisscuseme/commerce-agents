import pytest

from commerce_model_runtime import ModelMessage, ModelRequest, ModelTarget, SystemSegment, TextContent, ToolChoice
from commerce_model_runtime.providers.anthropic import AnthropicRuntime


class BrokenMessages:
    def stream(self, **kwargs):
        raise ValueError("client misconfigured")


class BrokenClient:
    messages = BrokenMessages()


def request():
    return ModelRequest(
        target=ModelTarget("anthropic", "claude-sonnet-5"),
        system=[SystemSegment("sys")],
        messages=[ModelMessage(role="user", content=[TextContent("hello")])],
        tools=[],
        tool_choice=ToolChoice.auto(),
        max_tokens=128,
    )


@pytest.mark.asyncio
async def test_pre_stream_value_error_is_not_reclassified_as_provider_failure():
    runtime = AnthropicRuntime(client=BrokenClient())
    with pytest.raises(ValueError, match="client misconfigured"):
        _ = [event async for event in runtime.stream(request())]
