from commerce_model_runtime import (
    BuiltinToolSpec,
    CachePolicy,
    FunctionToolSpec,
    ModelMessage,
    ModelRequest,
    ModelTarget,
    ReasoningConfig,
    ReasoningEffort,
    SegmentStability,
    SystemSegment,
    TextContent,
    ToolCallContent,
    ToolChoice,
    ToolResultContent,
)
from commerce_model_runtime.providers.anthropic import AnthropicRuntime


class FakeMessages:
    pass


class FakeClient:
    messages = FakeMessages()


def request(*, tool_choice=None, reasoning=None, cache=None):
    return ModelRequest(
        target=ModelTarget("anthropic", "claude-sonnet-5"),
        system=[
            SystemSegment("static", SegmentStability.STATIC),
            SystemSegment("dynamic", SegmentStability.DYNAMIC),
        ],
        messages=[
            ModelMessage(role="user", content=[TextContent("hello")]),
            ModelMessage(
                role="assistant",
                content=[ToolCallContent("c1", "search_products", {"query": "tent"}, "p1")],
            ),
            ModelMessage(role="user", content=[ToolResultContent("c1", "result")]),
        ],
        tools=[
            FunctionToolSpec(
                "search_products",
                "Search products",
                {"type": "object", "properties": {"query": {"type": "string"}}},
                progressive=True,
            ),
            BuiltinToolSpec("web_search", {"max_uses": 3}),
        ],
        tool_choice=tool_choice or ToolChoice.auto(),
        max_tokens=2048,
        reasoning=reasoning,
        cache=cache,
    )


def test_request_maps_tool_choice_cache_progressive_tools_and_messages():
    runtime = AnthropicRuntime(client=FakeClient())
    body = runtime._build_request(request(cache=CachePolicy(enabled=True, rolling_conversation=True)))
    assert body["model"] == "claude-sonnet-5"
    assert body["tool_choice"] == {"type": "auto"}
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in body["system"][1]
    assert body["tools"][0]["eager_input_streaming"] is True
    assert body["tools"][1] == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": 3,
        "cache_control": {"type": "ephemeral"},
    }
    assert body["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "p1",
        "name": "search_products",
        "input": {"query": "tent"},
    }
    assert body["messages"][2]["content"][0]["tool_use_id"] == "p1"
    assert body["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_request_maps_specific_none_and_reasoning_levels():
    runtime = AnthropicRuntime(client=FakeClient())
    forced = runtime._build_request(
        request(
            tool_choice=ToolChoice.specific("search_products"),
            reasoning=ReasoningConfig(ReasoningEffort.LOW),
        )
    )
    assert forced["tool_choice"] == {"type": "tool", "name": "search_products"}
    assert forced["thinking"] == {"type": "adaptive"}
    assert forced["output_config"] == {"effort": "low"}

    disabled = runtime._build_request(
        request(tool_choice=ToolChoice.none(), reasoning=ReasoningConfig(ReasoningEffort.OFF))
    )
    assert disabled["tool_choice"] == {"type": "none"}
    assert disabled["thinking"] == {"type": "disabled"}
