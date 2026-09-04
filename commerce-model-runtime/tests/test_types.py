import json
from dataclasses import asdict

import pytest

from commerce_model_runtime import (
    CachePolicy,
    FunctionToolSpec,
    ModelMessage,
    ModelRequest,
    ModelTarget,
    ProviderState,
    ReasoningConfig,
    ReasoningEffort,
    SegmentStability,
    SystemSegment,
    TextContent,
    ToolChoice,
    ToolChoiceMode,
)


def test_provider_state_is_persistable_shape():
    state = ProviderState(provider="anthropic", data={"response_id": "r_123"})
    assert asdict(state)["data"] == {"response_id": "r_123"}
    assert json.dumps(asdict(state))


def test_provider_state_rejects_non_json_serializable_data():
    with pytest.raises(TypeError, match="JSON-serializable"):
        ProviderState(provider="anthropic", data={"bad": object()})


def test_specific_tool_choice_requires_name():
    choice = ToolChoice.specific("search_products")
    assert choice.mode is ToolChoiceMode.SPECIFIC
    assert choice.name == "search_products"


def test_empty_specific_tool_name_is_rejected():
    with pytest.raises(ValueError, match="tool name"):
        ToolChoice.specific("")


def test_model_request_carries_semantic_provider_neutral_intent():
    request = ModelRequest(
        target=ModelTarget(provider="anthropic", model="claude-sonnet-5"),
        system=[SystemSegment("static", stability=SegmentStability.STATIC)],
        messages=[ModelMessage(role="user", content=[TextContent("hello")])],
        tools=[
            FunctionToolSpec(
                name="search_products",
                description="Search products",
                input_schema={"type": "object", "properties": {}},
                progressive=True,
            )
        ],
        tool_choice=ToolChoice.auto(),
        max_tokens=1024,
        reasoning=ReasoningConfig(ReasoningEffort.LOW),
        cache=CachePolicy(rolling_conversation=True),
    )
    assert request.target.provider == "anthropic"
    assert request.system[0].stability is SegmentStability.STATIC
    assert request.tools[0].progressive is True
    assert request.tool_choice.mode is ToolChoiceMode.AUTO
    assert request.reasoning.effort is ReasoningEffort.LOW
