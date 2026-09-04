from commerce_model_runtime import BuiltinToolSpec, CachePolicy, FunctionToolSpec, SegmentStability
from commerce_common.prompt_assembly import build_cache_policy, build_model_tools, build_system_segments


def test_system_segments_express_static_and_dynamic_intent_without_wire_keys():
    static, dynamic = build_system_segments("static prompt", "dynamic context")
    assert static.text == "static prompt" and static.stability is SegmentStability.STATIC
    assert dynamic.text == "dynamic context" and dynamic.stability is SegmentStability.DYNAMIC
    assert not hasattr(static, "cache_control")


def test_tool_conversion_marks_progressive_function_tools_semantically():
    raw = [
        {"name": "search", "description": "Search", "input_schema": {"type": "object"}},
        {"name": "present_products", "description": "Cards", "input_schema": {"type": "object"}},
    ]
    tools = build_model_tools(raw, progressive_names={"present_products"})
    assert tools == [
        FunctionToolSpec("search", "Search", {"type": "object"}, progressive=False),
        FunctionToolSpec("present_products", "Cards", {"type": "object"}, progressive=True),
    ]


def test_anthropic_web_search_wire_definition_becomes_portable_builtin_intent():
    raw = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]
    assert build_model_tools(raw) == [BuiltinToolSpec("web_search", {"max_uses": 3})]


def test_cache_policy_carries_rolling_intent_only():
    assert build_cache_policy(True) == CachePolicy(enabled=True, rolling_conversation=True)
    assert build_cache_policy(False) == CachePolicy(enabled=True, rolling_conversation=False)
