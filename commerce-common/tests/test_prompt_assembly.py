# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta, timezone

from commerce_model_runtime import BuiltinToolSpec, FunctionToolSpec, SegmentStability
from commerce_common.prompt_assembly import (
    build_cache_policy,
    build_model_tools,
    build_system_segments,
    context_clock,
)

CONTEXT = "# Session context\n<data>{}</data>"


def test_system_segments_are_semantic_static_then_dynamic():
    static, context = build_system_segments("# Static identity and rules", CONTEXT)
    assert static.text == "# Static identity and rules"
    assert static.stability is SegmentStability.STATIC
    assert context.text == CONTEXT and context.stability is SegmentStability.DYNAMIC


def test_context_clock_is_hour_in_session_offset():
    lisbon_summer = timezone(timedelta(hours=1))
    assert context_clock(datetime(2026, 5, 30, 10, 37, 12, tzinfo=lisbon_summer)) == (
        "2026-05-30T10:00+01:00"
    )
    assert context_clock(datetime(2026, 5, 30, 10, 2)) == context_clock(
        datetime(2026, 5, 30, 10, 58)
    )


def test_cache_policy_expresses_intent_without_wire_fields():
    enabled = build_cache_policy(True)
    disabled_rolling = build_cache_policy(False)
    assert enabled.enabled and enabled.rolling_conversation
    assert disabled_rolling.enabled and not disabled_rolling.rolling_conversation


def test_tool_conversion_marks_progressive_function_semantically():
    tools = [
        {
            "name": "search_products",
            "description": "search",
            "input_schema": {"type": "object"},
        },
        {
            "name": "present_products",
            "description": "show",
            "input_schema": {"type": "object"},
        },
    ]
    search, present = build_model_tools(tools, {"present_products"})
    assert isinstance(search, FunctionToolSpec) and not search.progressive
    assert isinstance(present, FunctionToolSpec) and present.progressive


def test_builtin_tools_are_normalized_to_provider_neutral_kinds():
    web, code = build_model_tools(
        [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
            {"type": "code_execution_20260120", "name": "code_execution"},
        ]
    )
    assert web == BuiltinToolSpec(kind="web_search", options={"max_uses": 3})
    assert code == BuiltinToolSpec(kind="code_execution", options={})
