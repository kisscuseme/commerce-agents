# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Prompt assembly for both the legacy Anthropic request shape and the provider-neutral model runtime."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from typing import Any

from commerce_model_runtime import (
    BuiltinToolSpec,
    CachePolicy,
    FunctionToolSpec,
    SegmentStability,
    SystemSegment,
    ToolSpec,
)


def context_clock(now: datetime) -> str:
    return now.replace(minute=0, second=0, microsecond=0).isoformat(timespec="minutes")


# Provider-neutral helpers. These express semantic intent only; adapters own wire syntax.
def build_system_segments(static_text: str, context: str) -> list[SystemSegment]:
    return [
        SystemSegment(static_text, SegmentStability.STATIC),
        SystemSegment(context, SegmentStability.DYNAMIC),
    ]


def build_cache_policy(rolling_breakpoint: bool = True) -> CachePolicy:
    return CachePolicy(enabled=True, rolling_conversation=rolling_breakpoint)


def build_model_tools(
    tools: list[dict[str, Any]],
    progressive_names: Collection[str] = (),
) -> list[ToolSpec]:
    progressive = frozenset(progressive_names)
    result: list[ToolSpec] = []
    for tool in tools:
        name = str(tool.get("name", ""))
        if "input_schema" in tool and name:
            result.append(
                FunctionToolSpec(
                    name=name,
                    description=str(tool.get("description", "")),
                    input_schema=dict(tool.get("input_schema") or {}),
                    progressive=name in progressive,
                )
            )
            continue
        raw_type = str(tool.get("type", ""))
        if raw_type.startswith("web_search_") or name == "web_search":
            result.append(
                BuiltinToolSpec(
                    "web_search",
                    {
                        key: value
                        for key, value in tool.items()
                        if key not in {"type", "name", "cache_control", "eager_input_streaming"}
                    },
                )
            )
            continue
        if raw_type.startswith("code_execution_") or name == "code_execution":
            result.append(
                BuiltinToolSpec(
                    "code_execution",
                    {
                        key: value
                        for key, value in tool.items()
                        if key
                        not in {
                            "type",
                            "name",
                            "cache_control",
                            "eager_input_streaming",
                            "allowed_callers",
                        }
                    },
                )
            )
            continue
        raise ValueError(f"unsupported tool definition: {tool!r}")
    return result


# Legacy helpers remain until both Messages API orchestrators migrate to ModelRuntime.
def build_system_blocks(static_text: str, context: str) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": context},
    ]


def with_eager_input(tools: list[dict[str, Any]], names: Collection[str]) -> list[dict[str, Any]]:
    return [t | {"eager_input_streaming": True} if t.get("name") in names else t for t in tools]


def with_tool_cache_control(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tools:
        return tools
    tools = [dict(t) for t in tools]
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


def build_request_messages(
    messages: list[dict[str, Any]],
    *,
    rolling_breakpoint: bool = True,
) -> list[dict[str, Any]]:
    if not messages:
        return []

    def without_marker(message: dict[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, list) or not any(
            isinstance(b, dict) and "cache_control" in b for b in content
        ):
            return message
        return message | {
            "content": [
                {k: v for k, v in b.items() if k != "cache_control"} if isinstance(b, dict) else b
                for b in content
            ]
        }

    def blocks(raw: Any) -> list[Any]:
        return [{"type": "text", "text": raw}] if isinstance(raw, str) else list(raw or [])

    request: list[dict[str, Any]] = []
    for message in messages:
        message = without_marker(message)
        if request and message.get("role") == "user" and request[-1].get("role") == "user":
            request[-1] = request[-1] | {
                "content": blocks(request[-1].get("content")) + blocks(message.get("content"))
            }
        else:
            request.append(message)
    if not rolling_breakpoint or len(request) < 2:
        return request
    content = blocks(request[-1].get("content"))
    if content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}
        request[-1] = request[-1] | {"content": content}
    return request
