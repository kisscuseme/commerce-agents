# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral prompt and tool assembly.

This module expresses stable/dynamic system segments, cache intent, and portable tool
contracts. Provider adapters own cache-control markers, progressive-input flags, and
all other wire-format details.
"""

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
                    kind="web_search",
                    options={
                        key: value
                        for key, value in tool.items()
                        if key not in {"type", "name"}
                    },
                )
            )
            continue

        if raw_type.startswith("code_execution_") or name == "code_execution":
            result.append(
                BuiltinToolSpec(
                    kind="code_execution",
                    options={
                        key: value
                        for key, value in tool.items()
                        if key not in {"type", "name", "allowed_callers"}
                    },
                )
            )
            continue

        raise ValueError(f"unsupported tool definition: {tool!r}")
    return result
