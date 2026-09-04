# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers.

Anthropic wire-format fakes live in ``commerce_model_runtime.providers.anthropic_testing``;
this module re-exports them temporarily for backward compatibility with the existing test
suite and keeps only Commerce-specific helpers locally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from commerce_model_runtime.providers.anthropic_testing import (
    FakeBlock,
    FakeClient,
    FakeCreateClient,
    FakeStream,
    create_response,
    extraction_client,
    text_block,
    text_message,
    tool_calls_message,
    tool_use_block,
    tool_use_message,
)

from commerce_common.memory import InMemoryMemoryStore
from commerce_common.types import MemoryCategory, MemoryFact


def result_text(result: Any) -> str:
    """The text of a tool result in the Agent SDK dict shape or MCP result object."""
    blocks = result["content"] if isinstance(result, dict) else result.content
    parts = [
        block["text"] if isinstance(block, dict) else block.text
        for block in blocks
        if _is_text(block)
    ]
    return " ".join(parts)


def _is_text(block: Any) -> bool:
    return (
        block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
    ) == "text"


class SpyStore(InMemoryMemoryStore):
    """An in-memory store counting reads made through memory tools and prompt context."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get_facts(self, subject_id: str) -> list[MemoryFact]:
        self.reads += 1
        return await super().get_facts(subject_id)

    async def search_facts(self, subject_id: str, query: str) -> list[MemoryFact]:
        self.reads += 1
        return await super().search_facts(subject_id, query)

    def seed(self, subject_id: str, key: str, value: str, *, days_old: int = 0) -> "SpyStore":
        fact = MemoryFact(
            key=key,
            value=value,
            category=MemoryCategory.CONSTRAINT,
            updated_at=datetime.now(UTC) - timedelta(days=days_old),
        )
        self._data.setdefault(subject_id, {})[key] = fact
        return self

    def keys(self, subject_id: str) -> list[str]:
        return list(self._data.get(subject_id, {}))


__all__ = [
    "FakeBlock",
    "FakeClient",
    "FakeCreateClient",
    "FakeStream",
    "SpyStore",
    "create_response",
    "extraction_client",
    "result_text",
    "text_block",
    "text_message",
    "tool_calls_message",
    "tool_use_block",
    "tool_use_message",
]
