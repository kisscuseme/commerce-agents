# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Provider-independent turn helpers shared by Commerce runtimes.

Provider stream parsing and model usage normalization live in ``commerce-model-runtime``
and :mod:`commerce_common.model_round`. This module owns only host conversation helpers,
eager Commerce-tool dispatch, compaction, outcome events, and interruption closure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Collection, Iterable, Iterator, Mapping
from typing import Any

from .presentation import CHIPS_TOOL
from .streaming import AgentEvent, ToolOutcome

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 200
_EXCERPT_MAX_CHARS = 1200
CLEARED_RESULT = "[result cleared from an earlier turn; call the tool again if it is needed]"
UNREADABLE_INPUT_TEXT = (
    "The arguments for this call did not arrive as valid JSON, so it was not run. "
    "Send the call again."
)
INTERRUPTED_RESULT_TEXT = (
    "The turn was interrupted before this call returned; call it again if it is still needed."
)


def _text_blocks(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _is_user_text(message: dict[str, Any], host_texts: Collection[str]) -> bool:
    if message.get("role") != "user":
        return False
    texts = _text_blocks(message.get("content"))
    return bool(texts) and not all(text in host_texts for text in texts)


def latest_user_text(messages: list[dict[str, Any]], host_texts: Collection[str] = ()) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        texts = _text_blocks(content)
        if texts and all(text in host_texts for text in texts):
            continue
        if texts:
            return "\n".join(texts)
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "tool_result" for block in content
        ):
            continue
        return ""
    return ""


def latest_exchange(
    messages: list[dict[str, Any]], host_texts: Collection[str] = ()
) -> list[dict[str, Any]]:
    for index in range(len(messages) - 1, -1, -1):
        if _is_user_text(messages[index], host_texts):
            return messages[index:]
    return messages


def transcript_text(messages: list[dict[str, Any]], host_texts: Collection[str] = ()) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "")
        for text in _text_blocks(message.get("content")):
            if text and text not in host_texts:
                lines.append(f"{role}: {text}")
    return "\n".join(lines)


def session_tag(session_id: str | None) -> str:
    return hashlib.sha256(session_id.encode()).hexdigest()[:12] if session_id else "-"


def usage_totals() -> dict[str, int]:
    """Legacy host-facing usage accumulator; provider normalization lives elsewhere."""
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


def compact_history(
    messages: list[dict[str, Any]], last_prompt_tokens: int, max_tokens: int, session_id: str
) -> int:
    if not max_tokens or last_prompt_tokens < max_tokens:
        return 0
    size = len(json.dumps(messages))
    target = size // 2
    cleared = 0
    results = (
        block
        for message in messages
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("content"), str)
    )
    for block in results:
        if size <= target:
            break
        if len(block["content"]) > len(CLEARED_RESULT):
            size -= len(block["content"]) - len(CLEARED_RESULT)
            block["content"] = CLEARED_RESULT
            cleared += 1
    logger.info(
        "history compacted session=%s prompt_tokens=%d results_cleared=%d",
        session_tag(session_id),
        last_prompt_tokens,
        cleared,
    )
    return cleared


async def fetched(coro: Any) -> Any:
    if coro is None:
        return None
    name = getattr(coro, "__qualname__", type(coro).__name__)
    try:
        return await coro
    except Exception:
        logger.warning("prefetch %s failed and the turn continues without it", name, exc_info=True)
        return None


def _call_arguments(block: Any) -> dict[str, Any]:
    arguments = getattr(block, "arguments", None)
    if arguments is None:
        arguments = getattr(block, "input", None)
    return dict(arguments or {})


class EagerDispatcher:
    """Run canonical tool proposals exactly once, optionally before model generation ends."""

    def __init__(
        self, execute: Callable[[str, dict[str, Any]], Awaitable[ToolOutcome]], enabled: bool
    ) -> None:
        self._execute = execute
        self._enabled = enabled
        self._tasks: dict[str, asyncio.Future[ToolOutcome]] = {}

    def started(self, tool_use_id: str) -> bool:
        return tool_use_id in self._tasks

    def dispatch(self, name: str, tool_use_id: str, args: dict[str, Any] | None) -> bool:
        if not self._enabled or args is None or tool_use_id in self._tasks:
            return False
        self._tasks[tool_use_id] = asyncio.ensure_future(self._execute(name, args))
        return True

    def settle(self, tool_use_id: str, outcome: ToolOutcome) -> None:
        if tool_use_id in self._tasks:
            return
        settled: asyncio.Future[ToolOutcome] = asyncio.get_running_loop().create_future()
        settled.set_result(outcome)
        self._tasks[tool_use_id] = settled

    def settled_outcomes(self) -> dict[str, ToolOutcome]:
        """Completed successful task results safe to persist before interruption cleanup."""
        results: dict[str, ToolOutcome] = {}
        for tool_use_id, task in self._tasks.items():
            if not task.done() or task.cancelled():
                continue
            try:
                outcome = task.result()
            except BaseException:
                continue
            results[tool_use_id] = outcome
        return results

    async def collect(self, tool_uses: Iterable[Any]) -> list[ToolOutcome]:
        blocks = list(tool_uses)
        return await asyncio.gather(
            *(
                self._tasks[block.id]
                if block.id in self._tasks
                else self._execute(block.name, _call_arguments(block))
                for block in blocks
            )
        )

    def cancel(self) -> None:
        for task in self._tasks.values():
            task.cancel()


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def outcome_events(tool: str, tool_use_id: str, outcome: ToolOutcome) -> Iterator[AgentEvent]:
    for event in outcome.events:
        if event.type == "ui":
            event.data = {**event.data, "stream_id": tool_use_id}
        yield event
    keep_text = outcome.refused or len(outcome.result_text) < _SUMMARY_MAX_CHARS
    yield AgentEvent.tool_result(
        tool,
        tool_use_id,
        outcome.result_text if keep_text else "ok",
        outcome.is_error,
        status="blocked" if outcome.blocked else None,
        reason=outcome.blocked,
        excerpt=None if keep_text else outcome.result_text[:_EXCERPT_MAX_CHARS],
    )


def round_closes_turn(
    calls: Iterable[tuple[str, ToolOutcome]], clean: Callable[[str, ToolOutcome], bool]
) -> bool:
    calls = list(calls)
    return any(name == CHIPS_TOOL for name, _ in calls) and all(clean(*call) for call in calls)


def tool_result_block(tool_use_id: str, outcome: ToolOutcome) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": outcome.result_text,
        "is_error": outcome.is_error,
    }


def close_open_tool_uses(
    messages: list[dict[str, Any]], settled: Mapping[str, ToolOutcome] | None = None
) -> int:
    if not messages or messages[-1].get("role") != "assistant":
        return 0
    content = messages[-1].get("content")
    if not isinstance(content, list):
        return 0
    ids = [
        (block.get("id") if isinstance(block, dict) else getattr(block, "id", None))
        for block in content
        if (block.get("type") if isinstance(block, dict) else getattr(block, "type", None))
        == "tool_use"
    ]
    ids = [tool_use_id for tool_use_id in ids if tool_use_id]
    if not ids:
        return 0
    settled = settled or {}
    messages.append(
        {
            "role": "user",
            "content": [
                tool_result_block(tool_use_id, settled[tool_use_id])
                if tool_use_id in settled
                else {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": INTERRUPTED_RESULT_TEXT,
                    "is_error": True,
                }
                for tool_use_id in ids
            ],
        }
    )
    return len(ids)
