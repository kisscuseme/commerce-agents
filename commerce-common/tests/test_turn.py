# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from commerce_common.streaming import AgentEvent, ToolOutcome
from commerce_common.turn import (
    CLEARED_RESULT,
    EagerDispatcher,
    INTERRUPTED_RESULT_TEXT,
    close_open_tool_uses,
    compact_history,
    fetched,
    latest_exchange,
    latest_user_text,
    outcome_events,
    round_closes_turn,
    session_tag,
    tool_result_block,
    transcript_text,
)

REMINDER = "Host check: stage it."
CONVERSATION = [
    {"role": "user", "content": "first ask"},
    {"role": "assistant", "content": [{"type": "text", "text": "first reply"}]},
    {"role": "user", "content": [{"type": "text", "text": "drop the price"}]},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "want me to?"}]},
    {"role": "user", "content": [{"type": "text", "text": REMINDER}]},
    {"role": "assistant", "content": [{"type": "text", "text": "staged"}]},
]


def test_conversation_helpers_skip_host_and_tool_messages():
    assert latest_user_text(CONVERSATION[:5]) == "drop the price"
    assert latest_user_text(CONVERSATION, {REMINDER}) == "drop the price"
    assert latest_exchange(CONVERSATION, {REMINDER}) == CONVERSATION[2:]
    assert transcript_text(latest_exchange(CONVERSATION, {REMINDER}), {REMINDER}).splitlines() == [
        "user: drop the price",
        "assistant: want me to?",
        "assistant: staged",
    ]


@pytest.mark.asyncio
async def test_eager_dispatcher_accepts_canonical_arguments_and_runs_once():
    calls = []

    async def execute(name, args):
        calls.append((name, args))
        return ToolOutcome("ok")

    block = SimpleNamespace(id="c1", name="search", arguments={"q": "tent"})
    dispatcher = EagerDispatcher(execute, enabled=True)
    assert dispatcher.dispatch(block.name, block.id, block.arguments)
    outcomes = await dispatcher.collect([block])
    assert [o.result_text for o in outcomes] == ["ok"]
    assert calls == [("search", {"q": "tent"})]


@pytest.mark.asyncio
async def test_completed_eager_outcome_can_be_recovered_before_cancel():
    finished = asyncio.Event()

    async def execute(name, args):
        finished.set()
        return ToolOutcome("write completed")

    dispatcher = EagerDispatcher(execute, enabled=True)
    assert dispatcher.dispatch("stage_price_update", "c-write", {"price": 26})
    await asyncio.wait_for(finished.wait(), 0.2)
    await asyncio.sleep(0)

    settled = dispatcher.settled_outcomes()

    assert settled["c-write"].result_text == "write completed"
    dispatcher.cancel()


DISPLAYED = "Displayed."
PRESENTS = {"present_products", "present_suggestions"}


def _clean(name, outcome):
    return name in PRESENTS and not outcome.refused and outcome.result_text == DISPLAYED


def test_only_clean_presentation_round_with_chips_closes_turn():
    card = ("present_products", ToolOutcome(DISPLAYED))
    chips = ("present_suggestions", ToolOutcome(DISPLAYED))
    assert round_closes_turn([card, chips], _clean)
    assert not round_closes_turn([card], _clean)
    assert not round_closes_turn([("search_products", ToolOutcome("data")), chips], _clean)


class _Backend:
    async def get_cart(self):
        return "cart"

    async def get_preferences(self):
        raise ConnectionError("profile service down")


@pytest.mark.asyncio
async def test_fetched_returns_none_for_failed_prefetch_and_logs(caplog):
    with caplog.at_level(logging.WARNING, logger="commerce_common.turn"):
        assert await fetched(_Backend().get_cart()) == "cart"
        assert await fetched(_Backend().get_preferences()) is None
    assert any("get_preferences" in record.getMessage() for record in caplog.records)


def test_outcome_events_stamp_ui_and_summarize_long_results():
    outcome = ToolOutcome("x" * 500, [AgentEvent.ui("card", {"a": 1})])
    ui, result = outcome_events("tool", "t1", outcome)
    assert ui.data["stream_id"] == "t1"
    assert result.data["summary"] == "ok" and result.data["excerpt"] == "x" * 500


def test_tool_result_and_interruption_closure_preserve_conversation_validity():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
        }
    ]
    assert close_open_tool_uses(messages) == 1
    assert messages[-1]["content"][0]["content"] == INTERRUPTED_RESULT_TEXT
    assert tool_result_block("t2", ToolOutcome.error("no"))["is_error"] is True


def _long_conversation(rounds):
    messages = [{"role": "user", "content": "find tents"}]
    for index in range(rounds):
        call_id = f"t{index}"
        messages += [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": call_id, "name": "x", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": "x" * 1000}
                ],
            },
        ]
    messages.append({"role": "user", "content": "and a stove"})
    return messages


def test_compaction_clears_oldest_results_after_threshold():
    messages = _long_conversation(6)
    size = len(json.dumps(messages))
    cleared = compact_history(messages, 100_000, 100_000, "s-1")
    bodies = [
        block["content"]
        for message in messages
        if isinstance(message["content"], list)
        for block in message["content"]
        if block["type"] == "tool_result"
    ]
    assert cleared == 4 and bodies[:4] == [CLEARED_RESULT] * 4
    assert len(json.dumps(messages)) <= size // 2


def test_session_tag_is_stable_short_and_not_credential():
    tag = session_tag("sess-credential-1")
    assert tag == session_tag("sess-credential-1") and len(tag) == 12
    assert tag != "sess-credential-1" and session_tag(None) == "-"
