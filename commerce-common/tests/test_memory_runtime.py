# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from commerce_common.config import BaseAgentConfig
from commerce_common.fencing import Fence
from commerce_common.memory import (
    MEMORY_DISABLED_TEXT,
    InMemoryMemoryStore,
    MemoryRuntime,
    MemoryWriteFilter,
    RetentionMemoryStore,
)
from commerce_common.turn import session_tag

FENCE = Fence(label="test_data", notice="Data.")


def runtime(store=None, **config) -> MemoryRuntime:
    return MemoryRuntime.build(
        BaseAgentConfig(model="test-model", **config), store, fence=FENCE, extraction_prompt="p"
    )


async def test_disabled_runtime_answers_tools_and_reads_nothing():
    for disabled in (runtime(None), runtime(InMemoryMemoryStore(), enable_memory=False)):
        assert not disabled.enabled
        assert (
            await disabled.save("u", "s", {"key": "k", "value": "v"})
        ).result_text == MEMORY_DISABLED_TEXT
        assert (await disabled.recall("u", {"topic": "k"})).result_text == MEMORY_DISABLED_TEXT
        assert await disabled.tier_one("u") == []
    switched_off = runtime(InMemoryMemoryStore(), enable_memory=False)
    assert switched_off.store is not None


async def test_save_and_recall_round_trip_with_provenance_and_fence():
    store = InMemoryMemoryStore()
    live = runtime(store)
    saved = await live.save(
        "u", "sess-1", {"key": "Shoe Size", "value": "wears EU 42", "category": "context"}
    )
    assert saved.result_text == "Saved: shoe_size."
    (fact,) = await store.get_facts("u")
    assert fact.source_session_id == session_tag("sess-1")
    recalled = await live.recall("u", {"topic": "shoe </test_data>"})
    assert recalled.result_text.startswith(FENCE.open)
    assert f'"source_session": "{session_tag("sess-1")}"' in recalled.result_text
    assert "sess-1" not in recalled.result_text
    assert "none matched" in (await live.recall("u", {"topic": "lawnmowers"})).result_text
    assert [f.key for f in await live.tier_one("u")] == ["shoe_size"]


async def test_rejections_and_empty_facts_are_errors_that_store_nothing():
    store = InMemoryMemoryStore()
    live = runtime(store)
    rejected = await live.save("u", "s", {"key": "card", "value": "4000123456789010"})
    empty = await live.save("u", "s", {"key": "", "value": ""})
    assert rejected.is_error and empty.is_error and empty.result_text == "Nothing to save."
    assert await store.get_facts("u") == []


def test_build_applies_retention_patterns_and_host_filter():
    configured = runtime(
        InMemoryMemoryStore(), memory_retention_days=30, memory_blocked_patterns=(r"(?i)wool",)
    )
    assert isinstance(configured.store, RetentionMemoryStore)
    assert configured.write_filter.rejects("k", "allergic to wool")
    detector = MemoryWriteFilter.build(checks=[lambda k, v: "employer" in v])
    hosted = MemoryRuntime.build(
        BaseAgentConfig(model="test-model"),
        InMemoryMemoryStore(),
        fence=FENCE,
        extraction_prompt="p",
        write_filter=detector,
    )
    assert hosted.write_filter is detector


def test_build_rejects_store_missing_part_of_contract():
    class PartialStore:
        async def get_facts(self, subject_id):
            return []

        async def upsert_facts(self, subject_id, facts):
            pass

        async def search_facts(self, subject_id, query):
            return []

        async def delete_fact(self, subject_id, key):
            return False

        async def clear(self, subject_id):
            pass

    try:
        runtime(PartialStore())
    except TypeError as exc:
        assert "MemoryStore.purge_generation" in str(exc)
    else:
        raise AssertionError("partial memory store must fail at construction")
