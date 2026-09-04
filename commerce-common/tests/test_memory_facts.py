# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import inspect
from datetime import UTC, datetime, timedelta

import pytest

from commerce_common import memory
from commerce_common.fencing import Fence
from commerce_common.memory import (
    DEFAULT_BLOCKED_PATTERNS,
    MEMORY_EXTRACTION_TEMPLATE,
    MEMORY_WRITE_REJECTED_TEXT,
    MemoryWriteFilter,
    MemoryWriteRejected,
    select_tier_one_facts,
    validate_fact,
    write_filter_for,
)
from commerce_common.types import MemoryCategory, MemoryFact

FENCE = Fence(label="test_data", notice="Data.")
DEFAULT_FILTER = write_filter_for()


def fact(
    key: str, value: str, category: str | None = None, *, write_filter=DEFAULT_FILTER
) -> MemoryFact:
    return validate_fact(key, value, category, fence=FENCE, write_filter=write_filter)


def test_validate_fact_normalizes_key_and_clamps_schema_caps():
    assert fact("Camping Style", "prefers lightweight gear").key == "camping_style"
    clamped = fact("k", "x" * 250)
    assert len(clamped.value) <= 200 and clamped.value.endswith("...[truncated]")
    assert len(fact("K " * 100, "v").key) <= 64


@pytest.mark.parametrize(
    "value",
    [
        "card ending in full: 4000 1234 5678 9010",
        "national id 123-45-6789",
        "call me on 010 555 0100 about deliveries",
        "loyalty account 001122334455",
        "pay from ZZ12ACME00001234567890",
        "reach me at avery@example.test",
    ],
)
def test_default_filter_rejects_identifiers_without_repeating_them(value):
    with pytest.raises(MemoryWriteRejected) as rejected:
        fact("note", value)
    assert str(rejected.value) == MEMORY_WRITE_REJECTED_TEXT


def test_filter_reads_key_and_host_checks_run_after_normalization():
    with pytest.raises(MemoryWriteRejected):
        fact("card 4000123456789010", "prefers this one")

    seen = []

    def detector(key, value):
        seen.append((key, value))
        return "employer" in value

    strict = MemoryWriteFilter.build(checks=[detector])
    assert fact("Coffee Setup", "buys whole beans", write_filter=strict).key == "coffee_setup"
    assert seen == [("coffee_setup", "buys whole beans")]
    with pytest.raises(MemoryWriteRejected):
        fact("job", "works for a named employer", write_filter=strict)


def test_deployment_patterns_add_to_defaults_and_none_disables_filter():
    strict = MemoryWriteFilter.build([r"(?i)\ballerg"])
    with pytest.raises(MemoryWriteRejected):
        fact("wool", "allergic to wool", write_filter=strict)
    assert fact("card", "4000123456789010", write_filter=None).value
    assert write_filter_for(("x",)) is write_filter_for(("x",))
    assert len(write_filter_for(("x", "y")).patterns) == len(DEFAULT_BLOCKED_PATTERNS) + 2


def test_extraction_template_continuations_keep_joining_space():
    source = inspect.getsource(memory)
    start = source.index('MEMORY_EXTRACTION_TEMPLATE = """')
    literal = source[start : source.index('"""', start + len('MEMORY_EXTRACTION_TEMPLATE = """'))]
    assert all(line.endswith(" \\") for line in literal.splitlines() if line.endswith("\\"))
    assert "\\n" not in literal


def test_memory_module_contains_no_provider_sdk_dependency():
    source = inspect.getsource(memory)
    assert "from anthropic" not in source
    assert "AsyncAnthropic" not in source
    assert "messages.create" not in source


def dated(key: str, category: str = "preference", days_ago: int = 0) -> MemoryFact:
    return MemoryFact(
        key=key,
        value=f"value of {key}",
        category=MemoryCategory(category),
        updated_at=datetime(2026, 6, 1, tzinfo=UTC) - timedelta(days=days_ago),
    )


def test_tier_one_keeps_constraints_and_most_recent_rest():
    facts = [
        dated("no_outdoor_space", "constraint", days_ago=400),
        *(dated(f"pref_{i}", days_ago=i) for i in range(9)),
    ]
    assert [f.key for f in select_tier_one_facts(facts, cap=8)] == [
        "no_outdoor_space",
        *(f"pref_{i}" for i in range(7)),
    ]
    undated = MemoryFact(key="undated", value="no timestamp", category=MemoryCategory.PREFERENCE)
    assert select_tier_one_facts([undated], cap=8) == [undated]
