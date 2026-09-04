from __future__ import annotations

import logging
from typing import Any

from commerce_model_runtime import (
    FunctionToolSpec,
    ModelMessage,
    ModelOperation,
    ModelRequest,
    ModelRequestMetadata,
    ModelRuntime,
    ModelTarget,
    SegmentStability,
    SystemSegment,
    TextContent,
    ToolCallContent,
    ToolChoice,
    validate_capabilities,
)

from .memory import MemoryRuntime, render_memory_block
from .turn import session_tag

logger = logging.getLogger(__name__)

_RECORD_FACT_TOOL = FunctionToolSpec(
    name="record_fact",
    description="Record one new durable fact about the user.",
    input_schema={
        "type": "object",
        "properties": {
            "key": {"type": "string", "maxLength": 64},
            "value": {"type": "string", "maxLength": 200},
            "category": {
                "type": "string",
                "enum": ["preference", "constraint", "context"],
            },
        },
        "required": ["key", "value", "category"],
        "additionalProperties": False,
    },
)


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _same_fact(a: str, b: str) -> bool:
    if a in b or b in a:
        return True
    tokens_a = {token.strip(".,;:!?'\"()") for token in a.split()} - {""}
    tokens_b = {token.strip(".,;:!?'\"()") for token in b.split()} - {""}
    if not tokens_a or not tokens_b:
        return False
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b) >= 0.6


async def _extract_facts(
    memory: MemoryRuntime,
    runtime: ModelRuntime,
    target: ModelTarget,
    transcript: str,
    existing_facts: list[Any],
    *,
    session_id: str,
    max_new_facts: int = 3,
) -> list[Any]:
    validate_capabilities(
        ModelOperation.MEMORY_EXTRACTION,
        runtime.capabilities_for(target),
    )
    prompt = (
        f"Already saved facts:\n{render_memory_block(existing_facts)}\n\n"
        f"Conversation:\n{memory.fence.sanitize_text(transcript, 8000)}"
    )
    response = await runtime.complete(
        ModelRequest(
            target=target,
            system=[SystemSegment(memory.extraction_prompt, SegmentStability.STATIC)],
            messages=[ModelMessage(role="user", content=[TextContent(prompt)])],
            tools=[_RECORD_FACT_TOOL],
            tool_choice=ToolChoice.auto(),
            max_tokens=600,
            metadata=ModelRequestMetadata(
                operation="memory_extract",
                data_classification="persistent_personal_data",
            ),
        )
    )

    held = {fact.key: _normalize(fact.value) for fact in existing_facts}
    known = set(held.values())
    facts: list[Any] = []
    for block in response.message.content if response.message is not None else []:
        if not isinstance(block, ToolCallContent):
            continue
        if block.name != "record_fact" or len(facts) >= max_new_facts:
            continue
        data = block.arguments
        try:
            fact = memory.validate(
                str(data.get("key", "")),
                str(data.get("value", "")),
                str(data.get("category", "preference")),
                source_session_id=session_tag(session_id),
            )
        except (ValueError, TypeError):
            continue
        if not fact.key or not fact.value:
            continue
        value = _normalize(fact.value)
        current = held.get(fact.key)
        if current is not None:
            if value == current:
                continue
            known.discard(current)
        elif any(_same_fact(value, seen) for seen in known):
            continue
        held[fact.key] = value
        known.add(value)
        facts.append(fact)
    return facts


async def extract_memory(
    memory: MemoryRuntime,
    runtime: ModelRuntime,
    target: ModelTarget,
    subject_id: str,
    session_id: str,
    transcript: str,
) -> list[Any]:
    """Provider-neutral post-turn extraction; failures never fail the completed turn."""
    if not memory.enabled or memory.store is None or not transcript:
        return []
    try:
        generation = await memory.store.purge_generation(subject_id)
        existing = await memory.store.get_facts(subject_id)
        new_facts = await _extract_facts(
            memory,
            runtime,
            target,
            transcript,
            list(existing),
            session_id=session_id,
        )
        if not new_facts or await memory.store.purge_generation(subject_id) != generation:
            return []
        await memory.store.upsert_facts(subject_id, new_facts)
        return new_facts
    except Exception:
        logger.warning(
            "memory extraction failed for session %s; the turn continues without it",
            session_tag(session_id),
            exc_info=True,
        )
        return []
