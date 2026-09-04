# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral Merchant Messages runtime."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from commerce_model_runtime import (
    ModelOperation,
    ModelRequest,
    ModelRequestMetadata,
    ModelRuntime,
    RuntimeRegistry,
    ToolChoice,
    ToolChoiceMode,
    ToolResultContent,
    validate_capabilities,
)
from commerce_model_runtime.providers import AnthropicRuntime

from commerce_common.conversation import LegacyConversationBridge
from commerce_common.delegation import DelegateExtension
from commerce_common.grounding import first_forced_tool
from commerce_common.memory import MemoryRuntime, MemoryStore, MemoryWriteFilter
from commerce_common.model_round import (
    ModelRoundRunner,
    accumulate_model_usage,
    host_usage_totals,
    model_prompt_tokens,
)
from commerce_common.presentation import (
    PresentationComponent,
    PresentationExtension,
    partial_ui_tool_names,
)
from commerce_common.prompt_assembly import (
    build_cache_policy,
    build_model_tools,
    build_system_segments,
)
from commerce_common.skills import SkillRegistry
from commerce_common.streaming import AgentEvent, ToolOutcome
from commerce_common.turn import (
    EagerDispatcher,
    compact_history,
    elapsed_ms,
    fetched,
    outcome_events,
    round_closes_turn,
    session_tag,
)
from commerce_common.types import MemoryFact
from merchant_agent.backend import MerchantBackend
from merchant_agent.config import MerchantAgentConfig
from merchant_agent.enrichment import PRESENTATION_COMPONENTS
from merchant_agent.executor import MerchantToolExecutor, build_memory
from merchant_agent.gates import STAGING_FOLLOWTHROUGH_REMINDER, turn_attempted_staging
from merchant_agent.grounding import GROUNDING_RULES, change_requested
from merchant_agent.prompt import build_dynamic_context, build_static_system
from merchant_agent.tools.registry import build_tools
from merchant_agent.types import MerchantSessionContext, MerchantSessionState

from .analysis import build_analysis_delegate

logger = logging.getLogger(__name__)
HOST_TEXTS = frozenset({STAGING_FOLLOWTHROUGH_REMINDER})


class MerchantAgent:
    """One provider-neutral deployment of the merchant Messages runtime."""

    def __init__(
        self,
        *,
        backend: MerchantBackend,
        skills: SkillRegistry | None = None,
        skills_dir: Path | None = None,
        config: MerchantAgentConfig | None = None,
        memory_store: MemoryStore | None = None,
        memory_write_filter: MemoryWriteFilter | None = None,
        runtime: ModelRuntime | None = None,
        runtimes: RuntimeRegistry | None = None,
        client: Any | None = None,
        extra_presentation_tools: Sequence[PresentationExtension] = (),
        extra_delegates: Sequence[DelegateExtension] = (),
        executor_class: type[MerchantToolExecutor] = MerchantToolExecutor,
    ) -> None:
        supplied = sum(value is not None for value in (runtime, runtimes, client))
        if supplied > 1:
            raise ValueError("pass only one of runtime, runtimes, or client")
        if skills is None:
            skills = SkillRegistry.from_dir(skills_dir) if skills_dir else SkillRegistry([])
        self.config = config or MerchantAgentConfig()
        self.executor_class = executor_class
        self.backend = backend
        self.skills = skills
        self.memory: MemoryRuntime = build_memory(self.config, memory_store, memory_write_filter)

        target = self.config.model_target()
        if runtimes is not None:
            self.runtimes = runtimes
            self.runtime = runtimes.resolve(target)
        elif runtime is not None:
            if runtime.provider.lower() != target.provider.lower():
                raise ValueError(
                    f"runtime provider {runtime.provider!r} does not match configured provider {target.provider!r}"
                )
            self.runtime = runtime
            self.runtimes = RuntimeRegistry([runtime])
        else:
            if client is not None and target.provider.lower() != "anthropic":
                raise ValueError(
                    "client= is the Anthropic compatibility path; use runtime= for other providers"
                )
            self.runtime = AnthropicRuntime(client=client, timeout_s=self.config.request_timeout_s)
            self.runtimes = RuntimeRegistry([self.runtime])
        self.client = client if client is not None else getattr(self.runtime, "client", None)
        self.capability_plan = validate_capabilities(
            ModelOperation.MAIN_TURN,
            self.runtime.capabilities_for(target),
            enable_web_search=self.config.enable_web_search,
        )

        self.extra_presentation_tools = tuple(extra_presentation_tools)
        self.extra_delegates = tuple(extra_delegates)
        built_in = (
            [build_analysis_delegate(self.client, self.backend, self.config)]
            if self.config.enable_analysis
            else []
        )
        self.delegates: tuple[DelegateExtension, ...] = (*built_in, *self.extra_delegates)
        self._specs: dict[str, PresentationComponent] = {
            **PRESENTATION_COMPONENTS,
            **{ext.name: ext for ext in self.extra_presentation_tools},
        }
        self._partial_ui_tools = partial_ui_tool_names(
            PRESENTATION_COMPONENTS, self.extra_presentation_tools
        )
        self._static_system = build_static_system(self.config, self.skills)
        self._tools = build_model_tools(
            build_tools(
                self.config,
                self.skills.names,
                self.extra_presentation_tools,
                self.extra_delegates,
            ),
            self._partial_ui_tools,
        )

    async def stream_turn(
        self,
        messages: list[dict[str, Any]],
        session: MerchantSessionContext,
        state: MerchantSessionState | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = state if state is not None else MerchantSessionState()
        bridge = LegacyConversationBridge(messages, HOST_TEXTS)
        turn_started = time.monotonic()
        usage = host_usage_totals()
        merchant_context, memory_facts = await asyncio.gather(
            fetched(self.backend.get_merchant_context(session)),
            fetched(self.memory.tier_one(session.merchant_id)),
        )
        context = build_dynamic_context(
            merchant_context=merchant_context,
            memory_facts=list(memory_facts or []),
            now=session.local_now(),
            merchant_context_max_chars=self.config.max_context_chars,
        )
        system = build_system_segments(self._static_system, context)
        progress: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        executor = self.executor_class(
            backend=self.backend,
            config=self.config,
            skills=self.skills,
            session=session,
            state=state,
            memory=self.memory,
            extensions=self.extra_presentation_tools,
            delegates=self.delegates,
            progress=progress.put_nowait,
            usage=usage,
        )
        user_text = bridge.latest_user_text()
        forced_tool = first_forced_tool(GROUNDING_RULES, self.config, user_text, state)
        remind = change_requested(self.config, user_text)
        stop_reason: str | None = None
        last_prompt = 0
        provider_state = None
        settled: dict[str, ToolOutcome] = {}

        try:
            for round_index in range(self.config.max_tool_iterations + 1):
                force_text = round_index == self.config.max_tool_iterations
                if force_text:
                    tool_choice = ToolChoice.none()
                elif round_index == 0 and forced_tool:
                    tool_choice = ToolChoice.specific(forced_tool)
                else:
                    tool_choice = ToolChoice.auto()

                request = ModelRequest(
                    target=self.config.model_target(),
                    system=system,
                    tools=self._tools,
                    tool_choice=tool_choice,
                    messages=bridge.model_messages(),
                    max_tokens=self.config.max_tokens,
                    reasoning=self.config.reasoning_config(),
                    cache=build_cache_policy(
                        self.config.rolling_conversation_cache
                        and tool_choice.mode is ToolChoiceMode.AUTO
                    ),
                    provider_state=provider_state,
                    metadata=ModelRequestMetadata(
                        operation="merchant_turn",
                        data_classification="business_operational_data",
                        attributes={"round": str(round_index)},
                    ),
                )
                dispatcher = EagerDispatcher(
                    executor.execute, self.config.eager_tool_dispatch and not force_text
                )
                runner = ModelRoundRunner(
                    specs=self._specs,
                    partial_tools=self._partial_ui_tools,
                    state=state,
                    eager_frames=self.config.eager_partial_frames,
                )
                call_started = time.monotonic()
                persisted = False
                try:
                    async for event in runner.relay(
                        self.runtime.stream(request), dispatcher, executor.tool_call_event
                    ):
                        yield event
                    result = runner.result
                    if result.message is not None:
                        bridge.append_assistant(result.message)
                    persisted = True
                    provider_state = result.provider_state
                    stop_reason = result.stop_reason.value
                    accumulate_model_usage(usage, result.usage)
                    last_prompt = model_prompt_tokens(result.usage)
                    logger.info(
                        "model call session=%s round=%d provider=%s model=%s stop=%s input=%d output=%d elapsed_ms=%d",
                        session_tag(session.session_id),
                        round_index,
                        request.target.provider,
                        request.target.model,
                        stop_reason,
                        result.usage.input_tokens or 0,
                        result.usage.output_tokens or 0,
                        elapsed_ms(call_started),
                    )
                    tool_uses = list(result.tool_calls)
                    unreadable = result.malformed_call_ids
                    if turn_attempted_staging(block.name for block in tool_uses):
                        remind = False
                    if not tool_uses or force_text:
                        if remind and not force_text:
                            remind = False
                            bridge.append_host_text(STAGING_FOLLOWTHROUGH_REMINDER)
                            continue
                        break

                    for block in tool_uses:
                        if block.id in unreadable or not dispatcher.started(block.id):
                            yield executor.tool_call_event(
                                block.name, block.id, dict(block.arguments)
                            )
                    dispatch_blocks = [
                        SimpleNamespace(id=block.id, name=block.name, input=block.arguments)
                        for block in tool_uses
                    ]

                    async def execute_all(joined=dispatcher, blocks=dispatch_blocks):
                        try:
                            return await joined.collect(blocks)
                        finally:
                            progress.put_nowait(None)

                    pending = asyncio.create_task(execute_all())
                    try:
                        while (event := await progress.get()) is not None:
                            yield event
                        outcomes = await pending
                    finally:
                        if not pending.done():
                            pending.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await pending
                        while not progress.empty():
                            progress.get_nowait()
                finally:
                    if not persisted and runner.result.message is not None:
                        bridge.append_assistant(runner.result.message)
                    dispatcher.cancel()

                calls = list(zip(tool_uses, outcomes, strict=True))
                settled = {block.id: outcome for block, outcome in calls}
                for block, outcome in calls:
                    for event in outcome_events(block.name, block.id, outcome):
                        yield event
                bridge.append_tool_results(
                    ToolResultContent(block.id, outcome.result_text, outcome.is_error)
                    for block, outcome in calls
                )
                settled = {}
                if self.config.close_on_presentation and round_closes_turn(
                    ((block.name, outcome) for block, outcome in calls), executor.ends_clean
                ):
                    if remind:
                        remind = False
                        bridge.append_host_text(STAGING_FOLLOWTHROUGH_REMINDER)
                        continue
                    stop_reason = "end_turn"
                    break
        finally:
            bridge.close_open_tool_calls(settled)

        cleared = compact_history(
            messages, last_prompt, self.config.compact_history_above_tokens, session.session_id
        )
        yield AgentEvent.turn_complete(stop_reason, usage, elapsed_ms(turn_started), cleared)

    async def update_memory(
        self, messages: list[dict[str, Any]], session: MerchantSessionContext
    ) -> list[MemoryFact]:
        bridge = LegacyConversationBridge(messages, HOST_TEXTS)
        transcript = bridge.transcript_text(bridge.latest_exchange())
        return await self.memory.extract(
            self.client, session.merchant_id, session.session_id, transcript
        )
