# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral Shopping Messages runtime."""

from __future__ import annotations

import asyncio
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
from commerce_common.grounding import first_forced_tool
from commerce_common.memory import MemoryRuntime, MemoryStore, MemoryWriteFilter
from commerce_common.model_memory import extract_memory
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
from shopping_agent.backend import StorefrontBackend
from shopping_agent.config import ShoppingAgentConfig
from shopping_agent.enrichment import PRESENTATION_COMPONENTS
from shopping_agent.executor import ShoppingToolExecutor, build_memory
from shopping_agent.grounding import GROUNDING_RULES
from shopping_agent.prompt import build_dynamic_context, build_static_system
from shopping_agent.tools.registry import build_tools
from shopping_agent.types import Cart, ShoppingSessionContext, ShoppingSessionState, UserPreferences

logger = logging.getLogger(__name__)


class ShoppingAgent:
    """One provider-neutral deployment of the shopping Messages runtime."""

    def __init__(
        self,
        *,
        backend: StorefrontBackend,
        skills: SkillRegistry | None = None,
        skills_dir: Path | None = None,
        config: ShoppingAgentConfig | None = None,
        memory_store: MemoryStore | None = None,
        memory_write_filter: MemoryWriteFilter | None = None,
        runtime: ModelRuntime | None = None,
        runtimes: RuntimeRegistry | None = None,
        client: Any | None = None,
        extra_presentation_tools: Sequence[PresentationExtension] = (),
        executor_class: type[ShoppingToolExecutor] = ShoppingToolExecutor,
    ) -> None:
        supplied = sum(value is not None for value in (runtime, runtimes, client))
        if supplied > 1:
            raise ValueError("pass only one of runtime, runtimes, or client")
        if skills is None:
            skills = SkillRegistry.from_dir(skills_dir) if skills_dir else SkillRegistry([])
        self.config = config or ShoppingAgentConfig()
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
        self._specs: dict[str, PresentationComponent] = {
            **PRESENTATION_COMPONENTS,
            **{ext.name: ext for ext in self.extra_presentation_tools},
        }
        self._partial_ui_tools = partial_ui_tool_names(
            PRESENTATION_COMPONENTS, self.extra_presentation_tools
        )
        self._static_system = build_static_system(self.config, self.skills)
        self._tools = build_model_tools(
            build_tools(self.config, self.skills.names, self.extra_presentation_tools),
            self._partial_ui_tools,
        )

    async def stream_turn(
        self,
        messages: list[dict[str, Any]],
        session: ShoppingSessionContext,
        state: ShoppingSessionState | None = None,
    ) -> AsyncIterator[AgentEvent]:
        state = state if state is not None else ShoppingSessionState()
        bridge = LegacyConversationBridge(messages)
        turn_started = time.monotonic()
        preferences, cart, memory_facts, account = await self._prefetch(session)
        context = build_dynamic_context(
            preferences=preferences,
            memory_facts=memory_facts,
            cart=cart,
            page=session.page,
            now=session.local_now(),
            account=account,
            account_max_chars=self.config.max_context_chars,
        )
        system = build_system_segments(self._static_system, context)
        executor = self.executor_class(
            backend=self.backend,
            config=self.config,
            skills=self.skills,
            session=session,
            state=state,
            memory=self.memory,
            extensions=self.extra_presentation_tools,
        )
        forced_tool = first_forced_tool(
            GROUNDING_RULES, self.config, bridge.latest_user_text(), state
        )
        usage = host_usage_totals()
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
                        operation="shopping_turn",
                        data_classification="user_context",
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
                    if not tool_uses or force_text:
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
                    outcomes = await dispatcher.collect(dispatch_blocks)
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
                    stop_reason = "end_turn"
                    break
        finally:
            bridge.close_open_tool_calls(settled)

        cleared = compact_history(
            messages, last_prompt, self.config.compact_history_above_tokens, session.session_id
        )
        yield AgentEvent.turn_complete(stop_reason, usage, elapsed_ms(turn_started), cleared)

    async def update_memory(
        self, messages: list[dict[str, Any]], session: ShoppingSessionContext
    ) -> list[MemoryFact]:
        bridge = LegacyConversationBridge(messages)
        transcript = bridge.transcript_text(bridge.latest_exchange())
        target = self.config.memory_target()
        runtime = self.runtimes.resolve(target)
        return await extract_memory(
            self.memory,
            runtime,
            target,
            session.user_id,
            session.session_id,
            transcript,
        )

    async def _prefetch(
        self, session: ShoppingSessionContext
    ) -> tuple[UserPreferences | None, Cart | None, list[MemoryFact], dict[str, Any] | None]:
        preferences, account, cart, facts = await asyncio.gather(
            fetched(self.backend.get_preferences(session)),
            fetched(self.backend.get_account_context(session)),
            fetched(self.backend.get_cart(session) if self.config.enable_cart else None),
            fetched(self.memory.tier_one(session.user_id)),
        )
        return preferences, cart, list(facts or []), account
