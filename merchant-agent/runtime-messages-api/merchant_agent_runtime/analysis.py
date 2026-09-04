# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral merchant analysis delegate with optional hosted code execution."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from commerce_model_runtime import (
    ModelMessage,
    ModelOperation,
    ModelRequest,
    ModelRequestMetadata,
    ModelRuntime,
    ProviderState,
    SegmentStability,
    StopReason,
    SystemSegment,
    TextContent,
    ToolCallContent,
    ToolChoice,
    ToolResultContent,
    validate_capabilities,
)
from commerce_model_runtime.providers import AnthropicRuntime
from commerce_common.delegation import DelegateExtension, DelegationContext
from commerce_common.execution import without_status
from commerce_common.model_round import accumulate_model_usage
from commerce_common.prompt_assembly import build_cache_policy, build_model_tools
from commerce_common.skills import SkillRegistry
from commerce_common.streaming import AgentEvent
from merchant_agent import (
    AnalysisResult,
    AnalysisTable,
    MerchantAgentConfig,
    MerchantBackend,
    MerchantSessionState,
    check_analysis_sql,
)
from merchant_agent.analysis import (
    ANALYSIS_QUERY_TOOL,
    ANALYSIS_READ_TOOLS,
    ANALYSIS_TOOL,
    CODE_EXECUTION_TOOL_TYPE,
    REPORT_PROGRESS_TOOL,
    SUBMIT_ANALYSIS_TOOL,
    build_analysis_query_tool,
    build_analysis_system_prompt,
    build_analysis_tool_definition,
    build_report_progress_tool,
    build_submit_analysis_tool,
    cap_analysis_table,
    derive_metrics_payload,
    summarize_result_for_model,
)
from merchant_agent.executor import MerchantToolExecutor
from merchant_agent.fencing import MERCHANT_FENCE
from merchant_agent.tools.registry import build_tools

logger = logging.getLogger(__name__)

_STEP_VERBS = {
    "get_business_snapshot": "reading the snapshot",
    "query_metrics": "querying metrics",
    "get_campaign_performance": "reading campaigns",
    "search_listings": "scanning listings",
    ANALYSIS_QUERY_TOOL: "running a query",
}
_PROGRESS_ONLY_GRACE = 3


class _LegacyAnalysisAnthropicRuntime(AnthropicRuntime):
    """V1 compatibility for tests/callers that inject the old Anthropic client directly.

    The old analysis loop sent a one-text user message as a scalar string. The canonical
    runtime path is provider-neutral; only this deprecated ``client=`` bridge preserves
    that wire detail for existing consumers.
    """

    def _map_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        mapped = super()._map_messages(request)
        for source, encoded in zip(request.messages, mapped, strict=True):
            if (
                source.role == "user"
                and len(source.content) == 1
                and isinstance(source.content[0], TextContent)
                and isinstance(encoded.get("content"), list)
            ):
                encoded["content"] = source.content[0].text
        return mapped


def _coerce_runtime(runtime_or_client: Any) -> ModelRuntime:
    if (
        hasattr(runtime_or_client, "complete")
        and hasattr(runtime_or_client, "capabilities_for")
        and hasattr(runtime_or_client, "provider")
    ):
        return cast(ModelRuntime, runtime_or_client)
    return _LegacyAnalysisAnthropicRuntime(client=runtime_or_client)


def present_analysis(result: BaseModel, context: DelegationContext) -> tuple[Any, list[AgentEvent]]:
    analysis = cast(AnalysisResult, result)
    context.state.remember_analysis(analysis)
    return summarize_result_for_model(analysis), [
        AgentEvent.ui("metrics", derive_metrics_payload(analysis))
    ]


def build_analysis_delegate(
    runtime: ModelRuntime | Any,
    backend: MerchantBackend,
    config: MerchantAgentConfig,
) -> DelegateExtension:
    definition = build_analysis_tool_definition()
    runner = AnalysisRunner(runtime=_coerce_runtime(runtime), backend=backend, config=config)
    return DelegateExtension(
        name=ANALYSIS_TOOL,
        description=definition["description"],
        input_schema=definition["input_schema"],
        result_model=AnalysisResult,
        run=runner.run,
        present=present_analysis,
    )


def backend_supports_analysis_query(backend: MerchantBackend) -> bool:
    return type(backend).execute_analysis_query is not MerchantBackend.execute_analysis_query


class AnalysisRunner:
    def __init__(
        self,
        *,
        runtime: ModelRuntime | None = None,
        client: Any | None = None,
        backend: MerchantBackend,
        config: MerchantAgentConfig,
    ) -> None:
        if runtime is not None and client is not None:
            raise ValueError("pass runtime= or client=, not both")
        if runtime is None:
            runtime = _LegacyAnalysisAnthropicRuntime(client=client)
        self._runtime = runtime
        self._backend = backend
        self._config = config
        self._target = config.analysis_target()
        operation = (
            ModelOperation.HOSTED_ANALYSIS
            if config.analysis_use_code_execution
            else ModelOperation.PORTABLE_ANALYSIS
        )
        validate_capabilities(
            operation,
            runtime.capabilities_for(self._target),
            require_hosted_code_execution=config.analysis_use_code_execution,
        )
        self._system = build_analysis_system_prompt(config)
        self._sql_supported = backend_supports_analysis_query(backend)
        # ``_tools`` stays as the historical raw contract for inspection/backward tests;
        # model calls use the provider-neutral conversion exclusively.
        self._tools = self._build_tools()
        self._model_tools = build_model_tools(self._tools)

    def _build_tools(self) -> list[dict[str, Any]]:
        registry = build_tools(self._config, [])
        sql_only = self._sql_supported and self._config.analysis_sql_only
        tools = (
            []
            if sql_only
            else [
                without_status(tool)
                for tool in registry
                if tool.get("name") in ANALYSIS_READ_TOOLS
            ]
        )
        tools.append(build_submit_analysis_tool())
        tools.append(build_report_progress_tool())
        if self._sql_supported:
            tools.append(build_analysis_query_tool())
        if self._config.analysis_use_code_execution:
            tools = [
                {**tool, "allowed_callers": [CODE_EXECUTION_TOOL_TYPE]}
                if "input_schema" in tool
                else tool
                for tool in tools
            ]
            tools.append({"type": CODE_EXECUTION_TOOL_TYPE, "name": "code_execution"})
        return tools

    async def _task_brief(self, session: Any, args: dict[str, Any]) -> str:
        def _clamp(value: Any, limit: int = 300) -> Any:
            if isinstance(value, str):
                return value[:limit]
            if isinstance(value, list):
                return [_clamp(item, 80) for item in value[:8]]
            return value

        brief = {key: _clamp(value) for key, value in args.items() if value}
        text = "Analysis task:\n" + json.dumps(brief, ensure_ascii=False, indent=2)
        if self._sql_supported:
            try:
                schema = await self._backend.get_analysis_schema(session)
            except Exception:
                logger.warning("get_analysis_schema failed; briefing without it", exc_info=True)
                schema = None
            if schema:
                text += "\n\nQueryable tables (reference data):\n" + MERCHANT_FENCE.fence_payload(
                    {"schema": schema}, self._config.max_fenced_chars
                )
        return text

    async def run(self, context: DelegationContext, args: dict[str, Any]) -> AnalysisResult:
        trace: list[str] = []
        series_names: list[str] = []
        try:
            async with asyncio.timeout(self._config.analysis_timeout_s):
                return await self._run_loop(context, args, trace, series_names)
        except TimeoutError:
            raise ValueError(
                f"the analysis run hit its {self._config.analysis_timeout_s:g}s time "
                f"budget before submitting. Iterations so far: {'; '.join(trace) or 'none'}; "
                f"series fetched: {', '.join(series_names) or 'none'}. "
                "Reuse what is already gathered or ask a narrower question."
            ) from None

    async def _run_loop(
        self,
        context: DelegationContext,
        args: dict[str, Any],
        trace: list[str],
        series_names: list[str],
    ) -> AnalysisResult:
        messages: list[ModelMessage] = [
            ModelMessage(
                role="user",
                content=[TextContent(await self._task_brief(context.session, args))],
            )
        ]
        nudged = False
        provider_state: ProviderState | None = None
        iterations = 0
        progress_grace_used = 0
        step = 0
        last_tool_names: list[str] = []

        while iterations < self._config.max_analysis_iterations:
            step += 1
            self._auto_progress(context, step, last_tool_names)
            response = await self._runtime.complete(
                ModelRequest(
                    target=self._target,
                    system=[SystemSegment(self._system, SegmentStability.STATIC)],
                    tools=self._model_tools,
                    tool_choice=ToolChoice.auto(),
                    messages=messages,
                    max_tokens=self._config.analysis_max_tokens,
                    reasoning=self._config.reasoning_config(),
                    cache=build_cache_policy(False),
                    provider_state=provider_state,
                    metadata=ModelRequestMetadata(
                        operation="merchant_analysis",
                        data_classification="business_operational_data",
                        attributes={"step": str(step)},
                    ),
                )
            )
            provider_state = response.provider_state
            if context.usage is not None:
                accumulate_model_usage(context.usage, response.usage)
            if response.message is not None and response.message.content:
                messages.append(response.message)

            tool_uses = [
                block
                for block in (response.message.content if response.message else [])
                if isinstance(block, ToolCallContent)
            ]
            kinds = sorted(
                {
                    type(block).__name__
                    for block in (response.message.content if response.message else [])
                }
            )
            trace.append(f"{response.stop_reason.value}:" + (",".join(kinds) or "empty"))
            is_progress_only = bool(tool_uses) and all(
                block.name == REPORT_PROGRESS_TOOL for block in tool_uses
            )
            if is_progress_only and progress_grace_used < _PROGRESS_ONLY_GRACE:
                progress_grace_used += 1
            else:
                iterations += 1
            last_tool_names = [block.name for block in tool_uses]
            if not tool_uses:
                if response.stop_reason is StopReason.PAUSE:
                    continue
                if nudged:
                    break
                nudged = True
                messages.append(
                    ModelMessage(
                        role="user",
                        content=[
                            TextContent(
                                f"Submit now with {SUBMIT_ANALYSIS_TOOL} — either the findings, "
                                "or a submission stating why the data cannot answer the question."
                            )
                        ],
                    )
                )
                continue

            submitted: AnalysisResult | None = None
            tool_results: list[ToolResultContent] = []
            for block in tool_uses:
                tool_input = dict(block.arguments)
                if block.name == SUBMIT_ANALYSIS_TOOL:
                    try:
                        submitted = AnalysisResult.model_validate(tool_input)
                        result_text, is_error = "Analysis submitted.", False
                    except ValidationError as invalid:
                        issues = "; ".join(
                            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                            for error in invalid.errors()
                        )
                        result_text = f"Invalid submission — {issues}. Fix and submit again."
                        is_error = True
                else:
                    result_text, is_error = await self._execute(
                        context, block.name, tool_input, series_names
                    )
                tool_results.append(
                    ToolResultContent(
                        tool_call_id=block.id,
                        content=result_text,
                        is_error=is_error,
                    )
                )
            messages.append(ModelMessage(role="user", content=tool_results))
            if submitted is not None:
                return submitted

        raise ValueError(
            "the analysis run ended without a submission — try a narrower question "
            f"(iterations: {'; '.join(trace)})"
        )

    def _auto_progress(self, context: DelegationContext, step: int, last_tools: list[str]) -> None:
        if context.emit_status is None or step == 1:
            return
        verbs = sorted({_STEP_VERBS.get(name, "working") for name in last_tools} or {"working"})
        context.emit_status(f"analysis: step {step} — {', '.join(verbs)}")

    def _sanitize(self, text: str, max_chars: int | None) -> str:
        return MERCHANT_FENCE.sanitize_text(text, max_chars)

    def _fence(self, payload: Any) -> str:
        return MERCHANT_FENCE.fence_payload(payload, self._config.max_fenced_chars)

    async def _read(
        self,
        context: DelegationContext,
        name: str,
        tool_input: dict[str, Any],
        series_names: list[str],
    ) -> tuple[str, bool]:
        scratch = MerchantSessionState()
        reads = MerchantToolExecutor(
            backend=self._backend,
            config=self._config,
            skills=SkillRegistry([]),
            session=context.session,
            state=scratch,
        )
        outcome = await reads.execute(name, tool_input)
        if scratch.latest_snapshot is not None:
            context.state.remember_snapshot(scratch.latest_snapshot)
        for series in scratch.seen_series.values():
            context.state.remember_series(series)
            series_names.append(series.metric)
        return outcome.result_text, outcome.is_error

    async def _execute(
        self,
        context: DelegationContext,
        name: str,
        tool_input: dict[str, Any],
        series_names: list[str],
    ) -> tuple[str, bool]:
        if name == REPORT_PROGRESS_TOOL:
            message = self._sanitize(str(tool_input.get("message", "")), None)
            if message and context.emit_status is not None:
                context.emit_status(message)
            return "Noted — continue the analysis.", False
        if name in ANALYSIS_READ_TOOLS:
            return await self._read(context, name, tool_input, series_names)
        if name != ANALYSIS_QUERY_TOOL or not self._sql_supported:
            return f"Unknown tool in the analysis context: {name}", True
        try:
            return await self._run_query(context.session, str(tool_input.get("sql", "")))
        except TimeoutError:
            return (
                f"{name} timed out after {self._config.analysis_query_timeout_s:g}s. "
                "Narrow the query and try again.",
                True,
            )
        except Exception as error:
            logger.warning("analysis tool %s failed", name, exc_info=True)
            return f"{name} failed: {self._sanitize(str(error), 200) or 'unavailable'}", True

    async def _run_query(self, session: Any, sql: str) -> tuple[str, bool]:
        if reason := check_analysis_sql(sql):
            return (
                f"Query refused: {reason}. Analysis queries are a single read-only "
                "SELECT statement.",
                True,
            )
        async with asyncio.timeout(self._config.analysis_query_timeout_s):
            table = await self._backend.execute_analysis_query(session, sql)
        if table is None:
            return "SQL analysis is not supported by this deployment.", True
        capped = cap_analysis_table(
            table if isinstance(table, AnalysisTable) else AnalysisTable.model_validate(table),
            self._config,
        )
        return self._fence(capped.model_dump(mode="json", exclude_none=True)), False
