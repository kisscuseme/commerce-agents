from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ..capabilities import ModelCapabilities
from ..errors import (
    AuthenticationError,
    InvalidRequestError,
    ModelRuntimeError,
    ModelUnavailableError,
    ProviderProtocolError,
    RateLimitError,
    StreamInterruptedError,
    TransientProviderError,
)
from ..events import (
    ModelEvent,
    ResponseCompleted,
    TextDelta,
    ToolArgumentsDelta,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    UsageUpdated,
)
from ..types import (
    BuiltinToolSpec,
    FunctionToolSpec,
    ModelContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelTarget,
    ModelUsage,
    ProviderOpaqueContent,
    ProviderState,
    ReasoningEffort,
    SegmentStability,
    StopReason,
    TextContent,
    ToolCallContent,
    ToolChoiceMode,
    ToolResultContent,
)

_WEB_SEARCH_TYPE = "web_search_20250305"
_CODE_EXECUTION_TYPE = "code_execution_20260120"


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _non_none(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if value is not None}


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(exclude_none=True, exclude={"citations"}))
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if item is not None}
    raise TypeError(f"cannot convert provider block {type(value).__name__} to a mapping")


class AnthropicRuntime:
    provider = "anthropic"

    def __init__(self, client: Any | None = None, *, timeout_s: float = 120.0) -> None:
        if client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:  # pragma: no cover - exercised only in real installs
                raise ImportError(
                    "AnthropicRuntime requires the 'anthropic' optional dependency"
                ) from exc
            client = AsyncAnthropic(timeout=timeout_s)
        self.client = client

    def capabilities_for(self, target: ModelTarget) -> ModelCapabilities:
        if target.provider.strip().lower() != self.provider:
            return ModelCapabilities()
        return ModelCapabilities.full()

    def _map_tool_choice(self, request: ModelRequest) -> dict[str, str]:
        choice = request.tool_choice
        if choice.mode is ToolChoiceMode.AUTO:
            return {"type": "auto"}
        if choice.mode is ToolChoiceMode.NONE:
            return {"type": "none"}
        assert choice.name is not None
        return {"type": "tool", "name": choice.name}

    def _map_system(self, request: ModelRequest) -> list[dict[str, Any]]:
        cache_enabled = bool(request.cache and request.cache.enabled)
        blocks: list[dict[str, Any]] = []
        for segment in request.system:
            block: dict[str, Any] = {"type": "text", "text": segment.text}
            if cache_enabled and segment.stability is SegmentStability.STATIC:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        return blocks

    def _map_tools(self, request: ModelRequest) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        has_code_execution = any(
            isinstance(tool, BuiltinToolSpec) and tool.kind == "code_execution"
            for tool in request.tools
        )
        for tool in request.tools:
            if isinstance(tool, FunctionToolSpec):
                mapped: dict[str, Any] = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                if tool.progressive:
                    mapped["eager_input_streaming"] = True
                if has_code_execution:
                    mapped["allowed_callers"] = [_CODE_EXECUTION_TYPE]
                tools.append(mapped)
                continue
            if tool.kind == "web_search":
                tools.append({"type": _WEB_SEARCH_TYPE, "name": "web_search", **tool.options})
                continue
            if tool.kind == "code_execution":
                tools.append({"type": _CODE_EXECUTION_TYPE, "name": "code_execution", **tool.options})
                continue
            raise ValueError(f"unsupported Anthropic built-in tool kind: {tool.kind}")
        if tools and request.cache and request.cache.enabled:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        return tools

    def _map_messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        provider_ids: dict[str, str] = {}
        mapped_messages: list[dict[str, Any]] = []
        for message in request.messages:
            content: list[dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, TextContent):
                    content.append({"type": "text", "text": block.text})
                elif isinstance(block, ToolCallContent):
                    provider_id = block.provider_tool_call_id or block.id
                    provider_ids[block.id] = provider_id
                    content.append(
                        {"type": "tool_use", "id": provider_id, "name": block.name, "input": block.arguments}
                    )
                elif isinstance(block, ToolResultContent):
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": provider_ids.get(block.tool_call_id, block.tool_call_id),
                            "content": block.content,
                            **({"is_error": True} if block.is_error else {}),
                        }
                    )
                elif isinstance(block, ProviderOpaqueContent):
                    if block.provider != self.provider:
                        raise ValueError(
                            f"cannot send {block.provider!r} opaque content through AnthropicRuntime"
                        )
                    content.append(dict(block.data))
                else:  # pragma: no cover
                    raise TypeError(f"unsupported model content: {type(block).__name__}")
            mapped_messages.append({"role": message.role, "content": content})

        if (
            mapped_messages
            and request.cache
            and request.cache.enabled
            and request.cache.rolling_conversation
            and len(mapped_messages) >= 2
        ):
            last = mapped_messages[-1]
            if last["content"]:
                last_content = list(last["content"])
                last_content[-1] = {**last_content[-1], "cache_control": {"type": "ephemeral"}}
                mapped_messages[-1] = {**last, "content": last_content}
        return mapped_messages

    def _reasoning_fields(self, request: ModelRequest) -> dict[str, Any]:
        if request.reasoning is None:
            return {}
        effort = request.reasoning.effort
        if effort is ReasoningEffort.OFF:
            return {"thinking": {"type": "disabled"}}
        fields: dict[str, Any] = {"thinking": {"type": "adaptive"}}
        if effort is not ReasoningEffort.DEFAULT:
            fields["output_config"] = {"effort": effort.value}
        return fields

    def _build_request(self, request: ModelRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.target.model,
            "max_tokens": request.max_tokens,
            "system": self._map_system(request),
            "messages": self._map_messages(request),
            **self._reasoning_fields(request),
        }
        tools = self._map_tools(request)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = self._map_tool_choice(request)
        if request.provider_state is not None:
            if request.provider_state.provider != self.provider:
                raise ValueError(
                    f"cannot send {request.provider_state.provider!r} provider state through AnthropicRuntime"
                )
            container = request.provider_state.data.get("container")
            if container:
                body["container"] = container
        return body

    def _usage(self, raw: Any) -> ModelUsage:
        if raw is None:
            return ModelUsage()
        cache_creation = _get(raw, "cache_creation_input_tokens")
        details = _non_none({"cache_creation_input_tokens": cache_creation})
        input_tokens = _get(raw, "input_tokens")
        output_tokens = _get(raw, "output_tokens")
        cached_input = _get(raw, "cache_read_input_tokens")
        total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
        return ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input,
            total_tokens=total,
            provider_details=details,
        )

    def _stop_reason(self, raw: str | None) -> StopReason:
        return {
            "end_turn": StopReason.END_TURN,
            "tool_use": StopReason.TOOL_USE,
            "max_tokens": StopReason.MAX_TOKENS,
            "pause_turn": StopReason.PAUSE,
            "refusal": StopReason.CONTENT_FILTER,
        }.get(raw, StopReason.UNKNOWN)

    def _message(self, raw: Any) -> ModelMessage:
        content: list[ModelContent] = []
        for block in _get(raw, "content", []) or []:
            block_type = _get(block, "type")
            if block_type == "text":
                content.append(TextContent(str(_get(block, "text", ""))))
            elif block_type == "tool_use":
                provider_id = str(_get(block, "id", ""))
                content.append(
                    ToolCallContent(
                        id=provider_id,
                        name=str(_get(block, "name", "")),
                        arguments=dict(_get(block, "input", {}) or {}),
                        provider_tool_call_id=provider_id,
                    )
                )
            else:
                content.append(ProviderOpaqueContent(provider=self.provider, data=_plain(block)))
        return ModelMessage(role="assistant", content=content)

    def _response(self, raw: Any) -> ModelResponse:
        container = _get(raw, "container")
        container_id = _get(container, "id") if container is not None else None
        return ModelResponse(
            message=self._message(raw),
            stop_reason=self._stop_reason(_get(raw, "stop_reason")),
            usage=self._usage(_get(raw, "usage")),
            provider_state=(
                ProviderState(self.provider, {"container": container_id}) if container_id else None
            ),
            provider_request_id=_get(raw, "id"),
        )

    def _normalize_error(self, error: Exception, request: ModelRequest) -> ModelRuntimeError:
        status = _get(error, "status_code")
        request_id = _get(error, "request_id")
        kwargs = {
            "provider": self.provider,
            "model": request.target.model,
            "provider_request_id": request_id,
        }
        if status in (401, 403):
            return AuthenticationError(str(error), **kwargs)
        if status == 429:
            return RateLimitError(str(error), **kwargs)
        if status in (400, 404, 409, 422):
            return InvalidRequestError(str(error), **kwargs)
        if status in (408, 500, 502, 503, 504):
            return TransientProviderError(str(error), **kwargs)
        if status == 529:
            return ModelUnavailableError(str(error), **kwargs)
        return ProviderProtocolError(str(error), **kwargs)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        body = self._build_request(request)
        tools: dict[int, dict[str, Any]] = {}
        emitted = False
        try:
            async with self.client.messages.stream(**body) as stream:
                async for raw in stream:
                    event_type = _get(raw, "type")
                    if event_type == "message_start":
                        emitted = True
                        yield UsageUpdated(self._usage(_get(_get(raw, "message"), "usage")))
                    elif event_type == "content_block_start":
                        block = _get(raw, "content_block")
                        if _get(block, "type") == "tool_use":
                            index = int(_get(raw, "index", 0))
                            tool = {
                                "id": str(_get(block, "id", "")),
                                "name": str(_get(block, "name", "")),
                                "parts": [],
                            }
                            tools[index] = tool
                            emitted = True
                            yield ToolCallStarted(
                                id=tool["id"],
                                name=tool["name"],
                                provider_tool_call_id=tool["id"],
                            )
                    elif event_type == "content_block_delta":
                        delta = _get(raw, "delta")
                        delta_type = _get(delta, "type")
                        if delta_type == "text_delta":
                            emitted = True
                            yield TextDelta(str(_get(delta, "text", "")))
                        elif delta_type == "input_json_delta":
                            index = int(_get(raw, "index", 0))
                            part = str(_get(delta, "partial_json", ""))
                            if index in tools:
                                tools[index]["parts"].append(part)
                                emitted = True
                                yield ToolArgumentsDelta(tools[index]["id"], part)
                    elif event_type == "content_block_stop":
                        index = int(_get(raw, "index", 0))
                        tool = tools.get(index)
                        if tool is not None:
                            raw_json = "".join(tool["parts"])
                            try:
                                arguments = json.loads(raw_json or "{}")
                                if not isinstance(arguments, dict):
                                    raise ValueError("tool arguments must be an object")
                            except (json.JSONDecodeError, ValueError):
                                emitted = True
                                yield ToolCallFailed(
                                    id=tool["id"],
                                    name=tool["name"],
                                    reason="tool input was not valid JSON",
                                    provider_tool_call_id=tool["id"],
                                )
                            else:
                                emitted = True
                                yield ToolCallCompleted(
                                    id=tool["id"],
                                    name=tool["name"],
                                    arguments=arguments,
                                    provider_tool_call_id=tool["id"],
                                )
                    elif event_type == "message_delta":
                        usage = _get(raw, "usage")
                        if usage is not None:
                            emitted = True
                            yield UsageUpdated(self._usage(usage))
                final = await stream.get_final_message()
                emitted = True
                yield ResponseCompleted(self._response(final))
        except ModelRuntimeError:
            raise
        except Exception as error:
            if emitted:
                raise StreamInterruptedError(
                    str(error),
                    provider=self.provider,
                    model=request.target.model,
                    provider_request_id=_get(error, "request_id"),
                ) from error
            raise self._normalize_error(error, request) from error

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            raw = await self.client.messages.create(**self._build_request(request))
        except Exception as error:
            raise self._normalize_error(error, request) from error
        return self._response(raw)
