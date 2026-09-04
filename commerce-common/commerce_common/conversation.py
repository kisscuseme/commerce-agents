from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

from commerce_model_runtime import (
    ModelContent,
    ModelMessage,
    ProviderOpaqueContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
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


def _decode_block(block: Any) -> ModelContent | None:
    if isinstance(block, str):
        return TextContent(block)
    if not isinstance(block, dict):
        return None
    kind = block.get("type")
    if kind == "text":
        return TextContent(str(block.get("text", "")))
    if kind == "tool_use":
        tool_id = str(block.get("id", ""))
        raw_input = block.get("input")
        return ToolCallContent(
            id=tool_id,
            name=str(block.get("name", "")),
            arguments=dict(raw_input) if isinstance(raw_input, dict) else {},
            provider_tool_call_id=tool_id,
        )
    if kind == "tool_result":
        return ToolResultContent(
            tool_call_id=str(block.get("tool_use_id", "")),
            content=str(block.get("content", "")),
            is_error=bool(block.get("is_error", False)),
        )
    return ProviderOpaqueContent(provider="anthropic", data=dict(block))


def _decode_message(message: dict[str, Any]) -> ModelMessage:
    role = "assistant" if message.get("role") == "assistant" else "user"
    raw_content = message.get("content")
    blocks = [raw_content] if isinstance(raw_content, str) else list(raw_content or [])
    content = [decoded for block in blocks if (decoded := _decode_block(block)) is not None]
    return ModelMessage(role=role, content=content)


def _encode_block(block: ModelContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolCallContent):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.arguments,
        }
    if isinstance(block, ToolResultContent):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_call_id,
            "content": block.content,
            "is_error": block.is_error,
        }
    if isinstance(block, ProviderOpaqueContent):
        if block.provider != "anthropic":
            raise ValueError(
                f"legacy conversation cannot encode opaque content for provider {block.provider!r}"
            )
        return dict(block.data)
    raise TypeError(f"unsupported model content: {type(block).__name__}")


def _encode_message(message: ModelMessage) -> dict[str, Any]:
    return {"role": message.role, "content": [_encode_block(block) for block in message.content]}


class LegacyConversationBridge:
    """Keep the v1 host conversation mutable while exposing canonical model messages."""

    def __init__(
        self,
        raw_messages: list[dict[str, Any]],
        host_texts: Collection[str] = (),
    ) -> None:
        self.raw_messages = raw_messages
        self.host_texts = frozenset(host_texts)

    def model_messages(self) -> list[ModelMessage]:
        decoded = [_decode_message(message) for message in self.raw_messages]
        merged: list[ModelMessage] = []
        for message in decoded:
            if merged and message.role == "user" and merged[-1].role == "user":
                previous = merged[-1]
                merged[-1] = ModelMessage(role="user", content=[*previous.content, *message.content])
            else:
                merged.append(message)
        return merged

    def append_assistant(self, message: ModelMessage) -> None:
        if message.role != "assistant":
            raise ValueError("append_assistant requires an assistant message")
        if message.content:
            self.raw_messages.append(_encode_message(message))

    def append_tool_results(self, results: Iterable[ToolResultContent]) -> None:
        content = list(results)
        if content:
            self.raw_messages.append(_encode_message(ModelMessage(role="user", content=content)))

    def append_host_text(self, text: str) -> None:
        self.raw_messages.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

    def latest_user_text(self) -> str:
        for message in reversed(self.raw_messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            texts = _text_blocks(content)
            if texts and all(text in self.host_texts for text in texts):
                continue
            if texts:
                return "\n".join(texts)
            if isinstance(content, list) and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            ):
                continue
            return ""
        return ""

    def latest_exchange(self) -> list[dict[str, Any]]:
        for index in range(len(self.raw_messages) - 1, -1, -1):
            if _is_user_text(self.raw_messages[index], self.host_texts):
                return self.raw_messages[index:]
        return self.raw_messages

    def transcript_text(self, messages: list[dict[str, Any]] | None = None) -> str:
        source = self.raw_messages if messages is None else messages
        lines: list[str] = []
        for message in source:
            role = message.get("role", "")
            for text in _text_blocks(message.get("content")):
                if text and text not in self.host_texts:
                    lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def close_open_tool_calls(self, settled: Mapping[str, Any] | None = None) -> int:
        if not self.raw_messages or self.raw_messages[-1].get("role") != "assistant":
            return 0
        content = self.raw_messages[-1].get("content")
        if not isinstance(content, list):
            return 0
        ids = [
            str(block.get("id", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
        ]
        if not ids:
            return 0
        settled = settled or {}
        results: list[ToolResultContent] = []
        for tool_id in ids:
            if tool_id in settled:
                outcome = settled[tool_id]
                results.append(
                    ToolResultContent(
                        tool_call_id=tool_id,
                        content=str(outcome.result_text),
                        is_error=bool(outcome.is_error),
                    )
                )
            else:
                results.append(
                    ToolResultContent(
                        tool_call_id=tool_id,
                        content=INTERRUPTED_RESULT_TEXT,
                        is_error=True,
                    )
                )
        self.append_tool_results(results)
        return len(results)


def latest_user_text(messages: list[dict[str, Any]], host_texts: Collection[str] = ()) -> str:
    return LegacyConversationBridge(messages, host_texts).latest_user_text()


def latest_exchange(
    messages: list[dict[str, Any]], host_texts: Collection[str] = ()
) -> list[dict[str, Any]]:
    return LegacyConversationBridge(messages, host_texts).latest_exchange()


def transcript_text(messages: list[dict[str, Any]], host_texts: Collection[str] = ()) -> str:
    return LegacyConversationBridge(messages, host_texts).transcript_text()
