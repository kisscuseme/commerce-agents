from __future__ import annotations

import copy
import itertools
import json
from collections.abc import Awaitable, Callable, Iterable
from types import SimpleNamespace
from typing import Any


class FakeBlock:
    """A content block that quacks like the Anthropic SDK block objects."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        for key, value in fields.items():
            setattr(self, key, value)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        return dict(self._fields)


def text_block(text: str) -> FakeBlock:
    return FakeBlock(type="text", text=text)


def tool_use_block(name: str, tool_input: dict[str, Any], block_id: str = "tu-1") -> FakeBlock:
    return FakeBlock(type="tool_use", id=block_id, name=name, input=tool_input)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=1,
        output_tokens=1,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


def text_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(stop_reason="end_turn", usage=_usage(), content=[text_block(text)])


def tool_use_message(name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return tool_calls_message((name, tool_input))


def tool_calls_message(*calls: tuple[Any, ...]) -> SimpleNamespace:
    content = [
        tool_use_block(call[0], call[1], call[2] if len(call) > 2 else f"tu-{index + 1}")
        for index, call in enumerate(calls)
    ]
    return SimpleNamespace(stop_reason="tool_use", usage=_usage(), content=content)


def create_response(*blocks: FakeBlock, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, usage=_usage())


Chunks = dict[int, list[str | BaseException]]


class FakeStream:
    def __init__(self, final: SimpleNamespace, chunks: Chunks | None = None) -> None:
        self._final = final
        self._chunks = chunks or {}

    async def __aenter__(self) -> "FakeStream":
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    def __aiter__(self):
        async def events():
            usage = getattr(self._final, "usage", None)
            if usage is not None:
                yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage=usage))
            for index, block in enumerate(self._final.content):
                yield SimpleNamespace(type="content_block_start", index=index, content_block=block)
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    delta = SimpleNamespace(type="text_delta", text=block.text)
                    yield SimpleNamespace(type="content_block_delta", index=index, delta=delta)
                elif block_type == "tool_use":
                    for piece in self._chunks.get(index, [json.dumps(block.input or {})]):
                        if isinstance(piece, BaseException):
                            raise piece
                        delta = SimpleNamespace(type="input_json_delta", partial_json=piece)
                        yield SimpleNamespace(type="content_block_delta", index=index, delta=delta)
                yield SimpleNamespace(type="content_block_stop", index=index)
            if usage is not None:
                yield SimpleNamespace(type="message_delta", usage=usage)

        return events()

    async def get_final_message(self) -> SimpleNamespace:
        return self._final


class FakeClient:
    def __init__(self, responses: Iterable[SimpleNamespace], chunks: Chunks | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = list(responses)
        self._chunks = chunks
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("more model calls than scripted responses")
        return FakeStream(self._responses.pop(0), self._chunks if len(self.calls) == 1 else None)


class FakeCreateClient:
    def __init__(
        self,
        responses: Iterable[SimpleNamespace],
        *,
        before_call: Callable[[int], Awaitable[object]] | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses = iter(responses)
        self._before_call = before_call
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> SimpleNamespace:
        index = len(self.calls)
        self.calls.append(copy.deepcopy(kwargs))
        if self._before_call is not None:
            await self._before_call(index)
        try:
            return next(self._responses)
        except StopIteration:
            raise AssertionError("more model calls than scripted responses") from None


def extraction_client(
    proposals: Iterable[dict[str, Any]],
    *,
    before_call: Callable[[int], Awaitable[object]] | None = None,
) -> FakeCreateClient:
    response = create_response(
        *(
            tool_use_block("record_fact", dict(proposal), f"tu-{i + 1}")
            for i, proposal in enumerate(proposals)
        )
    )
    return FakeCreateClient(itertools.repeat(response), before_call=before_call)
