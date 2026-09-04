from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .events import ModelEvent
from .types import ModelRequest, ModelResponse, ModelTarget

if TYPE_CHECKING:
    from .capabilities import ModelCapabilities


@runtime_checkable
class ModelRuntime(Protocol):
    @property
    def provider(self) -> str: ...

    def capabilities_for(self, target: ModelTarget) -> ModelCapabilities: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
