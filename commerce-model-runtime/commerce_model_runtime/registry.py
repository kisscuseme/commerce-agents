from __future__ import annotations

from collections.abc import Iterable

from .runtime import ModelRuntime
from .types import ModelTarget


class RuntimeNotRegisteredError(LookupError):
    pass


class RuntimeRegistry:
    def __init__(self, runtimes: Iterable[ModelRuntime] = ()) -> None:
        self._runtimes: dict[str, ModelRuntime] = {}
        for runtime in runtimes:
            self.register(runtime)

    def register(self, runtime: ModelRuntime) -> None:
        key = runtime.provider.strip().lower()
        if not key:
            raise ValueError("runtime provider must be non-empty")
        if key in self._runtimes:
            raise ValueError(f"duplicate provider runtime: {key}")
        self._runtimes[key] = runtime

    def resolve(self, target: ModelTarget) -> ModelRuntime:
        key = target.provider.strip().lower()
        try:
            return self._runtimes[key]
        except KeyError as exc:
            raise RuntimeNotRegisteredError(f"no runtime registered for provider: {key}") from exc

    @property
    def providers(self) -> frozenset[str]:
        return frozenset(self._runtimes)
