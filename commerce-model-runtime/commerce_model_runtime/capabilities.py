from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class ModelOperation(str, Enum):
    MAIN_TURN = "main_turn"
    MEMORY_EXTRACTION = "memory_extraction"
    PORTABLE_ANALYSIS = "portable_analysis"
    HOSTED_ANALYSIS = "hosted_analysis"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    DEGRADED = "degraded"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelCapabilities:
    stream_text: bool = False
    function_tools: bool = False
    tool_result_continuation: bool = False
    tool_choice_auto: bool = False
    tool_choice_none: bool = False
    tool_choice_specific: bool = False
    multiple_tool_calls: bool = False
    stream_tool_arguments: bool = False
    reasoning_effort: bool = False
    prompt_cache: bool = False
    builtin_web_search: bool = False
    hosted_code_execution: bool = False

    @classmethod
    def full(cls) -> ModelCapabilities:
        return cls(**{name: True for name in cls.__dataclass_fields__})

    def replace(self, **changes: bool) -> ModelCapabilities:
        return replace(self, **changes)


@dataclass(frozen=True)
class CapabilityPlan:
    status: CapabilityStatus
    degraded: frozenset[str] = frozenset()


class CapabilityValidationError(ValueError):
    def __init__(self, operation: ModelOperation, missing: set[str]) -> None:
        names = ", ".join(sorted(missing))
        super().__init__(f"{operation.value} requires unsupported capabilities: {names}")
        self.operation = operation
        self.missing = frozenset(missing)


_MAIN_REQUIRED = {
    "stream_text",
    "function_tools",
    "tool_result_continuation",
    "tool_choice_auto",
    "tool_choice_none",
    "tool_choice_specific",
    "multiple_tool_calls",
}
_MEMORY_REQUIRED = {"function_tools", "tool_choice_auto"}
_ANALYSIS_REQUIRED = {"function_tools", "tool_result_continuation", "multiple_tool_calls"}
_OPTIONAL_PERFORMANCE = {"prompt_cache", "stream_tool_arguments"}


def validate_capabilities(
    operation: ModelOperation,
    capabilities: ModelCapabilities,
    *,
    enable_web_search: bool = False,
    require_hosted_code_execution: bool = False,
) -> CapabilityPlan:
    if operation is ModelOperation.MAIN_TURN:
        required = set(_MAIN_REQUIRED)
        optional = set(_OPTIONAL_PERFORMANCE)
        if enable_web_search:
            required.add("builtin_web_search")
    elif operation is ModelOperation.MEMORY_EXTRACTION:
        required = set(_MEMORY_REQUIRED)
        optional = set()
    elif operation is ModelOperation.PORTABLE_ANALYSIS:
        required = set(_ANALYSIS_REQUIRED)
        optional = set()
    elif operation is ModelOperation.HOSTED_ANALYSIS:
        required = set(_ANALYSIS_REQUIRED)
        if require_hosted_code_execution:
            required.add("hosted_code_execution")
        optional = set()
    else:
        raise ValueError(f"unknown model operation: {operation!r}")

    missing = {name for name in required if not getattr(capabilities, name)}
    if missing:
        raise CapabilityValidationError(operation, missing)

    degraded = frozenset(name for name in optional if not getattr(capabilities, name))
    return CapabilityPlan(
        CapabilityStatus.DEGRADED if degraded else CapabilityStatus.SUPPORTED,
        degraded,
    )
