# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The config fields both roles share, in the section order the role configs continue."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from commerce_model_runtime import ModelTarget, ReasoningConfig, ReasoningEffort

from .fencing import MAX_FENCED_CHARS

DEFAULT_MEMORY_MODEL = "claude-haiku-4-5-20251001"

ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]


class BaseAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brand_name: str = "the store"
    assistant_name: str = "the assistant"
    brand_voice: str = "plain and specific"

    # Provider defaults preserve the existing Anthropic behavior. A memory provider left
    # unset inherits the main provider. If a non-Anthropic deployment also leaves the
    # legacy Claude memory model untouched, memory follows the main model instead of
    # constructing an invalid cross-provider target.
    provider: str = "anthropic"
    model: str
    memory_provider: str | None = None
    memory_model: str = DEFAULT_MEMORY_MODEL
    thinking_effort: ThinkingEffort | None = None

    max_tokens: int = 2048
    max_tool_iterations: int = 8
    request_timeout_s: float = 120.0

    eager_tool_dispatch: bool = True
    rolling_conversation_cache: bool = True
    eager_partial_frames: bool = False
    close_on_presentation: bool = True

    enable_web_search: bool = False
    enable_memory: bool = True

    memory_tier_one_cap: int = Field(default=8, ge=0)
    memory_blocked_patterns: tuple[str, ...] = ()
    memory_retention_days: int | None = Field(default=None, ge=1)

    max_context_chars: int = Field(default=2000, ge=0)
    max_search_results: int = Field(default=8, ge=1, le=25)
    max_fenced_chars: int = MAX_FENCED_CHARS
    compact_history_above_tokens: int = Field(default=100_000, ge=0)

    def absent_tools(self) -> frozenset[str]:
        return frozenset()

    def model_target(self) -> ModelTarget:
        return ModelTarget(self.provider, self.model)

    def memory_target(self) -> ModelTarget:
        provider = self.memory_provider or self.provider
        model = self.memory_model
        if (
            self.memory_provider is None
            and self.provider != "anthropic"
            and self.memory_model == DEFAULT_MEMORY_MODEL
        ):
            model = self.model
        return ModelTarget(provider, model)

    def reasoning_config(self) -> ReasoningConfig:
        if self.thinking_effort is None:
            return ReasoningConfig(ReasoningEffort.OFF)
        return ReasoningConfig(ReasoningEffort(self.thinking_effort))

    # Compatibility path for Claude Agent SDK and not-yet-migrated Messages runtimes.
    # Remove only when no legacy caller expects Anthropic wire fields from config.
    def thinking_request_fields(self) -> dict[str, Any]:
        if self.thinking_effort is None:
            return {"thinking": {"type": "disabled"}}
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.thinking_effort},
        }
