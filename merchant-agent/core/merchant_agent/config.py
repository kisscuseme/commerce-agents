# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Per-deployment settings for the merchant agent; per-request values travel in
``MerchantSessionContext``. Sections continue ``BaseAgentConfig``'s: capabilities
(analysis), guardrails, approval, grounding gates."""

from __future__ import annotations

from pydantic import Field

from commerce_model_runtime import ModelTarget
from commerce_common.config import BaseAgentConfig, ThinkingEffort


class MerchantAgentConfig(BaseAgentConfig):
    assistant_name: str = "the merchant assistant"
    brand_voice: str = "plain and specific, numbers first"
    model: str = "claude-opus-5"
    thinking_effort: ThinkingEffort | None = "low"

    # -- The analysis delegate. `analysis_model` None means the main model and
    # `analysis_provider` None means the main provider. Hosted code execution remains an
    # explicit deployment capability checked by the Messages runtime.
    enable_analysis: bool = False
    analysis_provider: str | None = None
    analysis_model: str | None = None
    analysis_use_code_execution: bool = True
    analysis_sql_only: bool = True
    max_analysis_iterations: int = Field(default=8, ge=1)
    analysis_max_tokens: int = Field(default=4096, ge=256)
    analysis_timeout_s: float = Field(default=120.0, gt=0)
    max_delegate_calls_per_turn: int = Field(default=2, ge=1)
    max_analysis_rows: int = Field(default=200, ge=1)
    max_analysis_table_chars: int = Field(default=8_000, ge=500)
    analysis_query_timeout_s: float = Field(default=10.0, gt=0)

    enable_listing_edits: bool = True
    enable_inventory: bool = True
    enable_pricing: bool = True
    enable_campaigns: bool = True

    max_items_per_change: int = Field(default=25, ge=1)
    max_price_delta_pct: float = Field(default=20.0, gt=0)
    max_promotion_discount_pct: float = Field(default=50.0, gt=0, le=90)
    max_restock_quantity: int = Field(default=500, ge=1)
    max_campaign_budget: float = Field(default=10_000.0, gt=0)
    max_listing_field_chars: int = Field(default=2000, ge=200)
    protected_fields: tuple[str, ...] = (
        "listing_id",
        "currency",
        "tax_category",
        "compliance_notes",
    )
    price_bearing_fields: tuple[str, ...] = ("price",)
    listing_update_blocked_fields: tuple[str, ...] = ("price", "stock")

    require_host_approval: bool = True
    approval_surface: str = "the host application's approval control"
    stage_shows_preview: bool = True

    metrics_grounding_gate: bool = True
    metrics_intent_terms: tuple[str, ...] = (
        "sales",
        "revenue",
        "orders",
        "traffic",
        "conversion",
        "aov",
        "average order value",
        "performance",
        "performing",
        "trend",
        "trending",
        "growth",
        "drop",
        "dropped",
        "spike",
        "returns rate",
        "return rate",
        "margin",
        "margins",
        "profit",
        "best seller",
        "best sellers",
        "slow mover",
        "slow movers",
        "sell-through",
        "campaign performance",
        "ad spend",
        "return on ad spend",
        "roas",
    )
    metrics_intent_cues: tuple[str, ...] = (
        "?",
        "how",
        "what",
        "why",
        "show me",
        "compare",
        "summarize",
        "summary",
        "report",
        "this week",
        "last week",
        "this month",
        "last month",
        "yesterday",
        "today",
        "vs",
        "versus",
    )
    staging_followthrough_gate: bool = True
    change_intent_terms: tuple[str, ...] = (
        "price",
        "prices",
        "pricing",
        "reprice",
        "markdown",
        "mark down",
        "discount",
        "promotion",
        "promo",
        "rate",
        "nightly rate",
        "restock",
        "listing",
        "listings",
        "title",
        "titles",
        "description",
        "descriptions",
        "category",
        "copy",
    )
    change_intent_cues: tuple[str, ...] = (
        "take",
        "drop",
        "cut",
        "set",
        "lower",
        "raise",
        "reduce",
        "increase",
        "bump",
        "change",
        "update",
        "move",
        "stage",
        "bring down",
        "knock",
        "append",
        "rewrite",
        "revise",
        "add",
        "fix",
    )
    queue_grounding_gate: bool = True
    apply_intent_phrases: tuple[str, ...] = (
        "apply",
        "put it through",
        "put through",
        "push it through",
        "push through",
    )

    def analysis_target(self) -> ModelTarget:
        return ModelTarget(
            self.analysis_provider or self.provider,
            self.analysis_model or self.model,
        )

    @property
    def stages_changes(self) -> bool:
        """Whether any stage_* tool remains for this deployment."""
        return (
            self.enable_listing_edits
            or self.enable_inventory
            or self.enable_pricing
            or self.enable_campaigns
        )

    def absent_tools(self) -> frozenset[str]:
        """Names `build_tools` leaves out for the systems switched off above."""
        names: set[str] = set()
        if not self.enable_listing_edits:
            names.add("stage_listing_update")
        if not self.enable_inventory:
            names |= {"get_inventory_alerts", "get_order_issues", "stage_inventory_action"}
        if not self.enable_pricing:
            names |= {"get_pricing_context", "stage_price_update", "stage_promotion"}
        if not self.enable_campaigns:
            names |= {"get_campaign_performance", "stage_campaign"}
        if not self.stages_changes:
            names |= {
                "get_pending_changes",
                "apply_change",
                "discard_change",
                "present_change_preview",
            }
        return frozenset(names)
