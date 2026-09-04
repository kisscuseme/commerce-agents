# Provider-Neutral Runtime Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce the provider-neutral model runtime contract and migrate the existing Anthropic Messages API path onto it without changing observable Shopping/Merchant behavior.

**Architecture:** Add `commerce-model-runtime` for canonical model types/events/runtime interfaces and an Anthropic adapter. Add a legacy conversation bridge and provider-neutral model-round runner to `commerce-common`, then migrate Shopping, Merchant, memory extraction, and merchant analysis while preserving current host APIs and safety semantics.

**Tech Stack:** Python 3.11+, Pydantic 2.x, Anthropic Python SDK, pytest, asyncio.

**Spec:** `docs/superpowers/specs/2026-09-05-provider-neutral-commerce-runtime-design.md`

## Global Constraints

- Keep `messages: list[dict[str, Any]]` and the current `AgentEvent` protocol compatible in v1.
- Do not change Claude Agent SDK or Managed Agents behavior.
- Do not add OpenAI/Gemini implementation in this plan.
- Anthropic is the behavioral golden reference.
- `commerce-model-runtime` must not import `commerce-common`, Shopping, or Merchant packages.
- Provider SDK objects/raw stream names must stop at provider adapters.
- Never auto-retry Commerce writes.
- Required safety/control capabilities fail validation; cache/partial-argument streaming may degrade.

---

### Task 1: Canonical runtime package and data model

**Files:**
- Create: `commerce-model-runtime/pyproject.toml`
- Create: `commerce-model-runtime/commerce_model_runtime/{__init__,types,events,errors}.py`
- Test: `commerce-model-runtime/tests/test_types.py`

**Interfaces:** Produces `ModelTarget`, `ModelRequest`, `ModelResponse`, `ModelMessage`, content/tool types, `ToolChoice`, `ReasoningConfig`, `CachePolicy`, `ProviderState`, `ModelUsage`, and canonical stream event classes.

- [ ] **Step 1: Write failing canonical type tests**

```python
from dataclasses import asdict
from commerce_model_runtime import ProviderState, ToolChoice, ToolChoiceMode


def test_provider_state_is_persistable_shape():
    state = ProviderState(provider="anthropic", data={"response_id": "r_123"})
    assert asdict(state)["data"] == {"response_id": "r_123"}


def test_specific_tool_choice_requires_name():
    choice = ToolChoice.specific("search_products")
    assert choice.mode is ToolChoiceMode.SPECIFIC
    assert choice.name == "search_products"
```

- [ ] **Step 2: Run `pytest commerce-model-runtime/tests/test_types.py -v` and verify import failures.**
- [ ] **Step 3: Implement dataclasses/enums/events.** `ModelUsage` unavailable counters are `None`, not zero; `ProviderState.data` is JSON-serializable `dict[str, Any]`.
- [ ] **Step 4: Re-run the test and verify PASS.**
- [ ] **Step 5: Commit:** `git commit -m "feat: add provider-neutral model runtime types"`.

---

### Task 2: Capability validation, `ModelRuntime`, and runtime registry

**Files:**
- Create: `commerce-model-runtime/commerce_model_runtime/capabilities.py`
- Create: `commerce-model-runtime/commerce_model_runtime/runtime.py`
- Create: `commerce-model-runtime/commerce_model_runtime/registry.py`
- Test: `commerce-model-runtime/tests/test_{capabilities,registry}.py`

**Interfaces:** Produces `ModelCapabilities`, `ModelOperation`, `CapabilityPlan`, `CapabilityValidationError`, `ModelRuntime`, and `RuntimeRegistry`.

- [ ] **Step 1: Write failing operation-aware validation tests.**

```python
from dataclasses import replace


def test_main_turn_rejects_missing_specific_tool_choice():
    caps = replace(ModelCapabilities.full(), tool_choice_specific=False)
    with pytest.raises(CapabilityValidationError):
        validate_capabilities(ModelOperation.MAIN_TURN, caps, enable_web_search=False)
```

Also assert missing `prompt_cache` produces DEGRADED rather than INVALID.
- [ ] **Step 2: Run the two test files and verify failure.**
- [ ] **Step 3: Implement rules from the spec:** main turn requires streaming/function tools/continuation/auto-none-specific/multiple calls; memory and portable analysis use non-streaming requirements; web search is required only when enabled; prompt cache and streamed args are optional.
- [ ] **Step 4: Define the protocol after `ModelCapabilities` exists:**

```python
class ModelRuntime(Protocol):
    @property
    def provider(self) -> str: ...

    def capabilities_for(self, target: ModelTarget) -> ModelCapabilities: ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
```

Then implement registry resolution errors for unregistered providers and run tests.
- [ ] **Step 5: Commit:** `git commit -m "feat: add runtime capability validation"`.

---

### Task 3: Provider-neutral `FakeModelRuntime`

**Files:**
- Create: `commerce-model-runtime/commerce_model_runtime/testing.py`
- Test: `commerce-model-runtime/tests/test_testing_runtime.py`

**Interfaces:** Produces `FakeModelRuntime`, `text_round()`, `tool_round()`, `multi_tool_round()`, and recorded canonical calls.

- [ ] **Step 1: Write a failing replay test.**

```python
async def test_fake_runtime_records_request_and_replays_events():
    runtime = FakeModelRuntime([text_round("hello")])
    events = [event async for event in runtime.stream(minimal_request())]
    assert len(runtime.calls) == 1
    assert [type(e).__name__ for e in events] == ["TextDelta", "ResponseCompleted"]
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement the fake entirely with canonical types; no `.messages.stream` or Anthropic blocks.**
- [ ] **Step 4: Verify PASS.**
- [ ] **Step 5: Commit:** `git commit -m "test: add fake provider-neutral runtime"`.

---

### Task 4: Anthropic adapter

**Files:**
- Create: `commerce-model-runtime/commerce_model_runtime/providers/{__init__,anthropic}.py`
- Test: `commerce-model-runtime/tests/providers/test_anthropic_{request,stream,complete}.py`

**Interfaces:** Produces `AnthropicRuntime(client: AsyncAnthropic | None = None, *, timeout_s: float = 120.0)`.

- [ ] **Step 1: Write failing request mapping tests** for forced/auto/none tool choice, static cache segment, progressive tool input, reasoning effort, and web-search built-in.

```python
request = canonical_request(tool_choice=ToolChoice.specific("search_products"))
body = runtime._build_request(request)
assert body["tool_choice"] == {"type": "tool", "name": "search_products"}
```

- [ ] **Step 2: Write failing stream tests** using synthetic `message_start`, `content_block_*`, `input_json_delta`, and `message_delta`; assert only canonical events escape.
- [ ] **Step 3: Write failing `complete()`/usage/error-normalization tests.**
- [ ] **Step 4: Implement all Anthropic-specific request/stream/cache/thinking/usage/exception behavior in this adapter.**
- [ ] **Step 5: Run `pytest commerce-model-runtime/tests/providers -v` and verify PASS.**
- [ ] **Step 6: Commit:** `git commit -m "feat: add Anthropic runtime adapter"`.

---

### Task 5: `LegacyConversationBridge`

**Files:**
- Create: `commerce-common/commerce_common/conversation.py`
- Test: `commerce-common/tests/test_conversation.py`

**Interfaces:** Produces canonical decoding plus incremental mutation helpers: `latest_user_text`, `latest_exchange`, `transcript_text`, `append_assistant`, `append_tool_results`, `append_host_text`, `close_open_tool_calls`.

- [ ] **Step 1: Write failing incremental-mutation and round-trip tests.**

```python
def test_append_assistant_updates_original_list_immediately():
    raw = [{"role": "user", "content": "hello"}]
    bridge = LegacyConversationBridge(raw)
    bridge.append_assistant(ModelMessage(role="assistant", content=[TextContent("hi")]))
    assert raw[-1]["role"] == "assistant"
```

Include the existing merged tool-result + subsequent user-text legacy shape.
- [ ] **Step 2: Run tests and verify failure.**
- [ ] **Step 3: Implement by preserving semantics of current turn helpers; do not change stored wire shape.**
- [ ] **Step 4: Verify PASS.**
- [ ] **Step 5: Commit:** `git commit -m "feat: add legacy conversation bridge"`.

---

### Task 6: `ModelRoundRunner` and removal of provider parsing from `turn.py`

**Files:**
- Create: `commerce-common/commerce_common/model_round.py`
- Modify: `commerce-common/commerce_common/turn.py`
- Test: `commerce-common/tests/test_model_round.py`

**Interfaces:** Consumes canonical model events, `EagerDispatcher`, presentation components, and `ToolOutcome`; produces `ModelRoundResult` and host events.

- [ ] **Step 1: Write failing tests** for text streaming, eager execution, partial UI, malformed tool calls, and generator cancellation using only `FakeModelRuntime`.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement:** `TextDelta` → `AgentEvent.text_delta`; partial args → presentation parser; dispatch only on `ToolCallCompleted`; `ToolCallFailed` → error outcome without execution; cancel started tasks on close/error.
- [ ] **Step 4: Remove `StreamedTool`, `StreamedRound`, raw event names, SDK `.model_dump()` assumptions, and provider usage parsing from `turn.py`; retain dispatcher/compaction/session/outcome/timing/log helpers.**
- [ ] **Step 5: Run `pytest commerce-common/tests/test_model_round.py commerce-common/tests/test_fencing.py -v`.**
- [ ] **Step 6: Commit:** `git commit -m "refactor: add provider-neutral model round runner"`.

---

### Task 7: Provider-neutral prompt assembly

**Files:**
- Modify: `commerce-common/commerce_common/prompt_assembly.py`
- Test: `commerce-common/tests/test_prompt_assembly.py`

- [ ] Write failing tests for STATIC/DYNAMIC system intent and `FunctionToolSpec.progressive`, not Anthropic wire keys.
- [ ] Run tests and verify failure.
- [ ] Remove `cache_control` and `eager_input_streaming` construction from common code while preserving prompt bytes/semantics.
- [ ] Run tests and commit with `git commit -m "refactor: express prompt caching as semantic intent"`.

---

### Task 8: Provider-aware config with backward-compatible model strings

**Files:**
- Modify: `commerce-common/commerce_common/config.py`
- Modify: `shopping-agent/core/shopping_agent/config.py`
- Modify: `merchant-agent/core/merchant_agent/config.py`
- Test: corresponding config/core tests.

**Interfaces:** Adds `provider="anthropic"`, `memory_provider=None`, Merchant `analysis_provider=None`, and `model_target()/memory_target()/analysis_target()`.

- [ ] Write failing tests proving old config defaults to Anthropic and memory/analysis providers inherit the main provider when `None`.
- [ ] Run role/core tests and verify failure.
- [ ] Implement target helpers and semantic reasoning config while preserving public `thinking_effort` compatibility.
- [ ] Run `pytest commerce-common/tests shopping-agent/core/tests merchant-agent/core/tests -q`.
- [ ] Commit with `git commit -m "feat: add provider-aware model targets"`.

---

### Task 9: Migrate `ShoppingAgent`

**Files:**
- Modify: `shopping-agent/runtime-messages-api/shopping_agent_runtime/orchestrator.py`
- Modify: `shopping-agent/runtime-messages-api/pyproject.toml`
- Test: Shopping orchestrator tests and `tests/test_turn_loop.py` shopping cases.

**Interfaces:** Preferred injection is `runtime=` or `runtimes=`; existing `client=AsyncAnthropic(...)` is accepted and wrapped in `AnthropicRuntime`.

- [ ] Migrate tests from `FakeClient` to canonical scripted runtime and assert canonical tool choice/cache/progressive intent.
- [ ] Run Shopping tests and verify failure.
- [ ] Build canonical `ModelRequest`s, use bridge + round runner, and preserve grounding, max-iteration forced text, close-on-presentation, usage, compaction, and final event.
- [ ] Add `client=` compatibility wrapping; reject ambiguous conflicting injections.
- [ ] Run Shopping tests and verify PASS.
- [ ] Commit with `git commit -m "refactor: migrate shopping runtime to ModelRuntime"`.

---

### Task 10: Migrate `MerchantAgent`

**Files:**
- Modify: `merchant-agent/runtime-messages-api/merchant_agent_runtime/orchestrator.py`
- Modify: `merchant-agent/runtime-messages-api/pyproject.toml`
- Test: follow-through/progress tests and Merchant cases in `tests/test_turn_loop.py`.

- [ ] Migrate tests to canonical scripted runtime, including “exactly one staging reminder” behavior.
- [ ] Run Merchant tests and verify failure.
- [ ] Use shared round runner but retain `HOST_TEXTS`, `STAGING_FOLLOWTHROUGH_REMINDER`, progress queue draining, staging detection, gates, and approvals in Merchant orchestration.
- [ ] Run Merchant tests and verify PASS.
- [ ] Commit with `git commit -m "refactor: migrate merchant runtime to ModelRuntime"`.

---

### Task 11: Migrate memory extraction

**Files:**
- Modify: `commerce-common/commerce_common/memory.py`
- Modify: Shopping/Merchant executor memory builders.
- Test: memory tests and memory cases in `tests/test_consumption_paths.py`.

**Interfaces:** Adds `MemoryExtractor` and `LLMMemoryExtractor`; `MemoryRuntime.extract()` no longer accepts an Anthropic client.

- [ ] Write failing tests that extraction uses a canonical function tool, excludes tool results from transcript, and still passes `validate_fact`, write filters, and purge generation.
- [ ] Assert `commerce_common.memory` no longer imports `anthropic`.
- [ ] Run memory tests and verify failure.
- [ ] Implement runtime-backed extraction without changing store/filter behavior.
- [ ] Run tests and commit with `git commit -m "refactor: make memory extraction provider-neutral"`.

---

### Task 12: Migrate portable Merchant analysis

**Files:**
- Modify: `merchant-agent/runtime-messages-api/merchant_agent_runtime/analysis.py`
- Test: analysis tests.

- [ ] Write failing canonical-loop tests for read → result → submit, SQL-only mode, invalid submission correction, budgets, and no provenance widening.
- [ ] Run analysis tests and verify failure.
- [ ] Replace direct `messages.create()` with `ModelRuntime.complete()`. Keep hosted execution optional and outside portable semantics.
- [ ] Run tests and commit with `git commit -m "refactor: make merchant analysis provider-neutral"`.

---

### Task 13: Split Anthropic platform seams from provider seams

**Files:**
- Modify: `tests/test_platform_seams.py`
- Create: `tests/test_provider_seams.py`

- [ ] Rewrite existing direct/Vertex/Bedrock/Foundry/gateway tests to wrap clients in `AnthropicRuntime`.
- [ ] Add provider-registry injection tests for both role agents using `FakeModelRuntime`.
- [ ] Search tests for `.messages.stream`, raw Anthropic block types, `cache_control`, and `eager_input_streaming`; move wire assertions to Anthropic adapter tests.
- [ ] Run both seam files and commit with `git commit -m "test: split provider and Anthropic transport seams"`.

---

### Task 14: Dependency graph and install wiring

**Files:**
- Modify: `commerce-common/pyproject.toml`
- Modify: both Messages-runtime pyprojects
- Modify: `requirements.txt`

- [ ] Add a check that `commerce-common/pyproject.toml` contains `commerce-model-runtime` and no direct `anthropic>=` requirement.
- [ ] Add `-e ./commerce-model-runtime[anthropic]` before common packages in root requirements; follow existing version-pin conventions.
- [ ] Run `pip install -r requirements.txt` and import smoke tests.
- [ ] Commit with `git commit -m "build: wire provider-neutral runtime package"`.

---

### Task 15: Anthropic parity gate and documentation

**Files:**
- Modify only as failures require.
- Update: `docs/deployment.md`, both Messages-runtime READMEs.

- [ ] Run `pytest -q`.
- [ ] Run `python scripts/verify_all.py`.
- [ ] Run a leakage scan:

```bash
grep -R "content_block_delta\|input_json_delta\|cache_control\|eager_input_streaming" \
  commerce-common/commerce_common \
  shopping-agent/runtime-messages-api/shopping_agent_runtime \
  merchant-agent/runtime-messages-api/merchant_agent_runtime
```

Raw provider wire references must not remain in common/orchestration logic.
- [ ] Document existing default Anthropic behavior and preferred `RuntimeRegistry` injection; label direct `client=` as compatibility/deprecated.
- [ ] Re-run `pytest -q && python scripts/verify_all.py`.
- [ ] Commit with `git commit -m "docs: document provider-neutral Anthropic runtime"`.

## Completion Gate

- [ ] full deterministic suite passes
- [ ] `scripts/verify_all.py` passes
- [ ] role orchestrators contain no raw provider stream parsing
- [ ] memory has no Anthropic model-call dependency
- [ ] `commerce-common` has no mandatory Anthropic dependency
- [ ] old Anthropic client injection remains compatible
- [ ] safety, approval, staging, persisted history, interruption behavior remain unchanged
