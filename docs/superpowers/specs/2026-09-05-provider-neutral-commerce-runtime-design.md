# Provider-Neutral Commerce Runtime Design

**Status:** Approved
**Date:** 2026-09-05
**Reference baseline:** `anthropics/commerce-agents@fd4d59224ab96b43c6dc6888207c67b3bd5a24cf`
**Target:** `kisscuseme/commerce-agents`

## 1. Goal

Make the existing Messages API runtimes provider-neutral while preserving the current Shopping/Merchant safety and business architecture. Version 1 supports Anthropic, OpenAI, and Gemini through a shared Python runtime contract. Claude Agent SDK and Managed Agents stay Claude-specific and are out of scope.

The implementation must preserve existing grounding, fencing, provenance, cart/write caps, merchant staging, guardrails, host approval, presentation enrichment, memory validation, history compaction, eager tool dispatch, interruption recovery, and the host-facing `AgentEvent` protocol.

## 2. Non-goals

- Do not merge `ShoppingAgent` and `MerchantAgent` into one generic agent.
- Do not redesign `StorefrontBackend` or `MerchantBackend`.
- Do not change write authority, approval semantics, presentation payloads, or skill content.
- Do not generalize Claude Agent SDK or Managed Agents.
- Do not implement automatic cross-provider fallback inside a turn.
- Do not require hosted code execution for portable analysis.

## 3. Package boundary

Add a new package:

```text
commerce-model-runtime/
  pyproject.toml
  commerce_model_runtime/
    __init__.py
    types.py
    events.py
    runtime.py
    registry.py
    capabilities.py
    errors.py
    testing.py
    providers/
      __init__.py
      anthropic.py
      openai.py
      gemini.py
```

Dependency direction is strictly:

```text
commerce-model-runtime
        ↓
commerce-common
        ↓
shopping-agent/core + merchant-agent/core
        ↓
runtime-messages-api
```

`commerce-model-runtime` must not import `commerce-common`, Shopping, or Merchant packages. Provider SDK objects and wire-format event names must stop at provider adapters.

## 4. Canonical model contract

The new package defines provider-neutral types:

```python
@dataclass(frozen=True)
class ModelTarget:
    provider: str
    model: str

@dataclass
class ModelRequest:
    target: ModelTarget
    system: list[SystemSegment]
    messages: list[ModelMessage]
    tools: list[ToolSpec]
    tool_choice: ToolChoice
    max_tokens: int
    reasoning: ReasoningConfig | None = None
    cache: CachePolicy | None = None
    provider_state: ProviderState | None = None
    metadata: ModelRequestMetadata | None = None

@dataclass
class ModelMessage:
    role: Literal["user", "assistant"]
    content: list[ModelContent]

@dataclass
class TextContent:
    text: str

@dataclass
class ToolCallContent:
    id: str
    name: str
    arguments: dict[str, Any]
    provider_tool_call_id: str | None = None

@dataclass
class ToolResultContent:
    tool_call_id: str
    content: str
    is_error: bool = False
```

Function tools remain JSON-Schema based:

```python
@dataclass
class FunctionToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    progressive: bool = False
```

Tool choice is limited to the modes the existing loop needs:

```python
class ToolChoiceMode(Enum):
    AUTO = "auto"
    NONE = "none"
    SPECIFIC = "specific"
```

Reasoning is semantic rather than provider-native:

```python
class ReasoningEffort(Enum):
    OFF = "off"
    DEFAULT = "default"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
```

Provider continuation data is opaque to Commerce code:

```python
@dataclass
class ProviderState:
    provider: str
    data: dict[str, Any]
```

`ProviderState.data` must be JSON-serializable.

## 5. Canonical streaming events

Provider adapters normalize raw streams into:

```text
TextDelta
ToolCallStarted
ToolArgumentsDelta
ToolCallCompleted
ToolCallFailed
UsageUpdated
ResponseCompleted
```

The adapter owns raw event parsing and determines when tool arguments are complete JSON. Shopping and Merchant never inspect provider stream event names.

`ModelRuntime` is the sole model communication interface:

```python
class ModelRuntime(Protocol):
    @property
    def provider(self) -> str: ...

    def capabilities_for(self, target: ModelTarget) -> ModelCapabilities: ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
```

## 6. Runtime registry and provider selection

`RuntimeRegistry` maps provider names to runtimes and resolves `ModelTarget`s. Main conversation, memory extraction, and merchant analysis may select different providers/models.

The public v1 agent config remains backwards-compatible with model strings and adds provider fields:

```text
provider = "anthropic"
model = "..."
memory_provider = None
memory_model = "..."
analysis_provider = None
analysis_model = "..."
```

`None` for memory/analysis provider inherits the main provider. Internally these become `ModelTarget` objects.

Existing `client=AsyncAnthropic(...)` construction remains accepted in v1 and is wrapped by `AnthropicRuntime`; it is deprecated in favor of runtime/registry injection.

## 7. Capabilities

`ModelCapabilities` includes:

```text
stream_text
function_tools
tool_result_continuation
tool_choice_auto
tool_choice_none
tool_choice_specific
multiple_tool_calls
stream_tool_arguments
reasoning_effort
prompt_cache
builtin_web_search
hosted_code_execution
```

Requirements are computed per operation:

- Main Shopping/Merchant turns require streaming, function tools, continuation, auto/none/specific tool choice, and multiple tool calls.
- Memory extraction requires non-streaming completion and function-tool semantics used by the extractor.
- Portable Merchant analysis requires non-streaming completion, function tools, continuation, and multiple tool calls.
- `enable_web_search=True` requires the selected main runtime/model to expose built-in web search in v1.
- Hosted analysis requires `hosted_code_execution` only when explicitly selected.
- Prompt caching and streamed tool arguments are optional performance capabilities.

Safe degradation is allowed only for optional performance features:

- no streamed tool arguments → final UI only;
- no prompt cache → uncached execution.

Safety/control capabilities never silently degrade. Missing forced-tool support, required web search, write validation, provenance, guardrails, or host approval is a startup/configuration error.

## 8. Conversation compatibility

The public host API stays `messages: list[dict[str, Any]]` in v1. Add `LegacyConversationBridge`, which maintains canonical `ModelMessage` objects and updates the original mutable list incrementally.

The bridge owns:

- decoding/encoding legacy text, `tool_use`, and `tool_result` blocks;
- appending assistant messages and tool results;
- Merchant host reminders;
- latest-user/exchange helpers;
- closing open tool calls after cancellation/interruption;
- preserving the existing persisted conversation shape.

Incremental mutation is required so a disconnect after a tool starts cannot leave stored history invalid.

## 9. Shared model round runner

Add `commerce_common/model_round.py`. Shopping and Merchant share `ModelRoundRunner`, not a common agent superclass.

The runner owns:

- consuming canonical runtime events;
- emitting host `text_delta` events;
- collecting tool calls;
- progressive presentation frames from argument deltas;
- eager dispatch after a valid `ToolCallCompleted`;
- joining outcomes;
- usage and stop-reason accumulation;
- malformed-call recovery;
- cancellation/backstop behavior.

It returns a provider-neutral `ModelRoundResult` containing the canonical assistant message, tool calls, stop reason, usage, provider state, and malformed call IDs.

Role-specific grounding, staging reminders, tool registries, provenance, guardrails, backend rules, and approval remain outside this runner.

## 10. Tool lifecycle and retries

Canonical lifecycle:

```text
PROPOSED → ARGUMENTS_COMPLETE → DISPATCHED → SETTLED
```

A tool executes only after valid complete arguments. Partial arguments may drive `ui_partial` but never execute a tool.

On stream interruption:

- settled calls persist their real outcomes;
- dispatched/unsettled calls are cancelled where possible and otherwise close with an interrupted result;
- proposed-only calls do not execute.

Generic runtime code must never automatically retry Commerce writes. Model-call retry is allowed only for safe pre-output transient failures. Once visible output/tool state has escaped, the same round is not blindly replayed.

## 11. Prompt assembly and caching

`commerce_common/prompt_assembly.py` remains responsible for semantic prompt construction but no longer writes Anthropic-specific cache/eager-input fields.

It expresses intent using stable/dynamic system segments, cache policy, and `FunctionToolSpec.progressive`. Provider adapters map those intents to native request fields.

## 12. Shopping and Merchant orchestration

`ShoppingAgent` and `MerchantAgent` stop constructing provider-native request dictionaries and stop reading provider-native response blocks.

Both build canonical `ModelRequest`s and call `ModelRoundRunner`.

Shopping retains prefetch, grounding, cart provenance, presentation, close-on-presentation, and history compaction.

Merchant retains merchant context prefetch, grounding, `STAGING_FOLLOWTHROUGH_REMINDER`, delegate progress, change ledger, approval, staging provenance, and guardrails.

The host-facing `AgentEvent` stream remains unchanged.

## 13. Memory extraction

`commerce_common.memory` must stop importing Anthropic clients/types.

Keep unchanged:

- `MemoryStore`;
- `validate_fact`;
- write filters;
- retention;
- purge generation;
- tier-one selection;
- source-session digest behavior.

Introduce a `MemoryExtractor` protocol and default `LLMMemoryExtractor` using `ModelRuntime.complete()`. Model-proposed facts still pass validation, write filters, and purge-generation checks before storage.

## 14. Merchant analysis

The analysis delegate remains an isolated constrained model loop.

Default v1 mode is portable analysis using read tools, `execute_analysis_query` when supported, progress reporting, schema-validated `submit_analysis`, and existing iteration/wall-clock budgets.

Hosted code execution is optional and provider-specific. Provider continuation/container/interaction identifiers live only inside `ProviderState`. Analysis output never grants write authority; subsequent writes still pass staging provenance, guardrails, host approval, and apply-time guardrail rechecks.

## 15. Error taxonomy

Provider SDK exceptions are normalized to:

```text
ModelRuntimeError
AuthenticationError
RateLimitError
TransientProviderError
ModelUnavailableError
InvalidRequestError
UnsupportedCapabilityError
ProviderProtocolError
StreamInterruptedError
```

Malformed model-generated tool arguments are a recoverable `MalformedToolCall`, converted into a tool error result so the model may retry. A malformed tool call does not execute.

## 16. Observability

Create canonical model-call records containing provider, model, operation, session tag, turn/round IDs, latency, stop reason, token usage, tool-call count, retry count, degraded capabilities, and normalized error class.

Canonical usage distinguishes unavailable values from zero. Recommended fields:

```text
input_tokens
output_tokens
cached_input_tokens
reasoning_tokens
total_tokens
provider_details
```

Runtime code does not hard-code model pricing. Deployment tooling may derive cost from provider/model/timestamp/usage.

Never log the raw session credential. DEBUG request/response body logging remains opt-in and must be treated as sensitive because prompts may contain memory, cart, account, or merchant context.

## 17. Testing architecture

Split tests into four layers:

1. Core safety tests — fencing, provenance, guardrails, backend rules; mostly unchanged.
2. Provider-neutral agent behavior — use `FakeModelRuntime`; migrate `test_turn_loop.py` and orchestrator/consumption tests away from Anthropic wire assertions.
3. Provider adapter contract tests — raw provider fixtures → canonical requests/events/responses.
4. Live provider evaluations — manual/nightly, not normal PR CI.

`FakeModelRuntime` replaces `FakeClient` as the primary agent-loop test seam. Keep `FakeClient` temporarily for Anthropic adapter compatibility tests, then remove after migration.

Anthropic must reach behavioral parity before OpenAI or Gemini adapters are accepted.

## 18. Migration sequence

Implementation is split into four independently reviewable plans:

1. `provider-neutral-runtime-foundation`: canonical runtime package, Anthropic adapter, conversation bridge, common round runner, Shopping/Merchant migration, memory/analysis migration, deterministic Anthropic parity.
2. `openai-runtime-adapter`: OpenAI request/stream/complete mapping, capability profile, adapter contract tests, provider seam integration.
3. `gemini-runtime-adapter`: Gemini request/stream/complete mapping, capability profile, adapter contract tests, provider seam integration.
4. `cross-provider-evals-rollout`: live evaluation runner/cases, reporting, deployment docs, migration/deprecation docs, release verification.

## 19. Definition of Done

The v1 work is complete when:

- `commerce-common` has no mandatory Anthropic dependency for Messages-runtime logic.
- Shopping and Merchant Messages runtimes run through `ModelRuntime` without provider SDK types in their orchestrators.
- Anthropic behavior matches the original deterministic suite: same backend calls, safety decisions, `AgentEvent`s, persisted conversation semantics, memory validation, staging/approval behavior, and interruption handling.
- OpenAI and Gemini pass the shared provider contract and provider-neutral agent behavior suite for supported capabilities.
- Optional capability degradation is explicit and tested; required safety/control capability gaps fail startup validation.
- Main, memory, and analysis model targets can use independent providers.
- Claude Agent SDK and Managed Agents continue working without being forced through the new abstraction.
- `scripts/verify_all.py` and the repository's existing deterministic test suite pass.
