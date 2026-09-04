# merchant-agent/runtime-messages-api (package `merchant_agent_runtime`)

The merchant agent's provider-neutral Messages-style turn loop. `MerchantAgent` resolves a
`ModelRuntime`, streams canonical model events, executes all reads/staged writes through the
existing `MerchantToolExecutor`, and keeps the same host `AgentEvent` protocol.

The v1 foundation ships `AnthropicRuntime` as the behavioral reference adapter. OpenAI and
Gemini adapters are follow-on plans. Claude Agent SDK and Managed Agents stay Claude-only.

| Module | Holds |
|---|---|
| `orchestrator.py` | `MerchantAgent`: constructor, `stream_turn`, `update_memory` |
| `analysis.py` | provider-neutral `AnalysisRunner` behind `run_analysis` |

## Use

The default remains Anthropic-compatible:

```python
from pathlib import Path

from commerce_common.streaming import to_sse
from merchant_agent import MerchantAgentConfig, MerchantSessionContext, MerchantSessionState
from merchant_agent_runtime import MerchantAgent

agent = MerchantAgent(
    backend=your_backend,
    skills_dir=Path("merchant-agent/skills"),
    config=MerchantAgentConfig(brand_name="Your Store"),
    memory_store=your_store,
)

state = MerchantSessionState()
session = MerchantSessionContext(session_id=sid, merchant_id=mid, operator=who)
async for event in agent.stream_turn(messages, session, state):
    send(to_sse(event))
await agent.update_memory(messages, session)
```

For explicit runtime selection:

```python
from commerce_model_runtime import RuntimeRegistry
from commerce_model_runtime.providers import AnthropicRuntime
from merchant_agent import MerchantAgentConfig
from merchant_agent_runtime import MerchantAgent

anthropic = AnthropicRuntime(client=your_anthropic_client)
registry = RuntimeRegistry([anthropic])

agent = MerchantAgent(
    backend=your_backend,
    config=MerchantAgentConfig(
        provider="anthropic",
        model="claude-opus-5",
        memory_provider="anthropic",
        analysis_provider="anthropic",
    ),
    runtimes=registry,
)
```

`runtime=` injects one runtime. `runtimes=` is preferred when the main conversation,
memory extraction, and analysis delegate may target different providers/models. `client=`
remains an Anthropic-only v1 compatibility path.

## Runtime behavior and safety

- Grounding, staging follow-through, provenance, guardrails, change staging, and host
  approval remain Merchant-layer policy. Provider adapters have no write authority.
- Eager dispatch starts only after canonical tool arguments are complete. Delegate progress
  still streams while the corresponding Commerce tool execution is in flight.
- `STAGING_FOLLOWTHROUGH_REMINDER` is still appended at most once when a change request
  would otherwise finish without a staging attempt.
- Cache and reasoning settings are semantic intents; each provider adapter maps them to its
  own request format.
- Malformed streamed tool input is returned as a tool error and logged by metadata/length,
  never by argument contents.
- `update_memory()` resolves `memory_target()` independently from the main model target.
- With `enable_analysis=True`, `AnalysisRunner` resolves `analysis_target()` independently.
  Portable analysis uses ordinary read/query tools and `ModelRuntime.complete()`.
- Hosted code execution is an optional runtime capability. Its provider continuation state
  stays opaque, while any later staged write still passes provenance, guardrails, approval,
  and apply-time checks.

`messages` remains the mutable persisted conversation and the public `AgentEvent` schema is
unchanged. The approval mark remains host code writing `state.approved_change_ids`.
[`docs/safety.md`](../../docs/safety.md) documents the enforced gates.

Credentials for the default adapter are still the Anthropic SDK's environment variables.
Existing tests may continue to use `commerce_common.testing.FakeClient`; new provider-
neutral tests should prefer `commerce_model_runtime.testing.FakeModelRuntime`.
