# shopping-agent/runtime-messages-api (package `shopping_agent_runtime`)

The shopping agent's provider-neutral Messages-style turn loop. `ShoppingAgent` builds
one semantic prompt/tool surface, resolves a `ModelRuntime`, streams canonical model
events, executes Commerce tools through `shopping_agent.executor`, and yields the same
host `AgentEvent` protocol regardless of model provider.

The v1 foundation ships `AnthropicRuntime` as the behavioral reference adapter. OpenAI
and Gemini adapters are separate follow-on plans. Claude Agent SDK and Managed Agents are
not routed through this package.

| Module | Holds |
|---|---|
| `orchestrator.py` | `ShoppingAgent`: constructor, `stream_turn`, `update_memory` |

## Use

The backward-compatible default remains Anthropic:

```python
from pathlib import Path

from commerce_common.streaming import to_sse
from shopping_agent import ShoppingAgentConfig, ShoppingSessionContext, ShoppingSessionState
from shopping_agent_runtime import ShoppingAgent

agent = ShoppingAgent(
    backend=your_backend,
    skills_dir=Path("shopping-agent/skills"),
    config=ShoppingAgentConfig(brand_name="Your Store"),
    memory_store=your_store,
)

state = ShoppingSessionState()
session = ShoppingSessionContext(session_id=sid, user_id=uid)
async for event in agent.stream_turn(messages, session, state):
    send(to_sse(event))
await agent.update_memory(messages, session)
```

For explicit runtime injection:

```python
from commerce_model_runtime import RuntimeRegistry
from commerce_model_runtime.providers import AnthropicRuntime
from shopping_agent import ShoppingAgentConfig
from shopping_agent_runtime import ShoppingAgent

anthropic = AnthropicRuntime(client=your_anthropic_client)
registry = RuntimeRegistry([anthropic])

agent = ShoppingAgent(
    backend=your_backend,
    config=ShoppingAgentConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        memory_provider="anthropic",
    ),
    runtimes=registry,
)
```

`runtime=` injects one runtime. `runtimes=` injects a registry and is the preferred form
when conversation and memory may use different providers. `client=` remains an Anthropic-
only v1 compatibility path and is not the new provider abstraction.

`messages` remains the mutable host conversation. The turn appends assistant messages and
tool results in place. Provider SDK objects and raw SSE events do not escape the adapter;
`ui`, `ui_partial`, `cart_update`, `tool_call`, `tool_result`, and `turn_complete` keep their
existing host shapes.

## Runtime behavior

- Grounding still forces the configured read tool on the first round when a grounding rule
  fires; the final iteration still forces text-only output.
- Commerce tools execute through the existing executor/gates. A provider adapter never
  executes a cart or backend action directly.
- Eager dispatch starts only after canonical tool arguments are complete. Progressive UI
  uses canonical argument deltas when the selected runtime supports them; otherwise only
  the final component is required.
- Cache and reasoning settings are semantic intents. `AnthropicRuntime` maps them to the
  Anthropic wire format; other adapters own their own mapping.
- Malformed streamed tool input is returned to the model as a tool error and is logged by
  size only, never with the malformed input text.
- `agent.memory` remains the deployment's `MemoryRuntime`; post-turn extraction resolves
  `config.memory_target()` independently from the conversation target.

The safety gates themselves (fencing, provenance, cart caps, memory validation) remain in
`shopping_agent` and `commerce_common`; [`docs/safety.md`](../../docs/safety.md) lists them.

Credentials for the default adapter are still the Anthropic SDK's environment variables.
Tests can use the compatibility `commerce_common.testing.FakeClient` or, for new provider-
neutral tests, `commerce_model_runtime.testing.FakeModelRuntime`.
