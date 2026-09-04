# Deployment platforms and model runtimes

The Messages-style Shopping and Merchant runtimes now depend on the provider-neutral
`commerce-model-runtime` contract. The v1 foundation keeps Anthropic as the default and
ships `AnthropicRuntime` as the golden-reference adapter. OpenAI and Gemini adapters are
implemented in separate follow-on plans; do not set `provider="openai"` or
`provider="gemini"` until those runtimes are registered by the deployment.

Claude Agent SDK and Managed Agents are deliberately unchanged and remain Claude-specific.
The provider-neutral work applies only to `shopping-agent/runtime-messages-api` and
`merchant-agent/runtime-messages-api`.

## Runtime selection

Each Messages agent accepts three mutually exclusive injection forms:

- no injection: construct the default `AnthropicRuntime`;
- `runtime=`: use one `ModelRuntime` for the configured main target;
- `runtimes=`: use a `RuntimeRegistry`, required when main conversation, memory extraction,
  or Merchant analysis use different providers;
- `client=`: Anthropic-only v1 compatibility path for existing deployments/tests.

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
        memory_model="claude-haiku-4-5-20251001",
    ),
    runtimes=registry,
)
```

Merchant adds `analysis_provider` and `analysis_model`. A field whose provider is `None`
inherits the main provider. `analysis_model=None` inherits the main model. When a non-
Anthropic main provider is selected and `memory_provider` is left unset, the untouched
legacy Claude memory-model default also follows the main model rather than constructing an
invalid cross-provider target.

Runtime capability validation runs before the first model request. Safety/control
requirements such as function tools and forced tool choice fail configuration rather than
silently degrading. Optional performance features such as prompt caching or streamed tool
arguments may degrade explicitly.

## Anthropic deployments

`AnthropicRuntime` accepts any compatible async Anthropic client, including the first-party
API, Vertex AI, Bedrock/Mantle, Foundry, and an in-house Messages-compatible gateway.
Existing `client=` deployments continue to work, but new code should prefer wrapping the
client in `AnthropicRuntime` and injecting `runtime=` or `runtimes=`.

| Anthropic target | Messages runtime client |
|---|---|
| Anthropic API | `AsyncAnthropic` (default) |
| GCP Vertex AI | `AsyncAnthropicVertex` |
| AWS Bedrock Mantle | `AsyncAnthropicBedrockMantle` |
| AWS Bedrock Invoke API | `AsyncAnthropicBedrock` |
| Microsoft Foundry | `AsyncAnthropicFoundry` |
| In-house gateway | `AsyncAnthropic(base_url=..., auth_token=...)` |

```python
from anthropic import AsyncAnthropicVertex
from commerce_model_runtime.providers import AnthropicRuntime
from shopping_agent import ShoppingAgentConfig
from shopping_agent_runtime import ShoppingAgent

runtime = AnthropicRuntime(
    client=AsyncAnthropicVertex(project_id="your-project", region="global")
)
agent = ShoppingAgent(
    backend=your_backend,
    config=ShoppingAgentConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        memory_model="claude-haiku-4-5@20251001",
    ),
    runtime=runtime,
)
```

The default client reads the Anthropic SDK's normal environment configuration, including
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, and `ANTHROPIC_BASE_URL` where applicable.
The repository root install includes `commerce-model-runtime[anthropic]`; individual
provider deployments may install only the extras they need as additional adapters land.

### Anthropic model ids

Model ids remain deployment strings and their grammar depends on the platform. The repo
defaults are `claude-sonnet-5` for Shopping, `claude-opus-5` for Merchant, and
`claude-haiku-4-5-20251001` for memory. Vertex commonly uses `@`-dated snapshots; Bedrock
uses the ids required by its selected endpoint; Foundry uses deployment names. Confirm ids
against the platform catalog.

## Main, memory, and analysis targets

Messages runtimes no longer assume that every operation must share one provider client.
`RuntimeRegistry` resolves each operation independently:

```text
Shopping turn  -> config.model_target()
Memory pass    -> config.memory_target()
Merchant turn  -> config.model_target()
Analysis       -> config.analysis_target()
```

Provider switching never occurs automatically in the middle of a turn. The selected main
runtime is fixed for that turn. Opaque provider continuation state is passed back only to
the same provider. Commerce provenance, gates, staged changes, approval marks, and memory
store state remain provider-independent.

Merchant portable analysis uses ordinary read/query tools through `ModelRuntime.complete()`.
Hosted code execution is optional and requires the selected analysis runtime/model to
advertise `hosted_code_execution`; provider container/continuation identifiers remain in
opaque `ProviderState` and never grant Commerce write authority.

## Claude Agent SDK runtimes

The Agent SDK paths are intentionally not generalized. `claude-agent-sdk` starts the Claude
Code CLI, and the CLI selects Anthropic/Vertex/Bedrock/Foundry or a compatible gateway from
its environment.

```python
from claude_agent_sdk import ClaudeSDKClient
from shopping_agent_sdk import make_options

options, toolset = make_options()
options.env.update({"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_REGION": "us-east-1"})
async with ClaudeSDKClient(options=options) as client:
    ...
```

Common selectors remain `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_MANTLE`,
`CLAUDE_CODE_USE_VERTEX`, `CLAUDE_CODE_USE_FOUNDRY`, and `ANTHROPIC_BASE_URL`. See Claude
Code documentation for the platform-specific credential and model-id variables.

## Managed Agents

Managed Agents are also unchanged. `scripts/deploy_managed_agent.sh` reads
`ANTHROPIC_API_KEY` and posts to `ANTHROPIC_API_URL` (default `https://api.anthropic.com`).
A gateway for this path must proxy the Skills, Agents, Environments, Sessions, and session
stream APIs, not only `/v1/messages`.

## Tests and verification

The test layers are now split by responsibility:

1. Commerce safety/business tests remain provider-independent.
2. Agent-loop behavior can use `commerce_model_runtime.testing.FakeModelRuntime`.
3. Anthropic request/event wire behavior lives in `commerce-model-runtime/tests/providers`.
4. Existing `commerce_common.testing.FakeClient` remains as a compatibility re-export of
   the Anthropic provider fake so the original parity suite can keep running.
5. Live provider evaluations are separate from normal deterministic CI.

`tests/test_platform_seams.py` continues to cover the Anthropic transport/platform clients.
`tests/test_provider_seams.py` covers the new runtime registry boundary. Run
`python scripts/verify_all.py` before merging; a deployment should additionally run a live
conversation against each provider/model it enables.
