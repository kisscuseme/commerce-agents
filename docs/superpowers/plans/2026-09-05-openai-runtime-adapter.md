# OpenAI Runtime Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an OpenAI adapter that satisfies the canonical `ModelRuntime` contract and passes shared provider-neutral agent behavior tests.

**Architecture:** Keep every OpenAI SDK/request/stream/tool/reasoning/usage/search/error detail in `commerce_model_runtime/providers/openai.py`. Shopping/Merchant must not gain OpenAI-specific branches.

**Tech Stack:** Python 3.11+, OpenAI Python SDK, pytest, asyncio.

**Spec:** `docs/superpowers/specs/2026-09-05-provider-neutral-commerce-runtime-design.md`

## Global Constraints

- Foundation/Anthropic parity must be green first.
- Raw OpenAI objects/events never escape the adapter.
- No automatic provider fallback.
- Required safety/control capability gaps fail validation.
- `enable_web_search=True` requires OpenAI built-in search capability in v1.

---

### Task 1: Dependency extra, runtime skeleton, capabilities

**Files:**
- Modify: `commerce-model-runtime/pyproject.toml`
- Create: `commerce-model-runtime/commerce_model_runtime/providers/openai.py`
- Modify: providers `__init__.py`
- Test: `commerce-model-runtime/tests/providers/test_openai_capabilities.py`

**Interfaces:** Produces `OpenAIRuntime(client: AsyncOpenAI | None = None, *, timeout_s: float = 120.0)`.

- [ ] Write a failing test asserting `provider == "openai"` and model capability resolution.
- [ ] Run the test and verify failure.
- [ ] Add optional `openai` dependency and runtime skeleton.
- [ ] Implement `capabilities_for()` inside the adapter only.
- [ ] Re-run and verify PASS.
- [ ] Commit: `git commit -m "feat: add OpenAI runtime skeleton"`.

---

### Task 2: Canonical request mapping

**Files:**
- Modify: OpenAI adapter
- Test: `test_openai_request.py`

- [ ] Write failing tests for system/messages, function tools, AUTO/NONE/SPECIFIC tool choice, max tokens, reasoning, metadata, and provider continuation state.

```python
request = canonical_request(tool_choice=ToolChoice.specific("search_products"))
body = runtime._build_request(request)
assert body["tool_choice"]
```

- [ ] Run and verify failure.
- [ ] Implement canonical → OpenAI Responses API mapping only inside the adapter.
- [ ] Preserve canonical tool IDs; retain provider-native IDs only as mapping metadata.
- [ ] Run tests and commit: `git commit -m "feat: map canonical requests to OpenAI"`.

---

### Task 3: Streaming normalization

**Files:**
- Modify: OpenAI adapter
- Test: `test_openai_stream.py`

- [ ] Write synthetic text-stream tests.
- [ ] Write function-argument delta/completion tests.
- [ ] Write malformed/incomplete JSON tests expecting `ToolCallFailed` and no completed call.
- [ ] Write multi-tool ordering tests.
- [ ] Implement canonical event normalization: text, call started, arg delta, call completed/failed, usage, response completed.
- [ ] Run tests and commit: `git commit -m "feat: normalize OpenAI streaming events"`.

---

### Task 4: Completion, usage, normalized errors

**Files:**
- Modify: OpenAI adapter
- Test: `test_openai_complete.py`, `test_openai_errors.py`

- [ ] Write failing tests for text/tool responses, stop reason, usage, provider state.
- [ ] Write error mapping tests for auth, rate limit, transient/network, invalid request, unavailable model.
- [ ] Implement `complete()` and `ModelUsage` mapping, using `None` for unavailable counters.
- [ ] Normalize SDK exceptions to runtime errors.
- [ ] Run tests and commit: `git commit -m "feat: complete OpenAI runtime contract"`.

---

### Task 5: Built-in web search

**Files:**
- Modify: OpenAI adapter
- Test: `test_openai_web_search.py`, `tests/test_provider_seams.py`

- [ ] Write capability validation tests proving unsupported models fail startup when web search is enabled.
- [ ] Implement `BuiltinToolSpec(kind="web_search")` mapping.
- [ ] Ensure returned third-party material remains untrusted/fenced before authoritative use.
- [ ] Run tests and commit: `git commit -m "feat: support OpenAI built-in web search"`.

---

### Task 6: Shared provider contract

**Files:**
- Create/Modify: `commerce-model-runtime/tests/providers/test_provider_contract.py`
- Modify: `tests/test_provider_seams.py`

- [ ] Parameterize common text, one-tool, multi-tool, forced-tool, no-tool, malformed-tool, usage, and optional streamed-argument scenarios.
- [ ] Run OpenAI through the shared contract.
- [ ] Run Shopping/Merchant provider-seam scenarios without live network calls.
- [ ] Fix abstraction bugs without provider branches in role agents.
- [ ] Run provider tests and commit: `git commit -m "test: verify OpenAI provider parity"`.

---

### Task 7: Docs and full verification

**Files:**
- Modify: `docs/deployment.md`, both Messages-runtime READMEs, `requirements.txt`

- [ ] Add OpenAI to all-provider/root development install.
- [ ] Document credentials and `provider="openai"` examples.
- [ ] Document capability caveats/no cross-provider fallback.
- [ ] Run `pytest -q` and `python scripts/verify_all.py`.
- [ ] Commit: `git commit -m "docs: document OpenAI runtime provider"`.

## Completion Gate

- [ ] OpenAI adapter contract passes
- [ ] shared provider contract passes for supported capabilities
- [ ] Shopping/Merchant contain no OpenAI-specific branches
- [ ] required capability gaps fail validation
- [ ] full deterministic suite and repository verifier pass
