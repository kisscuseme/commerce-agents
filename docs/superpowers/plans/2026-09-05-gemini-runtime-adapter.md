# Gemini Runtime Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gemini adapter satisfying the canonical `ModelRuntime` contract and the same shared provider behavior suite used by Anthropic/OpenAI.

**Architecture:** All Google Gen AI SDK/contents/parts/function-calling/thinking/usage/search details stay inside `commerce_model_runtime/providers/gemini.py`; provider differences are expressed with `ModelCapabilities`, never role-agent branches.

**Tech Stack:** Python 3.11+, Google Gen AI Python SDK, pytest, asyncio.

**Spec:** `docs/superpowers/specs/2026-09-05-provider-neutral-commerce-runtime-design.md`

## Global Constraints

- Anthropic foundation parity must be green first.
- Gemini-native objects/events never escape the adapter.
- Missing streamed tool-argument support may degrade to final-only UI; required forced-tool/control capabilities may not degrade.
- No cross-provider fallback inside a turn.

---

### Task 1: Dependency extra, skeleton, capabilities

**Files:**
- Modify: `commerce-model-runtime/pyproject.toml`
- Create: `commerce-model-runtime/commerce_model_runtime/providers/gemini.py`
- Modify: providers `__init__.py`
- Test: `test_gemini_capabilities.py`

- [ ] Write failing provider/capability tests.
- [ ] Run and verify failure.
- [ ] Add `google-genai` optional dependency and `GeminiRuntime` skeleton.
- [ ] Implement adapter-owned capability profiles.
- [ ] Re-run and commit: `git commit -m "feat: add Gemini runtime skeleton"`.

---

### Task 2: Canonical request mapping

**Files:**
- Modify: Gemini adapter
- Test: `test_gemini_request.py`

- [ ] Write failing tests for system instruction, user/assistant content, function declarations, AUTO/NONE/SPECIFIC tool choice, reasoning/thinking intent, and provider state.

```python
request = canonical_request(tool_choice=ToolChoice.specific("search_products"))
body = runtime._build_request(request)
assert body is not None
```

- [ ] Run and verify failure.
- [ ] Implement canonical → Gemini contents/tools/config mapping only inside the adapter.
- [ ] Preserve canonical tool IDs and keep provider interaction IDs opaque.
- [ ] Run tests and commit: `git commit -m "feat: map canonical requests to Gemini"`.

---

### Task 3: Stream normalization and capability-aware partial arguments

**Files:**
- Modify: Gemini adapter
- Test: `test_gemini_stream.py`

- [ ] Write text streaming tests.
- [ ] Write function call completion tests.
- [ ] For models/chunks providing partial args, assert canonical `ToolArgumentsDelta` events.
- [ ] For models without useful partial args, assert no partial-event requirement but a correct final `ToolCallCompleted`.
- [ ] Write malformed-call tests.
- [ ] Implement canonical stream normalization and run tests.
- [ ] Commit: `git commit -m "feat: normalize Gemini streaming events"`.

---

### Task 4: Completion, usage, thinking, normalized errors

**Files:**
- Modify: Gemini adapter
- Test: `test_gemini_complete.py`, `test_gemini_errors.py`

- [ ] Write failing tests for text/tool completion, stop reason, usage, provider state.
- [ ] Write auth/rate-limit/transient/invalid/unavailable error mapping tests.
- [ ] Implement `complete()` and usage normalization; unavailable counters use `None`.
- [ ] Map semantic reasoning only when capability says the selected model supports it; unsupported explicit levels fail validation.
- [ ] Run tests and commit: `git commit -m "feat: complete Gemini runtime contract"`.

---

### Task 5: Built-in web search

**Files:**
- Modify: Gemini adapter
- Test: `test_gemini_web_search.py`

- [ ] Write tests showing `enable_web_search=True` fails startup for unsupported capability profiles.
- [ ] Implement `BuiltinToolSpec(kind="web_search")` mapping.
- [ ] Ensure search material remains fenced/untrusted before authoritative use.
- [ ] Run and commit: `git commit -m "feat: support Gemini built-in web search"`.

---

### Task 6: Shared provider contract and role seams

**Files:**
- Modify: `test_provider_contract.py`, `tests/test_provider_seams.py`

- [ ] Add Gemini to shared text/tool/forced/none/multi/malformed/usage scenarios.
- [ ] Add capability-aware progressive UI expectations.
- [ ] Run Shopping/Merchant scenarios through Gemini adapter fixtures without live calls.
- [ ] Fix only adapter/provider-neutral abstraction bugs.
- [ ] Commit: `git commit -m "test: verify Gemini provider parity"`.

---

### Task 7: Docs and verification

**Files:**
- Modify: `docs/deployment.md`, both Messages-runtime READMEs, `requirements.txt`

- [ ] Add Google Gen AI to root all-provider development install.
- [ ] Document Gemini credentials/config/examples and capability validation.
- [ ] Run `pytest -q` and `python scripts/verify_all.py`.
- [ ] Commit: `git commit -m "docs: document Gemini runtime provider"`.

## Completion Gate

- [ ] Gemini adapter contract passes
- [ ] shared provider contract passes for supported capabilities
- [ ] progressive UI degrades explicitly where partial args are unavailable
- [ ] Shopping/Merchant contain no Gemini-specific branches
- [ ] full deterministic suite and repository verifier pass
