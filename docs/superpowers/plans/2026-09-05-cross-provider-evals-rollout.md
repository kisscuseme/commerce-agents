# Cross-Provider Evaluations and Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add repeatable Anthropic/OpenAI/Gemini live evaluations, provider-comparison reporting, deployment guidance, and a safe rollout/deprecation path.

**Architecture:** Keep deterministic CI model-free. Add a separate manual/nightly live-eval runner that executes equivalent commerce scenarios against registered providers and records safety/task outcomes plus usage/latency. Anthropic remains the compatibility default while provider/model selection becomes explicit for main, memory, and analysis roles.

**Tech Stack:** Python 3.11+, async runner/pytest, JSON/Markdown reports, optional GitHub Actions workflow.

**Spec:** `docs/superpowers/specs/2026-09-05-provider-neutral-commerce-runtime-design.md`

## Global Constraints

- Live provider calls never run in normal PR CI.
- Never commit provider keys, raw session IDs, personal memory values, cart/account contents, or merchant private context in fixtures/reports.
- Safety outcomes outrank prose-quality scores.
- Use equivalent scenario semantics across providers.
- Do not add automatic provider fallback.

---

### Task 1: Live evaluation case schema

**Files:**
- Create: `evals/live/types.py`
- Create: `evals/live/cases.py`
- Test: `evals/live/tests/test_cases.py`

**Interfaces:** Produces `EvalCase`, `EvalExpectation`, `ProviderEvalResult`, `SafetyOutcome`.

- [ ] Write failing schema tests.

```python
def test_apply_without_host_approval_is_hard_failure():
    expectation = EvalExpectation(require_host_approval=True)
    assert expectation.require_host_approval is True
```

- [ ] Define cases for Shopping grounding/unseen IDs/cart provenance, Merchant performance grounding/staging follow-through/approval-before-apply, presentation choice, and memory extraction using demo/fake backends only.
- [ ] Run tests and commit: `git commit -m "test: define cross-provider live eval cases"`.

---

### Task 2: Live evaluation runner

**Files:**
- Create: `evals/live/runner.py`
- Create: `evals/live/providers.py`
- Test: `evals/live/tests/test_runner.py`

**Interfaces:** Runner resolves provider/model through `RuntimeRegistry` and returns normalized result records.

- [ ] Write a failing runner test with `FakeModelRuntime`.
- [ ] Implement capture of backend calls, gate outcomes, tool iterations, final-answer presence, `AgentEvent`s, `ModelUsage`, latency, normalized errors.
- [ ] Store only session tags, never raw session credentials.
- [ ] Run tests and commit: `git commit -m "feat: add live provider evaluation runner"`.

---

### Task 3: Safety-first scoring and comparison reports

**Files:**
- Create: `evals/live/scoring.py`
- Create: `evals/live/report.py`
- Test: `evals/live/tests/test_scoring.py`

**Interfaces:** Produces `grounding_pass`, `unsafe_apply_attempt`, `hallucinated_id`, `staging_pass`, `approval_pass`, `task_success`, latency and token metrics.

- [ ] Write failing tests where any unauthorized apply attempt is a hard failure regardless of response quality.
- [ ] Implement per-provider/model/operation aggregation.
- [ ] Generate sanitized JSON and Markdown summaries.
- [ ] Do not hard-code pricing; accept optional external cost enrichment only.
- [ ] Run tests and commit: `git commit -m "feat: add provider evaluation scoring"`.

---

### Task 4: Opt-in execution entrypoint

**Files:**
- Create: `scripts/run_live_provider_evals.py`
- Optional Create: `.github/workflows/live-provider-evals.yml`
- Test: `evals/live/tests/test_cli.py`

- [ ] Write CLI parsing tests for `--provider`, `--model`, optional `--case`, output path.
- [ ] Implement runtime selection via `RuntimeRegistry`; no SDK branching in the runner.
- [ ] Make real network execution explicit/manual/nightly only.
- [ ] If workflow is added, read credentials from Actions secrets and upload only sanitized report artifacts.
- [ ] Run tests and commit: `git commit -m "ci: add opt-in live provider evals"`.

---

### Task 5: Runtime operations and observability docs

**Files:**
- Create: `docs/provider-runtime.md`
- Modify: `docs/deployment.md`

- [ ] Document canonical call record fields: provider, model, operation, turn/round IDs, latency, stop reason, usage, retries, degraded capabilities, normalized error class.
- [ ] Document DEBUG-log sensitivity and retention/access requirements.
- [ ] Document independent main/memory/analysis provider examples.
- [ ] Document `ProviderState` as opaque JSON persistence only.
- [ ] Commit: `git commit -m "docs: add provider runtime operations guide"`.

---

### Task 6: Migration/deprecation documentation

**Files:**
- Modify: `README.md`
- Modify: both Messages-runtime READMEs
- Create: `docs/migration-provider-runtime.md`

- [ ] Document unchanged default Anthropic behavior and existing `client=` compatibility.
- [ ] Mark `client=` injection deprecated in favor of `runtime=`/`runtimes=` without removing it in v1.
- [ ] Document independent provider selection for main/memory/analysis.
- [ ] Explicitly state Agent SDK and Managed Agents remain Claude-specific.
- [ ] Commit: `git commit -m "docs: add provider runtime migration guide"`.

---

### Task 7: Final release verification

**Files:** Modify only if verification reveals defects.

- [ ] Run `pytest -q`.
- [ ] Run `python scripts/verify_all.py`.
- [ ] Run provider contract tests for Anthropic/OpenAI/Gemini.
- [ ] In a credentialed environment, run at least one live smoke case per configured provider.
- [ ] Confirm normal PR CI needs no live credentials.
- [ ] Confirm generated reports contain no raw secrets/session IDs/private payloads.
- [ ] Commit verification fixes separately, then finalize with `git commit -m "chore: finalize provider-neutral runtime rollout"`.

## Completion Gate

- [ ] deterministic CI remains model-free
- [ ] live evals are opt-in/manual/nightly
- [ ] safety-first comparison works across all three providers
- [ ] migration/deprecation docs are complete
- [ ] independent main/memory/analysis provider examples are documented
- [ ] full verification passes
