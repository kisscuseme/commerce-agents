# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""The mechanisms both agent roles build on. Import from the submodules:

``config``            provider-aware ``BaseAgentConfig`` and model targets
``types``             ``MemoryFact``, ``MemoryCategory``, ``ClockContext``
``fencing``           ``Fence``, chip and display-text hygiene
``memory``            ``MemoryStore``, write filters, retention, ``MemoryRuntime``
``model_memory``      provider-neutral post-turn memory extraction
``conversation``      legacy host-message <-> canonical model-message bridge
``model_round``       canonical model-event runner and normalized usage helpers
``skills``            ``SkillRegistry``
``prompt_assembly``   semantic system/tool/cache intent for model runtimes
``grounding``         ``GroundingRule`` and the lexicon matchers
``presentation``      ``PresentationComponent``, ``PresentationExtension``, the runner
``delegation``        ``DelegateExtension``
``execution``         ``BaseToolExecutor``, the frame each role's executor extends
``streaming``         ``AgentEvent``, ``ToolOutcome``, ``to_sse``
``turn``              provider-independent turn, dispatch, compaction, and recovery helpers
``agent_sdk``         plumbing for the Claude Agent SDK runtimes
``mcp_server``        plumbing for the reference MCP servers
``manifest``          resolves a Managed Agent manifest
``testing``           Commerce test helpers plus deprecated Anthropic fake re-exports

Provider SDK request/response/event parsing lives in ``commerce-model-runtime`` rather than
this package.
"""
