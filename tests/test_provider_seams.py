from __future__ import annotations

import pytest

from commerce_model_runtime import RuntimeRegistry
from commerce_model_runtime.testing import FakeModelRuntime, text_round
from merchant_agent_runtime import MerchantAgent
from shopping_agent_runtime import ShoppingAgent

AGENTS = {"shopping": ShoppingAgent, "merchant": MerchantAgent}


@pytest.fixture(params=list(AGENTS))
def role(request) -> str:
    return request.param


def test_agent_accepts_a_provider_neutral_runtime(role, backend, skills, config):
    runtime = FakeModelRuntime([text_round("ok")], provider="anthropic")
    agent = AGENTS[role](backend=backend, skills=skills, config=config, runtime=runtime)
    assert agent.runtime is runtime
    assert agent.runtimes.resolve(config.model_target()) is runtime


def test_agent_resolves_main_runtime_from_registry(role, backend, skills, config):
    runtime = FakeModelRuntime([text_round("ok")], provider="fake")
    registry = RuntimeRegistry([runtime])
    provider_config = config.model_copy(update={"provider": "fake", "model": "fake-model"})
    agent = AGENTS[role](
        backend=backend,
        skills=skills,
        config=provider_config,
        runtimes=registry,
    )
    assert agent.runtime is runtime
    assert agent.config.model_target().provider == "fake"


def test_merchant_can_resolve_analysis_runtime_independently(backend, skills, config):
    main = FakeModelRuntime([text_round("ok")], provider="fake")
    analysis = FakeModelRuntime([text_round("analysis")], provider="analysis-fake")
    registry = RuntimeRegistry([main, analysis])
    merchant_config = config.model_copy(
        update={
            "provider": "fake",
            "model": "main-model",
            "enable_analysis": True,
            "analysis_provider": "analysis-fake",
            "analysis_model": "analysis-model",
            "analysis_use_code_execution": False,
        }
    )
    agent = MerchantAgent(
        backend=backend,
        skills=skills,
        config=merchant_config,
        runtimes=registry,
    )
    assert agent.runtime is main
    assert agent.runtimes.resolve(merchant_config.analysis_target()) is analysis
