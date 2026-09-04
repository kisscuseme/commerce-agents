from commerce_model_runtime import ModelTarget, ReasoningEffort
from commerce_common.config import BaseAgentConfig, DEFAULT_MEMORY_MODEL


class Config(BaseAgentConfig):
    model: str = "claude-main"


def test_existing_config_defaults_to_anthropic_targets():
    config = Config()
    assert config.model_target() == ModelTarget("anthropic", "claude-main")
    assert config.memory_target() == ModelTarget("anthropic", DEFAULT_MEMORY_MODEL)


def test_memory_provider_inherits_main_provider_and_model_when_legacy_default_is_not_portable():
    config = Config(provider="openai", model="gpt-main")
    assert config.memory_target() == ModelTarget("openai", "gpt-main")


def test_explicit_memory_provider_and_model_override_inheritance():
    config = Config(
        provider="openai",
        model="gpt-main",
        memory_provider="anthropic",
        memory_model="claude-haiku-custom",
    )
    assert config.memory_target() == ModelTarget("anthropic", "claude-haiku-custom")


def test_thinking_effort_maps_to_semantic_reasoning_without_changing_public_field():
    config = Config(thinking_effort="xhigh")
    assert config.thinking_effort == "xhigh"
    assert config.reasoning_config().effort is ReasoningEffort.XHIGH
    assert Config(thinking_effort=None).reasoning_config().effort is ReasoningEffort.OFF
