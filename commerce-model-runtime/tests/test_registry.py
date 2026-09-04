import pytest

from commerce_model_runtime import ModelTarget, RuntimeNotRegisteredError, RuntimeRegistry


class FakeRuntime:
    provider = "fake"

    def capabilities_for(self, target):
        return object()

    async def stream(self, request):
        if False:
            yield None

    async def complete(self, request):
        raise NotImplementedError


def test_registry_resolves_provider_case_insensitively():
    runtime = FakeRuntime()
    registry = RuntimeRegistry([runtime])
    assert registry.resolve(ModelTarget(provider="FAKE", model="m")) is runtime


def test_registry_rejects_duplicate_provider_names():
    with pytest.raises(ValueError, match="duplicate provider"):
        RuntimeRegistry([FakeRuntime(), FakeRuntime()])


def test_registry_names_missing_provider():
    registry = RuntimeRegistry([FakeRuntime()])
    with pytest.raises(RuntimeNotRegisteredError, match="openai"):
        registry.resolve(ModelTarget(provider="openai", model="gpt"))
