import pytest

from commerce_model_runtime import (
    CapabilityStatus,
    CapabilityValidationError,
    ModelCapabilities,
    ModelOperation,
    validate_capabilities,
)


def test_main_turn_rejects_missing_specific_tool_choice():
    caps = ModelCapabilities.full().replace(tool_choice_specific=False)
    with pytest.raises(CapabilityValidationError, match="tool_choice_specific"):
        validate_capabilities(ModelOperation.MAIN_TURN, caps, enable_web_search=False)


def test_main_turn_marks_optional_performance_gaps_as_degraded():
    caps = ModelCapabilities.full().replace(prompt_cache=False, stream_tool_arguments=False)
    plan = validate_capabilities(ModelOperation.MAIN_TURN, caps, enable_web_search=False)
    assert plan.status is CapabilityStatus.DEGRADED
    assert plan.degraded == frozenset({"prompt_cache", "stream_tool_arguments"})


def test_web_search_is_required_only_when_enabled():
    caps = ModelCapabilities.full().replace(builtin_web_search=False)
    assert validate_capabilities(ModelOperation.MAIN_TURN, caps, enable_web_search=False)
    with pytest.raises(CapabilityValidationError, match="builtin_web_search"):
        validate_capabilities(ModelOperation.MAIN_TURN, caps, enable_web_search=True)


def test_memory_extraction_does_not_require_streaming():
    caps = ModelCapabilities.full().replace(stream_text=False, stream_tool_arguments=False)
    plan = validate_capabilities(ModelOperation.MEMORY_EXTRACTION, caps)
    assert plan.status is CapabilityStatus.SUPPORTED


def test_portable_analysis_requires_function_tools_and_continuation():
    caps = ModelCapabilities.full().replace(tool_result_continuation=False)
    with pytest.raises(CapabilityValidationError, match="tool_result_continuation"):
        validate_capabilities(ModelOperation.PORTABLE_ANALYSIS, caps)
