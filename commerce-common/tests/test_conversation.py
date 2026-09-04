from dataclasses import dataclass

from commerce_model_runtime import (
    ModelMessage,
    ProviderOpaqueContent,
    TextContent,
    ToolCallContent,
    ToolResultContent,
)
from commerce_common.conversation import (
    INTERRUPTED_RESULT_TEXT,
    LegacyConversationBridge,
)


def test_append_assistant_updates_original_list_immediately():
    raw = [{"role": "user", "content": "hello"}]
    bridge = LegacyConversationBridge(raw)
    bridge.append_assistant(ModelMessage(role="assistant", content=[TextContent("hi")]))
    assert raw[-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
    }


def test_decode_preserves_tool_blocks_and_anthropic_opaque_blocks():
    raw = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hidden", "signature": "sig"},
                {"type": "tool_use", "id": "t1", "name": "search_products", "input": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok", "is_error": False}
            ],
        },
    ]
    messages = LegacyConversationBridge(raw).model_messages()
    assert messages[0].content == [
        ProviderOpaqueContent(
            provider="anthropic",
            data={"type": "thinking", "thinking": "hidden", "signature": "sig"},
        ),
        ToolCallContent(
            id="t1",
            name="search_products",
            arguments={"q": "x"},
            provider_tool_call_id="t1",
        ),
    ]
    assert messages[1].content == [ToolResultContent("t1", "ok", False)]


def test_model_messages_merge_adjacent_user_tool_results_and_new_text_without_mutating_raw():
    raw = [
        {"role": "user", "content": "find a tent"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "search", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "user", "content": "thanks"},
    ]
    messages = LegacyConversationBridge(raw).model_messages()
    assert len(messages) == 3
    assert messages[-1].role == "user"
    assert messages[-1].content == [ToolResultContent("t1", "ok", False), TextContent("thanks")]
    assert len(raw) == 4


def test_latest_user_text_and_transcript_ignore_host_text_and_tool_results():
    host = "stage the change before ending"
    raw = [
        {"role": "user", "content": "change the price"},
        {"role": "assistant", "content": [{"type": "text", "text": "checking"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
        {"role": "user", "content": [{"type": "text", "text": host}]},
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]
    bridge = LegacyConversationBridge(raw, host_texts={host})
    assert bridge.latest_user_text() == "change the price"
    assert bridge.transcript_text() == "user: change the price\nassistant: checking\nassistant: done"
    assert bridge.latest_exchange()[0]["content"] == "change the price"


def test_append_host_text_keeps_persisted_user_message_separate():
    raw = [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]
    bridge = LegacyConversationBridge(raw)
    bridge.append_host_text("reminder")
    assert len(raw) == 2
    assert raw[-1] == {"role": "user", "content": [{"type": "text", "text": "reminder"}]}


@dataclass
class Outcome:
    result_text: str
    is_error: bool = False


def test_close_open_tool_calls_uses_settled_outcome_or_interrupted_error():
    raw = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "a", "name": "write_a", "input": {}},
                {"type": "tool_use", "id": "b", "name": "write_b", "input": {}},
            ],
        }
    ]
    bridge = LegacyConversationBridge(raw)
    assert bridge.close_open_tool_calls({"a": Outcome("written")}) == 2
    assert raw[-1]["content"] == [
        {"type": "tool_result", "tool_use_id": "a", "content": "written", "is_error": False},
        {
            "type": "tool_result",
            "tool_use_id": "b",
            "content": INTERRUPTED_RESULT_TEXT,
            "is_error": True,
        },
    ]
