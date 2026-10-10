"""The student endpoint must preserve a structured tool call in streamed replies."""

import json

from serve_openai import parse_assistant_output, stream_payload


def test_parse_qwen_tool_block_and_stream_complete_call() -> None:
    text = '<tool_call>\n{"name":"read","arguments":{"path":"config.json"}}\n</tool_call>'
    content, calls = parse_assistant_output(text)
    assert content is None
    assert calls[0]["function"]["name"] == "read"
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "config.json"}
    response = {
        "id": "x", "object": "chat.completion", "created": 1, "model": "qwen",
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content, "tool_calls": calls},
                     "finish_reason": "tool_calls"}],
    }
    body = stream_payload(response).decode("utf-8")
    parts = [json.loads(line[6:]) for line in body.splitlines()
             if line.startswith("data: {")]
    assert parts[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "read"
    assert parts[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert body.endswith("data: [DONE]\n\n")


def test_malformed_tool_block_is_kept_as_text() -> None:
    text = '<tool_call>{"name":"read","arguments":</tool_call>'
    content, calls = parse_assistant_output(text)
    assert content == text
    assert calls == []
