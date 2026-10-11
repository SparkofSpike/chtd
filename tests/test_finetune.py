"""The fine-tune's own guarantees, tested where a test can reach them.

The GPU steps are not tested here — no test suite in this repository loads a model, and this one
does not break that rule. What is tested is the part that decides whether a sample is *faithful*:
the label masking, and the refusal that stops a sample whose training rendering differs from the
serving rendering. Both are the mechanism behind claims the write-up makes, and a mechanism with no
test is a comment.
"""

from __future__ import annotations

import json

import pytest

from distill.finetune import (
    PrefixDrift,
    SampleTooLong,
    build_example,
    build_example_all_assistant,
    load_sample_lines,
    render_chat,
    render_examples,
    samples_digest_over,
)
from distill.sft import samples_digest


class FakeTokenizer:
    """A tokenizer that renders the chat template the way a real one does, and no more.

    One token per character, so a prefix relationship between two renderings is a string prefix —
    which is exactly the relationship the real check tests, and the only one that matters here.
    ``drift`` makes the full rendering stop being an extension of the prompt one, which is the
    condition under which the real code must refuse.
    """

    pad_token_id = 0
    eos_token = "<eos>"
    vocab_size = 1000

    def __init__(self, *, drift: bool = False, template_drops_prefix: bool = False) -> None:
        self.drift = drift
        self.template_drops_prefix = template_drops_prefix
        self.seen_tools: list | None = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        self.seen_tools = tools
        if self.template_drops_prefix and add_generation_prompt:
            # A template that renders the prompt differently from the way it renders the same
            # messages plus the answer. Nothing a caller can see except the tokens.
            return "<|assistant|>"
        parts = [f"<|{m['role']}|>{m.get('content') or ''}" for m in messages]
        text = "".join(parts)
        text += "<|assistant|>" if add_generation_prompt else "<|end|>"
        if self.drift and not add_generation_prompt:
            text = "XX" + text
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def _sample(messages=None, tools=None):
    return {
        "task_id": "t1",
        "prefix_digest": "d" * 64,
        "harness_prefix_digest": "e" * 64,
        "messages": messages
        or [
            {"role": "system", "content": "You are on a track."},
            {"role": "user", "content": [{"type": "text", "text": "Read notes.txt"}]},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "t0-1", "type": "function",
                             "function": {"name": "read", "arguments": '{"path":"notes.txt"}'}}]},
        ],
        "tools": tools if tools is not None else [{"name": "read"}],
    }


# ------------------------------------------------------------- the label mask


def test_only_the_teacher_turn_is_supervised() -> None:
    """The loss is over the teacher's decision and nothing before it.

    Note what the input does *not* contain: the state is the conversation up to the decision, so the
    student's own action at that decision is absent — it is the thing being aligned. The tail of
    `labels` equals the tail of `input_ids` because the label was appended last, which is why the
    stronger evidence for the boundary is the test below: it re-derives the boundary from the
    template rather than restating this implementation's own formula.
    """
    example = build_example(FakeTokenizer(), _sample())
    prompt_len = len([v for v in example["labels"] if v == -100])

    assert example["labels"][:prompt_len] == [-100] * prompt_len
    assert prompt_len < len(example["labels"])
    # Every supervised position is a real token, and the masked region is exactly the prompt.
    assert example["labels"][prompt_len:] == example["input_ids"][prompt_len:]
    assert example["attention_mask"] == [1] * len(example["input_ids"])
    # The masked prefix is the state's own rendering, and the label follows it.
    state = _sample()["messages"][:-1]
    assert prompt_len == len(
        [ord(char) for char in render_chat(
            FakeTokenizer(), state, _sample()["tools"], add_generation_prompt=True)]
    )
    assert "read" in json.dumps(_sample()["messages"][-1])


def test_the_prompt_ends_where_the_model_started_answering() -> None:
    """The boundary is the generation prompt, not the end of the last state message.

    Asserted against a rendering done by hand rather than against a second call to the code under
    test: the point is that the boundary lands where the *template* puts it, and a test that asked
    the same function for its own idea of the boundary could not tell.
    """
    tokenizer = FakeTokenizer()
    sample = _sample()
    example = build_example(tokenizer, sample)

    state = sample["messages"][:-1]
    prompt_only = tokenizer.apply_chat_template(
        [{**m, "content": "Read notes.txt" if m["role"] == "user" else m.get("content") or ""}
         for m in state],
        tokenize=False, add_generation_prompt=True, tools=sample["tools"]
    )
    expected_full = prompt_only + "<|end|>"

    assert example["input_ids"] == [ord(char) for char in expected_full]
    assert len([value for value in example["labels"] if value == -100]) == len(prompt_only)


# -------------------------------------------------------------- the refusal


def test_a_rendering_that_disagrees_is_refused() -> None:
    """The check that keeps training and serving from showing different conversations.

    Without it a fine-tune trains against an input distribution the endpoint never produces, and
    nothing downstream can detect it: the loss looks normal and the adapter loads.
    """
    with pytest.raises(PrefixDrift, match="renderings disagree"):
        build_example(FakeTokenizer(drift=True), _sample())


def test_a_template_that_renders_the_prompt_apart_is_refused() -> None:
    with pytest.raises(PrefixDrift, match="renderings disagree"):
        build_example(FakeTokenizer(template_drops_prefix=True), _sample())


def test_a_label_that_renders_to_nothing_is_refused() -> None:
    class Silent(FakeTokenizer):
        """A template that renders the prompt and the whole conversation identically.

        The label contributed no tokens, so the sample carries nothing to learn from — which is a
        different failure from the two renderings disagreeing, and is refused separately.
        """

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
            return "<|assistant|>"

    with pytest.raises(PrefixDrift, match="no tokens"):
        build_example(Silent(), _sample())


def test_a_sample_with_nothing_to_mask_is_refused() -> None:
    sample = _sample(messages=[{"role": "user", "content": "only a question"}])
    with pytest.raises(ValueError, match="the minimum"):
        build_example(FakeTokenizer(), sample)


# ------------------------------------------------- every-turn supervision


class NestedFakeTokenizer(FakeTokenizer):
    """A tokenizer whose renderings nest, which per-message supervision requires.

    `FakeTokenizer` ends a rendering with a single `<|end|>`, so appending a message moves that
    terminator instead of extending the text — the render of two messages is not a prefix of the
    render of three, and a per-message mask cannot be placed against it. The real Qwen template does
    not have this problem because every message carries its own terminator; this class renders the
    same way, so the nesting check passes for the reason it passes in production.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, tools=None):
        self.seen_tools = tools
        text = "".join(f"<|{m['role']}|>{m.get('content') or ''}<|end|>" for m in messages)
        return text + "<|assistant|>" if add_generation_prompt else text


def _multi_turn_sample():
    """A conversation with two assistant turns: a tool call, then a diagnosis.

    The shape this round's samples have, and the shape that exposed the problem: an arm trained where
    only the final turn carries the loss learned to answer and never to look.
    """
    return _sample(messages=[
        {"role": "system", "content": "You are on a track."},
        {"role": "user", "content": "Read the directory and say what is wrong."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t0-1", "type": "function",
                         "function": {"name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "content": "check.py table.py", "tool_call_id": "t0-1"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1-1", "type": "function",
                         "function": {"name": "read", "arguments": '{"path":"table.py"}'}}]},
        {"role": "tool", "content": "def blanked(table): ...", "tool_call_id": "t1-1"},
        {"role": "user", "content": "State the defect."},
        {"role": "assistant", "content": "It mutates the caller's rows."},
    ])


def test_every_assistant_turn_is_supervised() -> None:
    """Both assistant turns carry the loss, and the tool results do not.

    The measured reason this exists: the last-turn-only rule produced an arm that made **zero** tool
    calls on fifteen held-out tasks, because it had never been shown the decision to look. Its whole
    supervised experience was prose placed after a conversation somebody else had conducted.
    """
    example = build_example_all_assistant(NestedFakeTokenizer(), _multi_turn_sample())
    labels = example["labels"]
    spans = []
    start = None
    for index, value in enumerate(labels):
        if value != -100 and start is None:
            start = index
        elif value == -100 and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))

    # Two assistant turns, and a tool result between them that is masked: so at least two separate
    # supervised runs, not one contiguous tail.
    assert len(spans) >= 2, f"expected separate supervised turns, got {spans}"
    assert any(value == -100 for value in labels[spans[0][1]:spans[1][0]]), \
        "the tool result between two turns must stay masked"


def test_every_turn_supervises_strictly_more_than_the_last_turn_rule() -> None:
    """The two rules are not the same rule wearing two names."""
    tokenizer = NestedFakeTokenizer()
    sample = _multi_turn_sample()
    last_only = build_example(tokenizer, sample)
    every = build_example_all_assistant(tokenizer, sample)

    def supervised(example):
        return sum(1 for value in example["labels"] if value != -100)

    assert supervised(every) > supervised(last_only)
    # and the input is identical: only the mask differs, so a comparison between the two is a
    # comparison of the supervision rule and not of the conversation.
    assert every["input_ids"] == last_only["input_ids"]
    assert every["attention_mask"] == last_only["attention_mask"]


def test_the_two_rules_agree_on_a_single_turn_sample() -> None:
    """On a one-turn sample the two rules supervise the same content, differing only at the boundary.

    `build_example` starts where the generation prompt ends, because that is where the endpoint starts
    asking the model to write. `build_example_all_assistant` starts at the message boundary, so it also
    covers the assistant header token. Both supervise the answer; the difference is a few tokens of
    framing, not a difference in what is learned — which is why the multi-turn case above is the one
    that carries the argument.
    """
    tokenizer = NestedFakeTokenizer()
    sample = _sample()
    last_only = build_example(tokenizer, sample)
    every = build_example_all_assistant(tokenizer, sample)

    def supervised(example):
        return sum(1 for value in example["labels"] if value != -100)

    assert supervised(last_only) > 0 and supervised(every) > 0
    # Both reach the end of the conversation: the answer itself is supervised either way.
    assert last_only["labels"][-1] != -100
    assert every["labels"][-1] != -100
    # and the all-turns rule covers at least as much, since it starts a boundary earlier.
    assert supervised(every) >= supervised(last_only)


def test_the_nesting_check_refuses_a_template_that_cannot_place_a_mask() -> None:
    """The plain FakeTokenizer terminates a rendering once, so its renderings do not nest.

    A per-message mask cannot be located against it, and the code says so instead of placing the mask
    against the wrong offsets — which would train on tokens it did not mean to and report a normal
    loss while doing it.
    """
    with pytest.raises(PrefixDrift, match="does not extend"):
        build_example_all_assistant(FakeTokenizer(), _multi_turn_sample())


def test_render_examples_refuses_an_unknown_supervision_rule() -> None:
    """`supervise` is a choice between two named rules; a typo must not silently mean the old one."""
    with pytest.raises(ValueError, match="supervise must be"):
        render_examples(FakeTokenizer(), [_sample()], limit=10_000, supervise="assitant",
                        log=lambda *_: None)


def test_render_examples_reports_the_supervised_count() -> None:
    """The dry run prints this number, and it is how the change from one turn to every turn was
    checked before a GPU-second was spent."""
    lines = []
    render_examples(NestedFakeTokenizer(), [_multi_turn_sample()], limit=10_000,
                    supervise="assistant", log=lines.append)
    assert lines and "supervise='assistant'" in lines[0], lines


# ----------------------------------------------------------------- rendering


def test_render_passes_the_tools_into_the_template() -> None:
    """A model not told which tools it has cannot be taught to call one."""
    tokenizer = FakeTokenizer()
    tools = [{"name": "read"}]
    render_chat(tokenizer, _sample()["messages"], tools, add_generation_prompt=True)
    assert tokenizer.seen_tools == tools


def test_render_flattens_openai_text_blocks_for_qwen() -> None:
    rendered = render_chat(
        FakeTokenizer(),
        [{"role": "user", "content": [{"type": "text", "text": "Read notes.txt"}]}],
        None,
        add_generation_prompt=True,
    )
    assert rendered == "<|user|>Read notes.txt<|assistant|>"


def test_render_rejects_image_blocks_on_text_only_route() -> None:
    with pytest.raises(ValueError, match="text content blocks only"):
        render_chat(FakeTokenizer(),
                    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
                    None, add_generation_prompt=True)


def test_render_turns_null_content_into_the_empty_string() -> None:
    """The endpoint sends `""`, so training must too. A `None` reaching a template renders as the
    literal text 'None' in some of them, which is a token sequence the model never saw."""
    text = render_chat(
        FakeTokenizer(),
        [{"role": "assistant", "content": None}],
        None,
        add_generation_prompt=False,
    )
    assert "None" not in text
    assert text.startswith("<|assistant|>")


# ------------------------------------------------------------------- samples


def test_sample_lines_are_what_gets_digested(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [json.dumps({"task_id": "a"}, sort_keys=True), json.dumps({"task_id": "b"}, sort_keys=True)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    lines = load_sample_lines(path)
    assert lines == rows
    # One function, so a run's recorded digest and a reader's rebuild agree.
    assert samples_digest_over(lines) == samples_digest(lines)


def test_a_digest_moves_when_a_row_does() -> None:
    a = samples_digest(['{"task_id":"a"}'])
    b = samples_digest(['{"task_id":"b"}'])
    assert a != b
    assert samples_digest(['{"task_id":"a"}']) == a


def test_an_empty_sample_file_is_refused(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        load_sample_lines(path)


def test_a_missing_sample_file_is_named(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no samples at"):
        load_sample_lines(tmp_path / "absent.jsonl")


def test_a_sample_longer_than_the_recipe_is_refused() -> None:
    """Refused rather than truncated: the label is last, so truncation deletes the answer.

    A `SampleTooLong` and not a `PrefixDrift` — the two name different fixes, and reporting one as
    the other sends a reader looking in the wrong place. Both refusals happen while rendering, which
    is why `render_examples` is a function of its own: a check that can only run on the training box
    is a check that mostly goes unexercised.
    """
    tokenizer = FakeTokenizer()
    with pytest.raises(SampleTooLong, match="over the recipe"):
        render_examples(tokenizer, [_sample()], limit=4, log=lambda *_: None)


def test_a_sample_inside_the_limit_renders() -> None:
    """The other half of the same check: the limit is a refusal, not a policy of refusing."""
    examples, lengths = render_examples(
        FakeTokenizer(), [_sample()], limit=10_000, log=lambda *_: None
    )
    assert len(examples) == 1
    assert lengths == [len(examples[0]["input_ids"])]
    assert any(value == -100 for value in examples[0]["labels"])
    assert any(value != -100 for value in examples[0]["labels"])
