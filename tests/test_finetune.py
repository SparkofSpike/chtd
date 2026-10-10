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
