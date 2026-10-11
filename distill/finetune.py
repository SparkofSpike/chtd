"""The fine-tune itself: the one module in this repository that needs a GPU.

`distill/train.py` produces the plan and refuses without an accelerator; this carries it out. The
two are separate because a plan has to be checkable anywhere — a reviewer can read a recipe and a
dataset digest on a laptop — while this cannot run anywhere but a machine with a GPU.

**What this refuses, and what that is worth.** The weights come out of a plan the run may not
alter, and a sample whose prompt does not tokenize to a prefix of its rendered conversation is
refused (:class:`PrefixDrift`), as is one that needs more tokens than the recipe allows
(:class:`SampleTooLong`). Those catch a template that renders one conversation two ways, and a
budget that was set too small for its data. They do **not** establish that :func:`render_chat`
still agrees with whatever serves the model — that is a separate program, and keeping the two in
step is checked on the box, by serving a request and comparing, not by anything here.

**Only the teacher's turn is supervised.** Every token before the label is masked to ``-100``, so
the loss is over the decision and nothing else — the difference between teaching a decision and
teaching a sentence.

**What the input contains, stated exactly.** The state is the conversation *up to* the decision the
dataset names, so the student's own action at that decision is not in it; that action is the thing
being aligned, and the teacher's answer replaces it. For a one-step task this means the sample
teaches the correct *first* move, which is what the shipped dataset does. "Recovering from its own
mistake" is a different training shape and needs a task where the student's error lands in the
state a later decision is made in — a multi-step task, which this project does not yet have. The
distinction is easy to lose in a summary and it changes what a result can be read as.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from distill.sft import samples_digest
from distill.train import LoraRecipe, TrainingPlan

__all__ = [
    "PrefixDrift",
    "SampleTooLong",
    "build_example",
    "load_sample_lines",
    "main",
    "render_chat",
    "render_examples",
    "samples_digest_over",
    "train",
]


class PrefixDrift(RuntimeError):
    """Training and serving would show the model the same conversation differently.

    Raised rather than warned about, because a warning here produces a fine-tune that looks
    healthy and was trained against an input distribution the endpoint never produces. The numbers
    would be unremarkable and the result unattributable.

    **What this checks, and what it does not.** Both renderings come from the same function and the
    same tokenizer, so this catches a *template* whose prompt rendering is not a prefix of its full
    rendering — a real class of bug, and the one a chat-template upgrade tends to produce. It does
    **not** prove that :func:`render_chat` still matches whatever serves the model: nothing here
    can, because the serving path is a separate program. Keeping the two in step is a separate
    obligation, discharged on the box by serving a request and comparing, not by this check.
    """


class SampleTooLong(RuntimeError):
    """A sample needs more tokens than the recipe allows.

    Its own exception rather than a :class:`PrefixDrift` because the two name different fixes: a
    drifted render means reconciling a template, an over-long sample means deciding to raise the
    recipe's limit — which changes the experiment and re-freezes it. Reporting one as the other
    sends a reader looking in the wrong place.
    """


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply the model's chat template the way the serving path does.

    Shared by training and `serve_openai.py`. AGH sends text as OpenAI content
    blocks; Qwen3's text tokenizer silently drops those blocks when passed
    directly to its template. Flatten text blocks before rendering. Reject
    non-text blocks because this route has no image input capability.
    """
    def text_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if not isinstance(value, list):
            raise ValueError("unsupported message content")
        parts = []
        for block in value:
            if not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str):
                raise ValueError("this model route accepts text content blocks only")
            parts.append(block["text"])
        return "\n".join(parts)

    clean = []
    for message in messages:
        entry: dict[str, Any] = {"role": message["role"], "content": text_content(message.get("content"))}
        if message.get("tool_call_id"):
            entry["tool_call_id"] = message["tool_call_id"]
        if message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]
        if message.get("name"):
            entry["name"] = message["name"]
        clean.append(entry)

    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if tools:
        kwargs["tools"] = list(tools)
    try:
        return tokenizer.apply_chat_template(clean, **kwargs)
    except TypeError:
        # An older template that does not accept tools still renders the messages. Same fallback as
        # the serving path, for the same reason: a template predating tool support should degrade to
        # a working render rather than fail a run.
        return tokenizer.apply_chat_template(
            clean, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def message_bounds(tokenizer: Any, messages: Sequence[Mapping[str, Any]],
                   tools: Sequence[Mapping[str, Any]] | None) -> list[int]:
    """The token offset each message ends at, by rendering successively longer prefixes.

    Needed to supervise more than the last turn. Rendering the whole conversation gives one run of
    tokens with no map back to the messages that produced it, and guessing where a turn starts by
    searching for its text is wrong the moment a tool result quotes that text.

    The prefixes are required to nest: each render must extend the previous one token-for-token. If
    it does not, a message boundary cannot be located this way, and the caller is told rather than
    handed an offset that silently lands mid-token.
    """
    bounds = [0]
    previous: list[int] = []
    for index in range(1, len(messages) + 1):
        text = render_chat(tokenizer, messages[:index], tools, add_generation_prompt=False)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids[: len(previous)] != previous:
            diverged = next((i for i, (a, b) in enumerate(zip(previous, ids)) if a != b),
                            min(len(previous), len(ids)))
            raise PrefixDrift(
                f"message {index} does not extend the conversation before it: the renderings "
                f"disagree at token {diverged}, so a per-message supervision mask cannot be "
                f"placed reliably"
            )
        previous = ids
        bounds.append(len(ids))
    return bounds


def build_example_all_assistant(tokenizer: Any, sample: Mapping[str, Any]) -> dict[str, list[int]]:
    """One sample with **every** assistant turn supervised, not just the last one.

    Why this exists, measured 2026-10-11. Supervising only the final turn is what this repository's
design calls for: the label is the teacher's decision at a divergence, and the turns before it are the
state. An arm trained that way learned to *answer* — and not to look. On fifteen held-out tasks it
made **zero** tool calls across all fifteen (the base made three on each), and answered every task
with the same sentence about no files being listed, because the only behaviour it had ever been shown
was prose placed after a conversation somebody else had already conducted.

    The turns before the divergence are the teacher's own decisions too — the same teacher, the same
    track, the same fixtures. Supervising them teaches the trajectory rather than its last word, and
    keeps what the method is for: every supervised turn is the teacher's, at a state the run actually
    reached.

    What is given up, stated plainly: this is no longer "only the divergence is learned". It is
    demonstration learning over a teacher trajectory that happens to contain a divergence. The
    comparison between arms still holds, because both arms' data is built the same way.
    """
    messages = list(sample["messages"])
    tools = sample.get("tools")
    if len(messages) < 2:
        raise ValueError(f"sample {sample.get('task_id')!r} has {len(messages)} messages; a state "
                         "and a label are the minimum")

    full_text = render_chat(tokenizer, messages, tools, add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    bounds = message_bounds(tokenizer, messages, tools)
    if bounds[-1] != len(full_ids):
        raise PrefixDrift(
            f"the final message boundary is {bounds[-1]} but the conversation renders to "
            f"{len(full_ids)} tokens; the mask would be placed against the wrong text"
        )

    labels = [-100] * len(full_ids)
    supervised_turns = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        start, end = bounds[index], bounds[index + 1]
        if end <= start:
            continue
        labels[start:end] = full_ids[start:end]
        supervised_turns += 1
    if not supervised_turns:
        raise ValueError(f"sample {sample.get('task_id')!r} has no assistant turn to supervise")

    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_example(tokenizer: Any, sample: Mapping[str, Any]) -> dict[str, list[int]]:
    """One sample as ``input_ids``, ``attention_mask`` and ``labels``.

    The prompt is rendered with ``add_generation_prompt=True`` and the whole conversation with
    ``False``. If the first does not tokenize to a prefix of the second, the two renderings disagree
    about the same conversation and the sample is refused: a trainer fed it would supervise tokens
    the endpoint never asks for, or ask for tokens the trainer never supervised.
    """
    messages = list(sample["messages"])
    tools = sample.get("tools")
    if len(messages) < 2:
        raise ValueError(f"sample {sample.get('task_id')!r} has {len(messages)} messages; a state "
                         "and a label are the minimum, and there is nothing to mask without both")

    prompt_text = render_chat(tokenizer, messages[:-1], tools, add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, tools, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        diverged = next(
            (i for i, (a, b) in enumerate(zip(prompt_ids, full_ids)) if a != b),
            min(len(prompt_ids), len(full_ids)),
        )
        raise PrefixDrift(
            f"the serving and training renderings disagree at token {diverged} "
            f"({len(prompt_ids)} prompt tokens vs {len(full_ids)} rendered): training would "
            "supervise a conversation the endpoint does not produce. The chat template, the tools "
            "argument or the serving path's render has changed; reconcile them before training."
        )
    if len(full_ids) <= len(prompt_ids):
        raise PrefixDrift(
            "the label rendered to no tokens, so the sample carries nothing to learn from"
        )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }


def load_sample_lines(path: str | Path) -> list[str]:
    """The rows as they sit in the file, one JSON object per line.

    Returned as text rather than as parsed objects because the digest has to be over what was
    written. Re-serialising a parsed object can differ from the original by a space or a float
    spelling, and a run whose recorded digest matches no file a reader can build is a run nobody
    can check.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no samples at {source}")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(
            f"{source} holds no samples; a fine-tune on nothing produces weights "
            "indistinguishable from the base model but reported as a run"
        )
    return lines


def _collator(pad_token_id: int):
    """Right-pad to the longest row in the batch, and never pad a label into the loss."""

    def collate(rows: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(row["input_ids"]) for row in rows)
        input_ids, attention, labels = [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_ids.append(list(row["input_ids"]) + [pad_token_id] * pad)
            attention.append(list(row["attention_mask"]) + [0] * pad)
            labels.append(list(row["labels"]) + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def samples_digest_over(row_lines: Sequence[str]) -> str:
    """The digest of the rows as they were trained, computed the one way there is.

    Delegates to :func:`distill.sft.samples_digest`, which hashes the JSONL that was written. A
    second implementation here would give the same rows two digests, and the failure mode is a run
    whose recorded ``samples_sha256`` matches no file a reader can build.
    """
    return samples_digest(row_lines)


def _base_weights_fingerprint(base_model: str) -> dict[str, Any]:
    """What the base weights were, in a form that can be compared later.

    ``base_model`` is a *path*, and on this box it ends in ``snapshots/master`` — a mutable ref. A
    record that names only the path cannot tell a later reader whether the weights changed under it,
    which is the one thing a reproduction needs to know about the base.

    Hashing the weights themselves would mean reading several gigabytes; hashing the files that
    *describe* them is cheap and discriminating: two different models do not ship the same config and
    tokenizer. A model whose weights were swapped while its config and tokenizer stayed identical is
    not a case this distinguishes, and saying so here is better than implying it does.
    """
    root = Path(base_model)
    digest = hashlib.sha256()
    parts: dict[str, str | None] = {}
    for name in ("config.json", "tokenizer_config.json", "generation_config.json"):
        candidate = root / name
        if candidate.exists():
            value = hashlib.sha256(candidate.read_bytes()).hexdigest()
            parts[name] = value
            digest.update(f"{name}\0{value}\n".encode())
        else:
            parts[name] = None
    return {"files": parts, "digest": digest.hexdigest()}


def _logical_parameter_count(config: Any) -> int:
    """The base model's parameters, counted from its config rather than from its tensors.

    Needed because a quantised model cannot report this. After ``load_in_4bit`` the Linears are
    packed two values per element, so summing ``p.numel()`` understates the model — measured on the
    student: 2,205,810,176 reported against 4,022,468,096 logical. A share computed against the
    packed figure is inflated by the same factor, which is the sort of number a write-up quotes
    without noticing.

    **`q_proj` is not square, and assuming it was is a mistake this function made.** Qwen3 projects
    the query to `num_attention_heads * head_dim`, which on this model is 4096 against a hidden size
    of 2560, and `o_proj` maps back from 4096. Counting both as `hidden * hidden` understated the
    model by 283,124,736 parameters and every share derived from it. The shapes are now taken from
    the checkpoint itself, which is the authority: this function reproduces the 4,022,468,096 that
    `count_from_weights.py` reads out of the safetensors headers exactly.
    """
    hidden = config.hidden_size
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", heads)
    head_dim = getattr(config, "head_dim", hidden // heads)
    intermediate = config.intermediate_size

    query_out = heads * head_dim
    key_value_out = kv_heads * head_dim

    attention = hidden * query_out + hidden * key_value_out * 2 + query_out * hidden
    mlp = hidden * intermediate * 3
    # Qwen3 normalises the query and key vectors before the attention itself, which is two more
    # vectors of `head_dim` per layer. Leaving them out put this function 9,216 parameters below the
    # checkpoint, which is how they were noticed.
    qk_norm = 2 * head_dim
    return (
        layers * (attention + mlp + hidden * 2 + qk_norm)
        + config.vocab_size * hidden
        + hidden
    )


def render_examples(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    supervise: str = "last",
    log=print,
) -> tuple[list[dict[str, list[int]]], list[int]]:
    """Render every sample, refusing a drifted one or one over the recipe's limit.

    Split out of :func:`train` so both refusals can be tested without a GPU: they happen while
    rendering, before any model is loaded, and a check that can only be exercised on the training box
    is a check that mostly goes unexercised.

    A sample over ``limit`` is refused rather than truncated. The label is the last thing in the
    conversation, so truncation does not shorten the input — it deletes the answer, and the run would
    report a healthy loss on a label that was not there.

    ``supervise`` chooses which assistant turns carry the loss: ``last`` for the repository's
    original rule (only the divergence), ``assistant`` for every turn in the conversation. See
    :func:`build_example_all_assistant` for why the second one exists.
    """
    if supervise not in ("last", "assistant"):
        raise ValueError(f"supervise must be 'last' or 'assistant', not {supervise!r}")
    build = build_example_all_assistant if supervise == "assistant" else build_example
    examples: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    supervised: list[int] = []
    for row in rows:
        example = build(tokenizer, row)
        if len(example["input_ids"]) > limit:
            raise SampleTooLong(
                f"sample {row.get('task_id')!r} renders to {len(example['input_ids'])} tokens, "
                f"over the recipe's max_sequence_length of {limit}. Truncating would cut the "
                "conversation at an arbitrary point and the label with it; raise the recipe's "
                "limit deliberately instead, which changes the experiment and re-freezes it."
            )
        examples.append(example)
        lengths.append(len(example["input_ids"]))
        supervised.append(sum(1 for value in example["labels"] if value != -100))
    log(f"rendered {len(examples)} samples: {min(lengths)}-{max(lengths)} tokens, "
        f"{min(supervised)}-{max(supervised)} supervised, supervise={supervise!r} "
        f"(prefix drift check passed)")
    return examples, lengths


def train(
    *,
    plan: TrainingPlan,
    row_lines: Sequence[str],
    base_model: str,
    output_dir: str | Path,
    max_length: int | None = None,
    supervise: str = "last",
    log=print,
) -> dict[str, Any]:
    """Run the fine-tune the plan describes, and return the record of what ran.

    The recipe comes from the plan and is not overridden here: a run that quietly changed a
    hyperparameter would produce weights that no recorded plan describes, which is the one thing
    the plan exists to prevent.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    recipe: LoraRecipe = plan.recipe
    limit = max_length or recipe.max_sequence_length
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in row_lines]

    # Trainer seeds its own loop only after the model has been constructed. Seed
    # here as well so both arms start from the same LoRA initialization.
    set_seed(recipe.seed)

    free, total = torch.cuda.mem_get_info()
    log(f"gpu free={free / 1024**3:.1f}GB total={total / 1024**3:.1f}GB")
    started = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"tokenizer loaded, vocab={tokenizer.vocab_size}")

    examples, lengths = render_examples(tokenizer, rows, limit=limit, supervise=supervise, log=log)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    log("base model loaded in 4-bit")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=recipe.rank,
            lora_alpha=recipe.alpha,
            lora_dropout=recipe.dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(recipe.target_modules),
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    # The 4-bit count is **not** the model's parameter count. A bitsandbytes `Params4bit` reports
    # its packed elements — about two 4-bit values per element — so sizing the adapter's share
    # against it roughly doubles the share. Measured on the box: 3,739,343,360 logical parameters
    # are reported as 2,205,810,176 packed elements. The logical count is the one a reader means, so
    # it is computed from the model's own config rather than read off a quantised tensor.
    logical_params = _logical_parameter_count(model.config)
    log(f"lora injected: trainable={trainable:,} / logical={logical_params:,} "
        f"({100 * trainable / logical_params:.4f}%)  [4-bit packed count={total_params:,}]")

    dataset = Dataset.from_list(examples)
    arguments = TrainingArguments(
        output_dir=str(target),
        per_device_train_batch_size=recipe.batch_size,
        gradient_accumulation_steps=recipe.gradient_accumulation,
        learning_rate=recipe.learning_rate,
        num_train_epochs=recipe.epochs,
        warmup_ratio=recipe.warmup_ratio,
        seed=recipe.seed,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="paged_adamw_8bit",
        report_to=[],
        # /dev/shm is 64 MB on this host, measured; a worker process dies on it.
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        data_collator=_collator(tokenizer.pad_token_id),
    )
    log("training")
    result = trainer.train()
    log(f"training done: loss={result.training_loss:.4f} over {result.global_step} steps")

    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    saved = sorted(path.name for path in target.glob("*"))
    log(f"saved adapter: {saved}")

    record = {
        "arm": plan.arm,
        "base_model": plan.base_model,
        "base_model_revision": plan.base_model_revision,
        "base_weights_fingerprint": _base_weights_fingerprint(base_model),
        "teacher_model": plan.teacher_model,
        "track_sha256": plan.track_sha256,
        "dataset_sha256": plan.dataset_sha256,
        "samples_sha256": samples_digest_over(row_lines),
        "entries": plan.entries,
        "samples_trained": len(examples),
        "token_lengths": {"min": min(lengths), "max": max(lengths)},
        "recipe": json.loads(recipe.fingerprint()),
        "max_sequence_length_used": limit,
        "trainable_parameters": trainable,
        "logical_parameters": logical_params,
        "packed_parameters_4bit": total_params,
        "trainable_share_of_logical_pct": round(100 * trainable / logical_params, 4),
        "training_loss": result.training_loss,
        "wall_clock_seconds": round(time.time() - started, 1),
        "global_step": result.global_step,
        "epochs": recipe.epochs,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "adapter_files": saved,
    }
    (target / "run.json").write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log(f"wrote {target / 'run.json'}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LoRA fine-tune the plan describes.")
    parser.add_argument("--plan", required=True, help="the TrainingPlan JSON")
    parser.add_argument("--samples", required=True, help="the JSONL distill.sft wrote")
    parser.add_argument("--base-model", required=True, help="a local path or a hub id")
    parser.add_argument("--output", required=False, default=None, help="where the adapter is written")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="override the recipe's max_sequence_length (a deliberate change, recorded)",
    )
    parser.add_argument(
        "--supervise",
        choices=("last", "assistant"),
        default="last",
        help=("which assistant turns carry the loss: 'last' is the original rule (only the "
              "divergence), 'assistant' supervises every turn in the conversation"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render and check the samples without loading the model",
    )
    args = parser.parse_args(argv)

    plan = TrainingPlan.from_json(Path(args.plan).read_text(encoding="utf-8"))
    row_lines = load_sample_lines(args.samples)
    print(f"plan : arm={plan.arm} base={plan.base_model} entries={plan.entries}")
    print(f"rows : {len(row_lines)}  digest={samples_digest_over(row_lines)[:24]}")

    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        limit = args.max_length or plan.recipe.max_sequence_length
        for line in row_lines:
            row = json.loads(line)
            build_one = (build_example_all_assistant if args.supervise == "assistant"
                         else build_example)
            example = build_one(tokenizer, row)
            supervised = sum(1 for value in example["labels"] if value != -100)
            turns = sum(1 for m in row["messages"] if m.get("role") == "assistant")
            print(f"  task={row.get('task_id')} tokens={len(example['input_ids'])} "
                  f"supervised={supervised} assistant_turns={turns} limit={limit}")
        return 0

    if not args.output:
        parser.error("--output is required unless --dry-run is given")

    train(
        plan=plan,
        row_lines=row_lines,
        base_model=args.base_model,
        output_dir=args.output,
        max_length=args.max_length,
        supervise=args.supervise,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
