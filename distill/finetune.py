"""The fine-tune itself: the one module in this repository that needs a GPU.

`distill/train.py` produces the plan and refuses without an accelerator; this carries it out. The
two are separate because a plan has to be checkable anywhere — a reviewer can read a recipe and a
dataset digest on a laptop — while this cannot run anywhere but a machine with a GPU.

**The rendering is copied from the serving path on purpose.** A fine-tune learns the distribution
it is shown, and if training renders a conversation differently from the way the endpoint renders
it, the model is trained on a world it will never be asked about. So :func:`render_chat` is the
same code as `serve_openai.py`'s `render` — same key filtering, same content coercion from `null`
to `""`, same `tools` argument — and :func:`build_example` refuses a sample whose prefix does not
tokenize identically under the two renderings. That refusal is the mechanism; a comment saying
"keep these in sync" is not.

**Only the teacher's turn is supervised.** Every token before the label is masked to ``-100``. The
student's own wrong answer is part of the state it has to learn to recover from, so it stays in the
input and out of the loss — which is the difference between teaching a decision and teaching a
sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from distill.sft import samples_digest
from distill.train import LoraRecipe, TrainingPlan

__all__ = [
    "PrefixDrift",
    "build_example",
    "load_sample_lines",
    "main",
    "render_chat",
    "samples_digest_over",
    "train",
]


class PrefixDrift(RuntimeError):
    """Training and serving would show the model the same conversation differently.

    Raised rather than warned about, because a warning here produces a fine-tune that looks
    healthy and was trained against an input distribution the endpoint never produces. The numbers
    would be unremarkable and the result unattributable.
    """


def render_chat(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply the model's chat template the way the serving path does.

    Kept byte-for-byte compatible with `serve_openai.py`'s ``render``. Two details are load-bearing
    and neither is obvious: a ``null`` content becomes the empty string, because that is what the
    endpoint sends; and ``tools`` is passed into the template, because a model that is not told
    which tools it has cannot be taught to call one.
    """
    clean = []
    for message in messages:
        entry: dict[str, Any] = {"role": message["role"], "content": message.get("content") or ""}
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


def train(
    *,
    plan: TrainingPlan,
    row_lines: Sequence[str],
    base_model: str,
    output_dir: str | Path,
    max_length: int | None = None,
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
    )

    recipe: LoraRecipe = plan.recipe
    limit = max_length or recipe.max_sequence_length
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in row_lines]

    free, total = torch.cuda.mem_get_info()
    log(f"gpu free={free / 1024**3:.1f}GB total={total / 1024**3:.1f}GB")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log(f"tokenizer loaded, vocab={tokenizer.vocab_size}")

    examples, lengths = [], []
    for row in rows:
        example = build_example(tokenizer, row)
        if len(example["input_ids"]) > limit:
            raise PrefixDrift(
                f"sample {row.get('task_id')!r} renders to {len(example['input_ids'])} tokens, "
                f"over the recipe's max_sequence_length of {limit}. Truncating would cut the "
                "conversation at an arbitrary point and the label with it; raise the recipe's "
                "limit deliberately instead, which changes the experiment and re-freezes it."
            )
        examples.append(example)
        lengths.append(len(example["input_ids"]))
    log(f"rendered {len(examples)} samples: {min(lengths)}-{max(lengths)} tokens "
        f"(prefix drift check passed)")

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
    log(f"lora injected: trainable={trainable:,} / total={total_params:,} "
        f"({100 * trainable / total_params:.3f}%)")

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
        "total_parameters": total_params,
        "training_loss": result.training_loss,
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
            example = build_example(tokenizer, row)
            supervised = sum(1 for value in example["labels"] if value != -100)
            print(f"  task={row.get('task_id')} tokens={len(example['input_ids'])} "
                  f"supervised={supervised} limit={limit}")
        return 0

    if not args.output:
        parser.error("--output is required unless --dry-run is given")

    train(
        plan=plan,
        row_lines=row_lines,
        base_model=args.base_model,
        output_dir=args.output,
        max_length=args.max_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
