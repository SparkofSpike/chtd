# Run 1 — the first fine-tune, and what it does not prove

The adapter this directory describes exists on the training box at
`/root/chtd_runs/chtd-1`. The weights are not in this repository: 47 MB of safetensors is
a binary this project has no reason to carry. What is here is everything needed to check
that the run happened and to reproduce it.

```
adapter_model.safetensors  d2923b1b19b2c9d58fbcf346c60c6ed361662f19a19ab07b6a8eda480c77dac2
adapter_config.json        6e532aff08fc21168c6827f2f56aebafa48b6bc3d647d0065a2b16e31ff20408
run.json                   ae51e9b1fc7026bf97abfaa5ab87cc30fe6d3e5ec2ff38de0119be372c84ee9c
```

## What ran

```
base model   Qwen3-4B-Instruct-2507, 4-bit NF4, LoRA on q/k/v/o_proj
adapter      11,796,480 trainable of 2,217,606,656 total (0.532%)
dataset      a8c373b104ab442c4780c7f3f9914014cafe1243ec5ecfe7123f14211e525f9b  (1 entry)
samples      5fbda536f5bac0eb5b9a8db1301c3379a9ec17f52bb90fa11b7ce8fcac825b4b  (1 row)
track        a37b90fe31115418c17c8294622df7eac37511abd9283356ea464828682b39d5
steps        3   (1 sample × 3 epochs)
loss         4.7375 → 4.7375 → 2.8909   (mean 4.1219)
wall clock   8.9 s
peak VRAM    17.79 GB
```

Reproduce with the plan and the samples, from `E:\Coding\chtd`:

```sh
python -m distill.sft --dataset data/datasets/official-1 \
    --student-ledger data/agh_home/data/sessions.db \
    --prefix harness/prefix/capture-1.json \
    --out data/sft/official-1.jsonl
# then build_plan/write_plan for the plan, and:
python -m distill.finetune --plan data/sft/plan-chtd.json \
    --samples data/sft/official-1.jsonl \
    --base-model <the local Qwen3-4B-Instruct-2507 directory> --output <somewhere writable>
```

Both inputs are regenerated from the ledger and the captured prefix, so a reproduction
starts earlier than the fine-tune and the row's digest is the thing to compare.

## The run is protocol-reproducible and not bit-reproducible

This adapter is the **second** run of the same plan against the same samples. The first
produced a different one:

```
                first run                  second run
loss            4.7375 / 4.7375 / 2.9237  4.7375 / 4.7375 / 2.8909
mean loss       4.1329                     4.1219
adapter sha256  db98b9c9…                  d2923b1b…
```

Same seed, same data, same code, different weights. This is expected — 4-bit quantisation
and bfloat16 accumulation on a GPU are not order-deterministic — and it matters for what a
reproduction can promise: **a reader can rebuild the inputs and the protocol, and should
expect the numbers to land near, not on, the ones recorded here.** A claim that depends on
two runs producing identical bytes is not available from this setup, and the write-up must
not imply one.

The first run's numbers are kept above rather than overwritten, because a reproducibility
record that only shows the successful attempt is not a record.

## What the loss numbers do and do not say

The first two steps report the same loss because the first step's learning rate is zero:
`warmup_ratio` of 0.03 over 3 steps puts the first update inside the warmup, so no weight
moves and the second loss is measured against unchanged weights. The third step is the
only one that trained anything, and its drop is that one update.

That is one optimizer step on one sample. It shows the mechanism works end to end; it
cannot show that anything was learned, and it is not evidence for the CHTD claim.

## What was verified, and how

**The adapter loads and serves behind the harness's student route.** The endpoint on the
training box runs with `--adapter /root/chtd_runs/chtd-1`, its `/health` reports the adapter
path, and rollouts through Agnes Harness's `student` route have run on the frozen track —
twice, once against each adapter:

```
session   agnes:local:local-dev:cli:workspace:6124e75de7f4ced5   (first adapter)
session   agnes:local:local-dev:cli:workspace:1d0e57b36b6f6446   (this one)
preset    chtd        track  a37b90fe…
answer    'Hello!HowcanIassistyoutoday?'   correct: False   (both)
```

The answer is wrong, and it is wrong the same way the untuned student was wrong. That is
the expected result at this size and it is the honest headline: **three steps on one
sample move the weights and do not move the behaviour.**

## The prefix check that makes a sample trustworthy

`distill.finetune` refuses a sample whose prompt does not tokenize to a prefix of its
rendered conversation, because the alternative is a fine-tune trained on a conversation
the endpoint never produces. The check is not decorative: the system prompt is assembled
from several packages and the disclosed tools come from the loaded assembly, so a
reconstruction would have been a guess. `harness/prefix/capture-1.json` holds what the
harness actually sent — 7279 characters of system prompt and **22 disclosed tools**, where
the frozen preset names two. That gap is the preset's own warning made concrete, and it is
recorded in `docs/finetune.md`.

## Why the recipe was changed before the run

Two fields are not at their defaults, both because a measurement said so:

- `max_sequence_length: 8192` — the sample renders to **6715 tokens**. The default of 4096
  would truncate, and the label is the last thing in the conversation, so truncation does
  not shorten the input; it deletes the answer.
- `gradient_accumulation: 1` — one sample with the default of 8 means the optimizer never
  steps. The run would report a loss and change no weights.

Both are properties of a one-entry dataset, not of the method. A larger dataset should
raise the accumulation back.
