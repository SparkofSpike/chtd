# Run 1 — the first fine-tune, and what it does not prove

The adapter this directory describes exists on the training box at `/root/chtd_runs/chtd-1`. The
weights are not in this repository: 47 MB of safetensors is a binary this project has no reason to
carry. What is here is the record of the run.

```
adapter_model.safetensors  62af2dc7bc62b9c44e25be412660b5a27f801fdfe294489440b4567928bb3678
adapter_config.json        (tracked in this directory)
run.json                   (tracked in this directory)
```

## What ran

```
base model   Qwen3-4B-Instruct-2507, 4-bit NF4, LoRA on q/k/v/o_proj
base weights config.json 5beea1a4…  tokenizer_config.json a62ff0a2…  generation_config.json 835fffe3…
adapter      11,796,480 trainable of 3,739,343,360 logical parameters  (0.3155%)
dataset      a8c373b104ab442c4780c7f3f9914014cafe1243ec5ecfe7123f14211e525f9b  (1 entry)
samples      5fbda536f5bac0eb5b9a8db1301c3379a9ec17f52bb90fa11b7ce8fcac825b4b  (1 row)
track        a37b90fe31115418c17c8294622df7eac37511abd9283356ea464828682b39d5
steps        3   (1 sample × 3 epochs)
loss         4.7375 → 4.7375 → 2.9035   (trainer's mean 4.1261)
wall clock   8.9 s (training) / 16.6 s (including the base-weights fingerprint)
peak VRAM    17.79 GB
```

**On the parameter count.** `run.json` carries two figures and they mean different things.
`logical_parameters` is the model's real size, computed from its config. `packed_parameters_4bit`
is what torch reports after quantisation — 2,217,606,656 — because a bitsandbytes `Params4bit`
reports its *packed elements*, roughly two 4-bit values per element. An earlier version of this
record quoted the adapter's share against the packed figure and so reported 0.532% where the truth
is **0.3155%**. The two are both kept, labelled, because deleting one would hide the trap rather
than document it.

## Reproducing it

```sh
# rows, rebuilt from the ledger and the captured prefix
python -m distill.sft --dataset data/datasets/official-1 \
    --student-ledger data/agh_home/data/sessions.db \
    --prefix harness/prefix/capture-1.json \
    --out data/sft/official-1.jsonl

# the plan, then the run
python -m distill.finetune --plan data/sft/plan-chtd.json \
    --samples data/sft/official-1.jsonl \
    --base-model <the local Qwen3-4B-Instruct-2507 directory> --output <somewhere writable>
```

`--dry-run` renders and checks the samples without loading a model.

Note what this repository does *not* carry: the dataset, the samples and the ledger all live under
gitignored `data/`, because they are working artifacts. The two digests above (`dataset_sha256`,
`samples_sha256`) are what a rebuild is compared against, and the command above is what rebuilds
them.

## The run is protocol-reproducible and not bit-reproducible

This adapter is the **third** run of the same plan against the same samples. All three produced
different weights:

```
              run 1        run 2        run 3
last loss     2.9237       2.8909       2.9035
mean loss     4.1329       4.1219       4.1261
adapter sha   db98b9c9…    d2923b1b…    62af2dc7…
```

Same seed, same data, same code, same machine, different weights. This is expected — 4-bit
quantisation and bfloat16 accumulation on a GPU are not order-deterministic — and it decides what a
reproduction can promise: **a reader can rebuild the inputs and the protocol, and should expect the
numbers to land near, not on, the ones recorded here.** A claim that needs two runs to agree to the
byte is not available from this setup, and the write-up must not imply one. The run that is
described above is the third; the earlier two are kept in this table rather than overwritten,
because a reproducibility record showing only the successful attempt is not a record.

## What the loss numbers do and do not say

The first two steps report the same loss because the first step's learning rate is zero:
`warmup_ratio` of 0.03 over 3 steps puts the first update inside the warmup, so no weight moves and
the second loss is measured against unchanged weights. The third step is the only one that trained
anything, and its drop is that one update.

That is one optimizer step on one sample. It shows the mechanism works end to end; it cannot show
that anything was learned, and it is not evidence for the CHTD claim.

The trainer's reported mean (4.1261) is over the three logged values, i.e. 12.3784/3 = 4.12613…;
the ladder above is rounded at four decimals and a recomputation from it gives 4.1261 as well. The
mean is not a property of the training curve — with a zero learning rate on step 1 it includes a
step that trained nothing.

## What was verified, and how

**The adapter loads and serves behind the harness's student route.** The endpoint on the training
box runs with `--adapter /root/chtd_runs/chtd-1`, its `/health` reports
`"adapter": "/root/chtd_runs/chtd-1"`, and rollouts through Agnes Harness's `student` route have run
on the frozen track — three times, once per adapter:

```
session   agnes:local:local-dev:cli:workspace:6124e75de7f4ced5
session   agnes:local:local-dev:cli:workspace:1d0e57b36b6f6446
preset    chtd        track  a37b90fe…
answer    'Hello!HowcanIassistyoutoday?'   correct: False   (every time)
```

The answer is wrong, and it is wrong the same way the untuned student was wrong. That is the
expected result at this size and it is the honest headline: **three steps on one sample move the
weights and do not move the behaviour.**

**What the sample actually teaches.** The state is the conversation *up to* the student's divergent
decision, so the student's wrong turn is **not** in the input — that turn is the thing being
aligned, and the teacher's answer replaces it. The row is `[system, user, assistant(tool_call)]`
with the tool call as the label, so this run teaches the correct *first* move. Recovery-from-a-
mistake is a different training shape and needs a task where the student's error lands in the state
a later decision is made in.

## The prefix check that makes a sample trustworthy

`distill.finetune` refuses a sample whose prompt does not tokenize to a prefix of its rendered
conversation, because the alternative is a fine-tune trained on a conversation the endpoint never
produces. The check is not decorative: the system prompt is assembled from several packages and the
disclosed tools come from the loaded assembly, so a reconstruction would have been a guess.
`harness/prefix/capture-1.json` holds what the harness actually sent — 7279 characters of system
prompt and **22 disclosed tools**, where the frozen preset names two. That gap is the preset's own
warning made concrete, and it is recorded in `docs/finetune.md`.

It also holds an `assistant` and a `tool` turn, captured by putting a stand-in endpoint in the
student route and letting the harness run a real tool call. Those two turns are what
`tests/test_sft.py` checks the rebuild against, field for field — which is how a stripping bug in
the rebuild was found: tool output carries a trailing carriage return, and an earlier version of
the rebuild dropped it.

## Why the recipe was changed before the run

Two fields are not at their defaults, both because a measurement said so:

- `max_sequence_length: 8192` — the sample renders to **6715 tokens**. The default of 4096 would
  truncate, and the label is the last thing in the conversation, so truncation does not shorten the
  input; it deletes the answer.
- `gradient_accumulation: 1` — one sample with the default of 8 means the optimizer never steps.
  The run would report a loss and change no weights.

Both are properties of a one-entry dataset, not of the method. A larger dataset should raise the
accumulation back.
