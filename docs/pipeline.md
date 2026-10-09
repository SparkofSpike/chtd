# The pipeline

What each module does, what it refuses to do, and how to run it. The method is in
[design.md](design.md); this is the code that implements it.

The whole pipeline runs without a model endpoint. The model enters at exactly one place — the
runner that drives a session — and everything downstream reads ledgers. That is deliberate: a
pipeline that could only be exercised against a live endpoint is a pipeline nobody can check, and
the dataset digest a write-up asks a reader to reproduce would be reproducible only by whoever has
the account.

## The modules

```
harness/ledger.py        read a session ledger, read-only, enforced by SQLite rather than promised
harness/trajectory.py    a ledger as a sequence of decision points, each with a prefix digest
harness/rollout.py       drive sessions, collect trajectories, record what each run was
distill/divergence.py    pair two trajectories by state; report what differed and what did not meet
distill/gate.py          the signal-quality gate: admit only outcome-validated divergences
distill/dataset.py       rows + manifest + refusals; refuses to mix tracks
distill/sft.py           rows + the captured harness prefix -> the JSONL a trainer eats
distill/train.py         the LoRA plan, and the accelerator check that stops it without a GPU
distill/finetune.py      the run itself, and the prefix-drift refusal that keeps it honest
distill/pipeline.py      the command that runs the above over recorded rollouts
eval/metrics.py          one run's numbers, from the trajectory rather than from a self-report
eval/arms.py             the baseline-vs-CHTD comparison, after refusing mismatched conditions
eval/stripping.py        the ladder, and the verdict when a gain does not survive it
eval/distractors.py      co-movers, and the reason a kept one must have
eval/report.py           the write-up, generated from the objects above
```

## Running it

Collect rollouts (needs a built harness; the runner is the one model-facing piece):

```
python -c "from harness.rollout import ..."   # see harness/rollout.py — CliRunner
```

Build a dataset from what was collected:

```
python -m distill.pipeline dataset \
    --student-ledger data/rollouts/student/t1/sessions.db --student-session <key> \
    --teacher <ledger>:<session> --teacher <ledger>:<session> \
    --task t1 --track <track-sha256> --out data/datasets/round1 \
    --outcomes data/outcomes.jsonl
```

`--outcomes` is a JSONL of `{session_key, prefix_digest, outcome}`; `--verdicts` is a JSONL of
`{prefix_digest, admitted}` for judgements a human or a verifier recorded. At least one is required —
the command refuses to build an ungated dataset, because a gate with no validators admits everything
and admitting everything is the behaviour the gate exists to prevent.

Turn the dataset into training rows, and run the fine-tune:

```
# rows: the state from the ledger, the teacher's turn as the label
python -m distill.sft \
    --dataset data/datasets/official-1 \
    --student-ledger data/agh_home/data/sessions.db \
    --prefix harness/prefix/capture-1.json \
    --out data/sft/official-1.jsonl

# the recipe, the dataset digest and the base model, frozen together
python -c "from distill.train import ..."   # build_plan + write_plan; see distill/train.py

# the run itself, on a machine with a GPU
python -m distill.finetune \
    --plan data/sft/plan-chtd.json --samples data/sft/official-1.jsonl \
    --base-model <local Qwen3-4B-Instruct-2507> --output <adapter dir>
```

`distill.finetune` refuses a sample whose prompt does not tokenize to a prefix of its rendering
(`PrefixDrift`) and refuses a sample longer than the recipe's `max_sequence_length` rather than
truncate it — the label is the last thing in the conversation, so truncation removes the answer.
`--dry-run` renders and checks without loading the model.

Then compare, strip and report with `eval/arms.py`, `eval/stripping.py`, `eval/distractors.py` and
`eval/report.py`. `tests/test_pipeline.py` runs the first half of that chain end to end over real
ledgers.

## Four things the code refuses, and why

Each is a refusal rather than a warning, because the failure it prevents is one that nothing
downstream could detect afterwards.

**A ledger opened writable.** `harness/ledger.py` opens with SQLite's `mode=ro` URI and sets
`PRAGMA query_only`, and the test suite asserts that a write through the reader fails. The ledger is
the record the experiment rests on; a bug that wrote to one would not announce itself.

**A dataset from two tracks.** `distill/dataset.py` compares the `track_sha256` on every rollout
record and raises `MixedTracksError` if they disagree. Two tracks are two experiments, and a dataset
that merged them would measure something nobody chose.

**A comparison of arms that were not run identically.** `eval/arms.py` refuses when the tracks, task
sets, step budgets or LoRA recipes differ. The design's claim is that the *only* difference between
the arms is where the teacher's data came from; a comparison where that is not true is
unattributable, and the loss is silent — the numbers look fine.

**A kept co-mover without a stated reason.** `eval/distractors.py` strips a metric that moved with
the capability unless somebody wrote down why it is causal, and `eval/report.py` prints that reason
next to the metric. Without it a reader cannot tell the number from the capability it moved with.

## Two things that were measured rather than assumed

**The prefix digest covers the conversation, not the ledger.** A same-prefix fork writes rows of its
own at the boundary — a child's `session/start`, its `budget.state`, its `inbox`. A digest that
counted them would give the teacher a different prefix from the student *in the very state the fork
exists to reproduce*, and nothing would ever align. `harness/trajectory.py` excludes a named list of
bookkeeping rows; anything not on that list counts as state, so a type nobody has seen before
produces a visible failure to align rather than a quiet alignment of two different states.

**`BEGIN IMMEDIATE` does not prove a connection is read-only.** An earlier version of the ledger
reader used it as a writability probe. It turns out a read-only connection accepts that lock, so the
probe refused every healthy ledger while SQLite was in fact refusing writes through the URI. The
reader no longer probes; `tests/test_ledger.py` does, by attempting a write and expecting it to
fail.

## What is not built yet

- **The training loop is built, and one run has happened.** `distill/finetune.py` carries out the
  plan `distill/train.py` writes: 4-bit load, LoRA injection, the sample rendering with its
  prefix-drift refusal, and the adapter on disk. Its first real run is recorded in
  `runs/chtd-1/` and described in [finetune.md](finetune.md). What is *not* built is a dataset large
  enough for the run to mean anything: one entry, 20 supervised tokens, one effective step.
- **The runner has driven rollouts against live models, and the training-side gap is the ladder.**
  Both arms have run the frozen track against real endpoints — the teacher at the organisers'
  Agnes endpoint and the student at the box's local one — and the teacher's session was forked into
  the student's state to produce the divergence the gate admitted. What the runner still cannot do
  is produce a **stripping-ladder rung**: it takes a route and a model but no preset and no tool or
  turn limit, so `eval/stripping.py` assesses a ladder somebody else's run would have to supply.

  Building needs `AGNES_NODE_HEADERS` pointed at a Node header cache (see *Prerequisites* below);
  without it the build stops in its own native step.
- **One DAgger round.** `distill/pipeline.py` collects one round. Aggregation across rounds is
  future work, and the limitation is carried in every report.
- **A task set.** `harness/tasks/example.jsonl` is a placeholder and says so in its first line; the
  one task that has actually run was written into a script. The target capability's tasks have to
  induce the error the student must recover from, and each needs a known-good recovery so the gate
  has an outcome to validate against. See [finetune.md](finetune.md) for the collection arithmetic.

## Prerequisites

Building the harness needs a C++ toolchain, the Windows SDK, and **the Node headers**. The first two
this machine had (MSVC 14.41 at `D:\VS\IDE`, SDK 10.0.22621.0); the headers it did not, and the
build stops with `Node headers/library missing`.

`packages/system-node/scripts/build-native.mjs` looks in exactly two places on Windows:
`AGNES_NODE_HEADERS`, or `%LOCALAPPDATA%\node-gyp\Cache\<node version>\`. Both want the node-gyp
layout — `<root>\include\node\node_api.h` and `<root>\<arch>\node.lib` — and neither exists until
something downloads them. A checkout with no `node_modules` has nothing to trigger that download,
which is why a fresh machine hits this.

```sh
# fetch once, then point the build at it
export AGNES_NODE_HEADERS=/path/to/node-headers   # ...\include\node\node_api.h lives here
pnpm --filter @agnes/cli build:local
```

The two files come from `https://nodejs.org/download/release/v<version>/`:
`node-v<version>-headers.tar.gz` (the `include/node` tree) and `win-x64/node.lib`.

## Running a session

Three things about the harness's own CLI are worth knowing before pointing a runner at it, all of
them found by running it rather than by reading it:

- **A profile name must be `local-dev` or `enterprise`.** The resolver passes the name through as
  the builtin template and refuses anything else (`E_DEP_MISSING: no builtin template <name>`). A
  deployment customises a template by putting its own `profile.yaml` under that template's name —
  the user layer merges over the builtin one — so `harness/profile/chtd.yaml` belongs at
  `<home>/profiles/local-dev/profile.yaml`.
- **`--standalone` is needed here**, because the daemon does not come up on this machine
  (`local daemon startup failed: daemon child exited before readiness (1)`). With it, a session
  opens and writes its ledger.
- **Every `ModelRecord` repeats its `route` and `baseUrl`.** Both are required fields on the record,
  and the assembled registry matches a slot on the pair read off the record rather than off its
  enclosing route: a model that inherits them is refused with `model-unserved`.

## Tests

```
python -m pytest
```

The suite builds real ledgers with the harness's own DDL and reads them back, so the reader is
tested against the schema the harness writes rather than one this repository invented. No test needs
a model, a GPU, or a network.
