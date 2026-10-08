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
distill/train.py         the LoRA plan, and the accelerator check that stops it without a GPU
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

- **The training loop.** `distill/train.py` produces the plan — recipe, provenance, dataset digest —
  and refuses without an accelerator. It does not run PEFT; that is a GPU-side step.
- **The runner has not been exercised.** Building Agnes Harness on the machine this was written on
  fails in its own native step (`packages/system-node/scripts/build-native.mjs` wants Node headers
  that are not installed), so `CliRunner` is written from the CLI's documented arguments and
  verified only by construction. Everything it produces — ledger reading, trajectory extraction,
  divergence, gating — is tested against real ledgers.
- **One DAgger round.** `distill/pipeline.py` collects one round. Aggregation across rounds is
  future work, and the limitation is carried in every report.

## Tests

```
python -m pytest
```

The suite builds real ledgers with the harness's own DDL and reads them back, so the reader is
tested against the schema the harness writes rather than one this repository invented. No test needs
a model, a GPU, or a network.
