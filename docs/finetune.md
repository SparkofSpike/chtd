# The fine-tune path

From a recorded dataset to an adapter the harness can serve. Three modules and one capture:

```
data/datasets/<round>/        the gate's output: a state and the teacher's decision in it
        │
        ├─ distill/sft.py      + harness/prefix/capture-1.json  →  the rows a trainer eats
        │                                                          data/sft/<round>.jsonl
        ├─ distill/train.py    the plan: recipe, dataset digest, base model   data/sft/plan-*.json
        └─ distill/finetune.py the run itself, on a machine with a GPU
```

The split is deliberate. `distill/train.py` refuses without an accelerator and `distill/finetune.py`
cannot run without one, so a reviewer can check the recipe and the provenance on any machine while
only the weight-level step needs the box. What that step produced on its first run is recorded in
`runs/chtd-1/`.

## The harness prefix is a capture, not a reconstruction

A sample is the conversation the student was in plus the teacher's turn as the label. The
conversation comes from the ledger. What the ledger does **not** hold is the two things that sit in
front of it, and neither can be rebuilt from this repository:

- **The system prompt.** It is assembled from several packages — an untrusted-content rule, a
  computer-use note, and an `<available_skills>` block — and its composition is a property of the
  harness build, not of the preset.
- **The tool disclosure.** It comes from the extensions the assembly loads. The frozen preset's
  `tools.core: [shell, edit]` is `NOT CONSUMED` — the file says so itself — and the warning it
  carries is exact: *"if the deployed assembly ships more tools than this line names, both models see
  them, so the comparison survives — the track is merely wider than declared."*

Measured, on the first capture: **the assembly disclosed 22 tools**, including `plugin_helper_*`,
`skill_helper_*` and `mcp_manage`. A reconstruction from the preset would have shown the student two
tools and trained it on a conversation that never happened.

So `harness/prefix/capture-1.json` holds the bytes a live run sent, with the digest of the request
they were captured from and the route and model that received them. It is an input, kept under
`harness/` rather than `data/` because `data/` is gitignored and a fine-tune cannot be reproduced
without it.

**How to take another one.** The endpoint on the box dumps each request body when `SERVE_DUMP_DIR`
is set; `serve_openai.py --adapter` is the current version and the dump path is a two-line change
that was reverted after use. One ordinary student session on the frozen track is enough — a capture
is not a scored rollout.

## The lock that keeps training honest

`distill.finetune.build_example` renders each sample twice: once with `add_generation_prompt=True`
(the prompt, as the endpoint renders it) and once without (the whole conversation). If the first is
not a token-for-token prefix of the second, the sample is refused with `PrefixDrift`.

The alternative — a comment asking future editors to keep two renderers in sync — fails silently.
A model trained against a differently-rendered prompt learns a distribution the endpoint never
produces, and nothing downstream can detect it: the loss looks normal, the adapter loads, and the
result is unattributable.

Two consequences of the same reasoning are visible in the recipe:

- **Only the teacher's turn is supervised.** Everything before it is masked to `-100`. The student's
  own wrong answer is part of the state it must learn to recover from, so it stays in the input and
  out of the loss.
- **The prefix stays in.** The untrusted-content rules and the skills block read like noise for a
  task about opening a file. They are kept because the claim under test is that a capability
  transfers *within the harness*, and the harness is what the student meets at inference.

## The first run, and the size of the problem

One entry, three steps, 8.9 seconds. The loss falls on the third step and the adapter loads and
serves — the mechanism works end to end. The behaviour does not move, and `runs/chtd-1/README.md`
carries the numbers and the reasoning. The short version:

> Three steps on one sample move the weights and do not move the behaviour.

**And it is not bit-reproducible.** The same plan against the same samples was run twice and
produced two different adapters (`db98b9c9…` and `d2923b1b…`, mean loss 4.1329 and 4.1219). 4-bit
quantisation and bfloat16 accumulation on a GPU are not order-deterministic, so what a reader can
check is the **protocol** — the inputs, the recipe, the track, the digest — and the numbers land
near the recorded ones rather than on them. A claim that needs two runs to agree to the byte is not
available from this setup, and the preserved first-run numbers are in `runs/chtd-1/README.md`.

One row cannot support a statistical claim, and the first run must not be presented as if it could.
What the size problem actually is:

| | |
| --- | --- |
| Admitted samples | 1 |
| Supervision per sample | 20 tokens (one tool call) |
| Tracks | 1 task (`read-second-line`) |
| Decision points per rollout | 1 for the student, 2 for the teacher |

The bottleneck is not the training step — that takes nine seconds. It is **how many decisions a
rollout contains, and how many rollouts exist.** A sample is one divergence, so the dataset grows
with *decision points*, not with tasks: a task that both models complete in one step contributes
nothing.

## What would have to happen to get a dataset worth the name

Three things, in order, each measurable before the next is attempted.

**1. A task set whose failures are legible.** The current tasks are placeholders and
`harness/tasks/example.jsonl` says so in its first line; the one task that has actually run
(`read-second-line`) is written into a script. The target capability for the first experiment is
error recovery, so a task has to *induce* an error rather than merely permit one — a test suite that
fails, a command whose flag changed, an edit that left a module inconsistent. The three placeholder
prompts are the right shape and the wrong status: they need to be real, with a known-good recovery
each, so the gate has an outcome to validate against.

**2. More steps per rollout.** `max_steps: 60` in the frozen preset bounds a trajectory; the
observed rollouts used one or two. A task that takes four to eight steps with at least one failing
tool call is what produces decision points to align. The preset does not need to change — the tasks
do.

**3. Rollouts with the rate limit respected.** The Agnes endpoint returns 429 on a burst and one
request every six seconds succeeded ten times out of ten, so collection is throttled rather than
parallel. Cost per rollout, at the observed shape: a student session of ~1 request and a teacher
fork of ~2, so a five-step task is on the order of 5–10 requests and 30–60 seconds each arm.

The arithmetic that matters: **to get 100 admitted samples, roughly 50–100 task instances must
produce at least one divergence each.** At ~1 minute per arm per task, that is a few hours of
collection, not a few minutes — and the gate will refuse some of what it collects, which is the
gate working.

Until then the honest reading of `official-1` is: the path is complete and exercised end to end,
and the dataset is a smoke test.

## What is not built

- **Multi-round collection.** One DAgger round is implemented; aggregation across rounds is not.
- **Evaluation numbers.** `eval/` compares arms and assesses a stripping ladder, and nothing yet
  produces a ladder rung — the runner takes a route and a model but no preset and no tool or turn
  limit. That gap is unchanged by this work and is stated in `docs/pipeline.md`.
- **The baseline arm's data.** The deciding comparison needs teacher-from-its-own-trajectory data;
  only the CHTD arm's dataset exists.
