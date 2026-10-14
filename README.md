# CHTD — Co-Harness Targeted Distillation

**Teacher and student LLMs share one harness and run the same task loop, so the teacher's agentic capability is targetedly distilled into the student while distractors unrelated to that capability are automatically stripped.**

Status: experiments complete (2026-10-14) — the method, the measurements and the negative result are summarised in [docs/findings-20261014.md](docs/findings-20261014.md); built on [Agnes Harness](https://github.com/AgnesAI-Labs/agnes-harness)

---

## The problem

Agentic capability is usually distilled across harnesses. The teacher runs in its own scaffolding — its own tool schemas, step limits, retry policy and context assembly — and the student is trained on the transcripts that scaffolding produced. Two failures follow.

1. **The student learns the scaffolding, not the capability.** Trajectories carry the teacher harness's artifacts. Once the student runs behind a different harness, exactly those artifacts stop matching, and the transfer does not hold.
2. **The evaluation rewards the wrong thing.** Aggregate metrics — length, latency, token spend, success on unrelated subtasks — move together with the target capability. An improvement is then attributed to the wrong cause.

## The approach

Three commitments define CHTD.

### 1. Co-harness execution

Teacher and student run the **same harness**: same tool set, same loop, same step and stop rules, same context assembly. The only difference is the model behind the loop. Rollouts land on one shared, append-only ledger, so a teacher step and a student step are directly comparable.

In practice this means a **byte-frozen preset**. Agnes Harness already ships one — `minimal-rl`, whose exact bytes are hash-checked because *a training run measured against this recipe is only comparable to another run if the recipe was byte-identical*. CHTD starts from that spirit: persona, environment noise, subagents and telemetry are stripped so the only variable left is the model.

### 2. Targeted distillation: teach on the student's states

The capability to transfer is named up front — error recovery, tool selection, multi-step decomposition, or another axis. Then the teacher is not asked to demonstrate on its own trajectory, but to respond at the states the **student** actually reached. The student runs first; the teacher is dropped into the same prefix through a fork and answers there; where the two diverge, the teacher's decision becomes the label. Supervision therefore lands on the states where the student needs help, rather than on the ones where the teacher was already comfortable.

### 3. Distractor stripping, and the test that keeps it honest

Metrics that correlate with the target capability without being caused by it are identified and removed — from both the training objective and the reported evidence. What remains is the signal the work claims to improve.

Stripping addresses one failure mode and invites another: a student can learn to behave well *inside this harness* without acquiring the capability. So the result is also measured with the harness stripped away — a different preset, no tools, a single turn. A gain that exists only on the training track is reported as scaffold overfit, not as a capability.

## Current experiment setup

| Element | Choice |
| --- | --- |
| Harness | Agnes Harness, one frozen preset shared by both sides |
| Teacher | Qwen (large), frozen weights |
| Student | Qwen (small), LoRA |
| Signal | Teacher decisions at the student's states, admitted only where the outcome validates them |
| Deciding comparison | Teacher data from its own trajectory vs from the student's — same track, same recipe |
| Evaluation | Held-out tasks, plus a stripping ladder: different preset, no tools, single turn |

Model sizes are fixed together with the target capability; see [docs/design.md](docs/design.md).

## Why Agnes Harness

Agnes Harness already provides the pieces co-harness execution needs:

- **One loop, pluggable models.** Teacher and student are two routes behind the same runtime loop; the track names neither model, so the same recipe resolves to either.
- **A verifiable track.** The track is delivered as a package whose bytes are hash-checked at load, so a drifted track fails to load at all; the integrity that covered it is recorded per run, in the lockfile.
- **Append-only session records.** Trajectories are exact and replayable rather than reconstructed after the fact.
- **Fork and rewind.** The same prefix can be continued under a different policy — which is exactly what dropping the teacher into the student's states requires.
- **A frozen-baseline precedent.** The shipped `minimal-rl` preset already freezes its bytes, for the reason this project rests on: a baseline that drifts measures a different experiment.

The harness is therefore not merely a container for the experiment; it is the invariant that makes the comparison meaningful.

## Repository layout

```
harness/   The frozen track, its freeze gate, the Profile, and rollout collection
packages/  The frozen track delivered as a package the harness can install
distill/   Dataset construction, the signal-quality gate, SFT sample building, and LoRA training
eval/      Co-harness evaluation, the arm comparison, the instrument benchmarks and the power analysis
tests/     The pipeline's tests — real ledgers, no model, no GPU, no network
docs/      Design, method, experiment plan, and the findings write-up
runs/      Per-run records: what was run, what it measured, and what it cannot support
reports/   Audits of individual batches (e.g. the collaborator-run holdout-4)
data/      Generated artifacts (not tracked)
```

## Status

The method is specified in [docs/design.md](docs/design.md); the delivery plan is in [docs/plan.md](docs/plan.md), and the code that implements it in [docs/pipeline.md](docs/pipeline.md).

The frozen co-harness track is in place and verified: `harness/presets/chtd.yaml` is byte-frozen by `harness/freeze.py`, a one-character edit is refused, and the stored bytes survive git unchanged. The track is also delivered as an installable package — `packages/chtd/` — whose own load-time gate refuses drifted bytes; `packages/chtd/README.md` records why it ships as a JS module and what an upstream loader quirk requires of it.

The pipeline is built and tested without a model: reading a ledger, extracting decision points,
aligning the teacher's forks against the student's states, gating the divergences, building the
dataset with a digest a reader can reproduce, refusing to compare arms that were not run
identically, assessing the stripping ladder, and generating the write-up. Its four refusals — a
writable ledger, a dataset from two tracks, a comparison of unlike arms, a kept co-mover without a
reason — are listed with the reason each is a refusal rather than a warning in
[docs/pipeline.md](docs/pipeline.md).

**Rollouts and a training run have now happened, at smoke-test scale.** A student and a teacher ran
the same frozen track against real endpoints, the gate admitted one divergence, and the first LoRA
fine-tune was carried out on a 48 GB 4090 — the adapter loads, serves behind the harness's student
route, and runs on the frozen track ([docs/finetune.md](docs/finetune.md), [runs/chtd-1/](runs/chtd-1/)).

**What that is not is evidence for the claim.** One entry, 20 supervised tokens, one effective
optimizer step: the tuned student's answer is wrong in the same way the untuned one was. The dataset
is a smoke test, and what would have to happen to get a real one — a task set whose failures are
legible, tasks with more steps, and throttled collection against a rate limit — is set out with its
arithmetic in [docs/finetune.md](docs/finetune.md).

**The full programme has since run, and it is written up.** Three fresh held-out sets (15, 12 and
30 tasks, pooled to 57), a five-point dose ladder, up to three training seeds per arm, every answer
graded on a three-level rubric (one set by hand, the rest by the validated LLM judge), the phrase
gate benchmarked against two human readers, and the measurement's own noise floors measured rather
than assumed. The headline, with its certainty: **fine-tuning works in the endpoint the design
names** — base 4–5 failures of 15 diagnosis tasks, every trained configuration 0–2, joint
probability ≤ 3.4 × 10⁻⁹ under the observed base rate; pooled over 57 tasks the student-state
family halves the base's failure rate (31.6 % → 17.5 %, p = 0.077). The CHTD shape's distinct claim
is **data efficiency**: 9 student-state rows reach the 33-row teacher-demo point, a one-cell margin
on single evaluations. Two measurement findings constrain everything else: the **base is
deterministic** (six evaluations, byte-identical answers) while **an adapter is not** (same weights,
two evaluations, 9/15 answers differ — worth 0.33 ordinal points), so single 15-task comparisons of
trained arms are uninformative by construction. The fine-tunes' instability across seeds and
prompts is itself a reproduced finding, rooted in `last` supervision teaching the answer but never
the exploration policy. The single entry point, with what the evidence supports and what it does
not, is [docs/findings-20261014.md](docs/findings-20261014.md).

## License

Apache-2.0 — see [LICENSE](LICENSE). Chosen to match the harness this work builds on.
