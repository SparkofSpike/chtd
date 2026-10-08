# CHTD — Co-Harness Targeted Distillation

**Teacher and student LLMs share one harness and run the same task loop, so the teacher's agentic capability is targetedly distilled into the student while distractors unrelated to that capability are automatically stripped.**

Status: design stage · built on [Agnes Harness](https://github.com/AgnesAI-Labs/agnes-harness)

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
- **A verifiable track.** A session records the resolved preset hash, so two runs can be *shown* to have run the same track instead of being claimed to have done so.
- **Append-only session records.** Trajectories are exact and replayable rather than reconstructed after the fact.
- **Fork and rewind.** The same prefix can be continued under a different policy — which is exactly what dropping the teacher into the student's states requires.
- **A frozen-baseline precedent.** The shipped `minimal-rl` preset already freezes its bytes, for the reason this project rests on: a baseline that drifts measures a different experiment.

The harness is therefore not merely a container for the experiment; it is the invariant that makes the comparison meaningful.

## Repository layout

```
harness/   The frozen preset, its freeze gate, and rollout collection
distill/   Dataset construction and LoRA training
eval/      Co-harness evaluation, the arm comparison and distractor analysis
docs/      Design, method and experiment plan
data/      Generated artifacts (not tracked)
```

## Status

The method is specified in [docs/design.md](docs/design.md); the delivery plan is in [docs/plan.md](docs/plan.md).

The frozen co-harness track is in place and verified: `harness/presets/chtd.yaml` is byte-frozen by `harness/freeze.py`, a one-character edit is refused, and the stored bytes survive git unchanged. No rollout or training run has been performed yet; those wait on the open questions in the design (compute, model sizes, target capability).

## License

TBD — Apache-2.0 proposed, to match the harness this work builds on.
