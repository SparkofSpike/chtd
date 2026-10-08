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

### 2. Targeted distillation

The capability to transfer is named up front — error recovery, tool selection, multi-step decomposition, or another axis. Supervision is built from **aligned decision points** in the shared loop, not from whole transcripts: where teacher and student diverge on the same prefix, the teacher's decision becomes the label.

### 3. Distractor stripping

Metrics that correlate with the target capability without being caused by it are identified and removed — from both the training objective and the reported evidence. What remains is the signal the work claims to improve.

## Current experiment setup

| Element | Choice |
| --- | --- |
| Harness | Agnes Harness, one frozen preset shared by both sides |
| Teacher | Qwen (large), frozen weights |
| Student | Qwen (small), LoRA |
| Signal | Decision-point divergence on aligned co-harness rollouts |
| Evaluation | Same tasks, same harness, before/after the student's fine-tune |

Model sizes are fixed together with the target capability; see [docs/design.md](docs/design.md).

## Repository layout

```
harness/   Integration with Agnes Harness: the frozen preset, rollout collection
distill/   Dataset construction and LoRA training
eval/      Co-harness evaluation and distractor-stripping analysis
docs/      Design, method and experiment plan
data/      Generated artifacts (not tracked)
```

## Status

Design stage. The method is specified in [docs/design.md](docs/design.md); the delivery plan is in [docs/plan.md](docs/plan.md). No training run has been performed yet.

## License

TBD — Apache-2.0 proposed, to match the harness this work builds on.
