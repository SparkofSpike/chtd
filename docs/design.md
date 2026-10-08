# CHTD Design

> The core design document for CHTD. Open questions are marked as open; nothing here is written as settled when it is not.

## 1. The problem

Agentic capability is usually distilled across harnesses. The teacher runs in its own scaffolding — its own tool schemas, step limits, retry policy and context assembly — and the student is trained on the transcripts that scaffolding produced. Two failures follow.

**The student learns the scaffolding, not the capability.** Trajectories carry the teacher harness's artifacts: its prompt structure, its tool-call formatting, its stopping behaviour. Once the student runs behind a different harness, exactly those artifacts stop matching, and the transfer does not hold.

**The evaluation rewards the wrong thing.** Length, latency, token spend and success on unrelated subtasks all move together with the target capability. An improvement is attributed to the wrong cause.

## 2. The core insight: why co-harness is required

When teacher and student run behind different harnesses, the difference between their trajectories mixes at least three sources: the model, the harness, and the task. To distil anything, you first have to prove which one you attributed correctly.

Co-harness collapses that three-way problem into one: **only the model varies**. Teacher and student run the same tool set, the same loop, the same stop rules and the same context assembly. The only difference is the model behind the loop. This gives two things:

- A decision divergence under an identical prefix is a clean, attributable supervision signal.
- Any improvement can only come from the model, never from an accident of scaffolding.

This is not a constraint CHTD invented. Agnes Harness already ships `packages/code/presets/minimal-rl.yaml`, an reinforcement-learning baseline that empties `prompt_sections` (no persona, no doctrine, no environment), sets `subagent.max_depth: 0`, `verifier` tier 0 and `telemetry` DISABLED — and freezes its bytes, with the reason stated in the file: *a training run measured against this recipe is only comparable to another run if the recipe was byte-identical.*

That reason is CHTD's reason. Co-harness was already an admitted premise of the harness; CHTD builds on it.

**Implementation note.** `minimal-rl` is signed: `minimal-rl.sha256` verifies its bytes and it must not be edited. CHTD needs its own preset — same spirit, adjusted tool set and step budget — frozen against its own hash inside this repository.

## 3. Three mechanisms

### 3.1 Co-harness execution

Teacher and student share one frozen preset and run the same task set, writing to the same append-only ledger.

What Agnes Harness provides:

- Session history is an append-only event ledger with an integrity chain; history cannot be edited in place.
- Fork and rewind branch a new session from a completed-turn boundary, inheriting the parent's immutable prefix.
- The preset freeze mechanism makes two runs comparable by construction.

### 3.2 Targeted distillation

The central design choice: **do not learn whole trajectories; learn the decision divergences.**

1. On the same task under the same prefix, teacher and student each produce a next step.
2. Align the two trajectories and locate the **divergence points** — a different tool, different arguments, a different stopping decision.
3. Use the teacher's decision as the label at that point.

A dataset built this way is targeted by construction. Steps unrelated to the target capability — environment output, routine operations with no divergence — never enter the training set.

Scope is deliberately narrow: the first experiment targets **one** capability, not several. See §5.

### 3.3 Distractor stripping

Two layers. Both are required, or the claim does not hold.

**Training layer.** Keep only divergence points; discard trajectory segments unrelated to the target capability; discard fields such as length and wall-clock that correlate with capability without causing it.

**Reporting layer.** Do not report total tokens, total steps or total duration — the metrics that move together with everything. Report the target capability's success rate on held-out tasks, and the corresponding change in failure modes.

Stripping must be defended with evidence, not asserted. For every candidate distractor, run a correlation check: if it moves with the target capability and has no causal path to it, strip it from the report; if it does have a causal path — a capability that inherently requires fewer steps — keep it and say why.

## 4. The L2 technical path

Levels: L1 changes the context only; **L2 fine-tunes weights**; L3 is full-parameter pretraining.

CHTD takes L2. L1 is too light to support the claim that a capability actually moved; L3 does not fit any realistic window for this project.

### 4.1 Roles

| Role | Model | Weights | Responsibility |
| --- | --- | --- | --- |
| Teacher | Qwen (large) | Frozen | Runs tasks on the frozen track; supplies decision supervision |
| Student | Qwen (small) | LoRA | Runs tasks on the **same frozen track**; receives the distillation |

The student must be small enough for LoRA; the teacher must be clearly stronger. Model sizes are open — see §5.

### 4.2 Pipeline

```
[1] Task set            one Agnes Harness session per task

[2] Teacher rollout     run on the frozen track -> trajectory A

[3] Student rollout     same track, same task, same prefix -> trajectory B

[4] Alignment           align A/B step by step -> set of divergence points

[5] Dataset             divergence points -> (state, teacher decision) pairs

[6] LoRA fine-tune      student base weights -> tuned weights

[7] Co-harness re-eval  same tasks, same track, before vs after

[8] Distractor analysis strip co-moving but non-causal metrics; report the net effect
```

Hard constraint: **[2] and [3] must use the identical frozen preset**, or the alignment in [4] means nothing. This is the assumption the whole pipeline rests on and must be enforced by hash, not by convention.

### 4.3 Interface to Agnes Harness

CHTD does not modify harness core code. Three integration points, and no more:

1. **Preset** — CHTD's own frozen track under `harness/presets/`.
2. **Task driver** — AGH CLI and SDK to launch and run sessions in batch.
3. **Trace reader** — read trajectories from the append-only ledger (session fork, event stream).

## 5. Open questions

These must be settled before implementation, or the work gets redone.

- **Target capability.** Exactly one for the first experiment. Candidates: error recovery, tool selection, multi-step decomposition, verification behaviour. Recommendation: **error recovery** — divergence points are easy to produce and failure modes are legible.
- **Model sizes.** Which Qwen variants. Constraints: the teacher clearly stronger than the student; the student small enough for single-GPU LoRA; both from the same family, to rule out tokenizer and pretraining differences.
- **Compute.** LoRA training needs a GPU. Type and VRAM are unconfirmed, and this is the largest unknown in the project.
- **Task set.** Source (AGH examples, self-built, public benchmark) and size.
- **Compliance.** The hackathon registration carries a technical commitment that models are limited to Agnes, while this project uses Qwen. The boundary must be confirmed through official channels.

## 6. Non-goals

- **No changes to harness core.** Integration goes through the preset, the CLI and the ledger.
- **No general distillation framework.** CHTD is one capability, one experiment, one set of reproducible evidence. Framework work comes after the result holds, if ever.
- **No reporting of metrics without a causal basis.**
