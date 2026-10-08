# CHTD Design

> The core design document for CHTD. Open questions are marked as open; nothing here is written as settled when it is not.

## 1. The problem

Agentic capability is usually distilled across harnesses. The teacher runs in its own scaffolding — its own tool schemas, step limits, retry policy and context assembly — and the student is trained on the transcripts that scaffolding produced. Two failures follow.

**The student learns the scaffolding, not the capability.** Trajectories carry the teacher harness's artifacts: its prompt structure, its tool-call formatting, its stopping behaviour. Once the student runs behind a different harness, exactly those artifacts stop matching, and the transfer does not hold.

**The evaluation rewards the wrong thing.** Length, latency, token spend and success on unrelated subtasks all move together with the target capability. An improvement is then attributed to the wrong cause.

## 2. The core insight: why co-harness is required

When teacher and student run behind different harnesses, the difference between their trajectories mixes at least three sources: the model, the harness, and the task. With `H_B` for the teacher's scaffolding and `H_A` for the student's, what a cross-harness measurement reports is

```
delta = f_B(theta_B, H_B) - f_A(theta_A, H_A)
```

and nothing in that quantity separates the model term from the scaffolding term. Co-harness removes the scaffolding term by construction:

```
delta = f_B(theta_B, H) - f_A(theta_A, H)     # only theta varies
```

Teacher and student run the same tool set, the same loop, the same stop rules and the same context assembly. The only difference is the model behind the loop. This gives two things:

- A decision divergence under an identical prefix is a clean, attributable supervision signal.
- Any improvement can only come from the model, never from an accident of scaffolding.

This is not a constraint CHTD invented. Agnes Harness already ships `packages/code/presets/minimal-rl.yaml`, a reinforcement-learning baseline that empties `prompt_sections` (no persona, no doctrine, no environment), sets `subagent.max_depth: 0`, `verifier` tier 0 and `telemetry` DISABLED — and freezes its bytes, with the reason stated in the file: *a training run measured against this recipe is only comparable to another run if the recipe was byte-identical.*

That reason is CHTD's reason. Co-harness was already an admitted premise of the harness; CHTD builds on it.

### 2.1 What co-harness does *not* buy

Sharing a harness does not by itself make the signal targeted. It converts the question from *teacher agent vs student agent* into *teacher model vs student model*. What remains to be earned:

- the target capability must be named and measured;
- the teacher's feedback must land on states the student actually reaches;
- the fidelity of that feedback must be checked;
- distractor variables must be controlled, not merely hoped away;
- and the result must survive a harness change.

Each of those is a mechanism below, not an assumption.

**Implementation note.** `minimal-rl` is signed and deliberately not extendable: a preset that extended it would change what it measures while still being called by its name. CHTD needs its own preset — same spirit, adjusted tool set and step budget — frozen against its own hash.

## 3. Mechanisms

### 3.1 Co-harness execution

One frozen preset, shared by both sides, with the model supplied by the deployment rather than the recipe. The preset's route targets are the `default` sentinel, which is what lets the *same* preset resolve to a teacher adapter in one session and a student adapter in another and remain one track.

This is verifiable rather than asserted: a session records `resolvedPresetHash` in its metadata (`packages/protocol/schema/session-v1.json`), so two runs can be shown to have run the same track instead of being claimed to have done so. CHTD refuses to compare runs whose hashes differ.

### 3.2 Targeted distillation: teach on the student's states

The central design choice, and the difference between this and off-policy imitation:

> The teacher is not asked to demonstrate on its own trajectory. It is asked to respond at the states the **student** reached.

Off-policy imitation trains on `(state, action)` pairs drawn from the teacher's distribution. The student's errors put it into states the teacher's data never covers, and those are exactly the states where the student needs help. Following the DAgger argument, supervision is collected on the student's own state distribution:

1. The student runs the task on the frozen track; its trajectory defines the states of interest.
2. The teacher is shown each of those states — reconstructed from the shared ledger, via fork from the same boundary — and produces its decision there.
3. Divergences between the student's action and the teacher's become `(state, teacher decision)` pairs.
4. The student is fine-tuned on those pairs.

A dataset built this way is targeted by construction: steps where the two agree, and steps unrelated to the target capability, never enter the training set.

**Signal quality gate.** A divergence is not automatically a place where the teacher is right. Without a filter, the dataset teaches the teacher's quirks along with its capability. Only divergences whose teacher-side choice is validated by the task outcome — or by a verifier — are admitted.

### 3.3 Teacher privilege stays out of the student's input

The teacher may see more than the student: the student's full trajectory, task outcomes, verifier results, and the reference solution where one exists. That privilege is used to *produce supervision*, never to *inform the student*. Concretely:

- teacher and student are separate instances with no shared hidden state;
- the student observes only what the track gives it;
- privileged information appears in the training label, never in the model's input;
- at evaluation time the teacher is absent entirely.

Without this separation the student can learn to depend on hints that will not exist at deployment, and the measured gain is an artifact of the scaffold rather than a capability.

### 3.4 Distractor stripping

Two layers. Both are required, or the claim does not hold.

**Training layer.** Keep only divergence points; discard trajectory segments unrelated to the target capability; discard fields such as length and wall-clock that correlate with capability without causing it.

**Reporting layer.** Decompose the result instead of collapsing it into one number:

| Group | Examples | Role in the report |
| --- | --- | --- |
| Target capability | tool-call correctness, code pass rate, recovery-after-failure rate | The claim |
| Distractor | output length, verbosity, formatting preference, step count, memory dependence | Reported to show they did *not* move, or stripped with a stated reason |
| Transfer | target-capability retention under a changed harness | Guards against scaffold overfit (§3.5) |
| Regression | general capability of the base model | Guards against capability loss |

Stripping must be defended with evidence, not asserted. For every candidate distractor, run a correlation check: if it moves with the target capability and has no causal path to it, strip it from the report; if it does have a causal path — a capability that inherently requires fewer steps — keep it and say why.

### 3.5 Harness-stripping test: the anti-overfit gate

Co-harness removes the *confound* between model and scaffold. It introduces a distinct risk: a student trained inside one harness may learn *how to behave inside that harness* rather than acquiring the capability. The failure is invisible if evaluation also happens inside that harness.

So evaluation runs on more than the training track. Ordered by how much they strip away:

1. **Same track** — the primary measurement.
2. **Different preset** — e.g. the shipped `standard` recipe: different prompt sections and budgets, same harness.
3. **No tools** — the task attempted without the action space.
4. **Single turn** — no loop at all.

If the gain survives (2) it is not merely a prompt-format fit; if it survives (3) and (4) the capability has partially detached from the scaffold. A gain that exists only under (1) is scaffold overfit and must be reported as such, not as a capability.

Using the same harness under a different preset is a deliberate substitute for swapping in a foreign harness: it varies the parts of `H` that a student could overfit to, while keeping the ledger and the comparison machinery identical. A genuinely foreign harness is the stronger test and is left as future work.

## 4. The L2 technical path

Levels: L1 tunes the context only; **L2 fine-tunes weights**; L3 is full-parameter pretraining. CHTD takes L2: L1 is too light to support the claim that a capability actually moved, and L3 does not fit the window.

### 4.0 Delivery shape: how the track reaches the harness

Verified against the harness source, not assumed. A preset reaches the runtime through a **package's `presets` export**: `packages/host/src/assemble.ts` collects `module.presets` from the package modules selected for the Host generation, and `validatePresetCatalog` refuses a profile whose `presets.allowed` names a preset no package provides.

A package outside the built-in set is resolved through the **lockfile**: `packages/host/src/profile/resolve.ts` throws `E_DEP_MISSING` for any non-builtin package without an `agnes-lock.json` entry, and only the closed builtin set receives the `builtin:` integrity sentinel. Everything else carries its recorded `version`, `integrity` and `trust`.

The consequence for CHTD — and the reason this section exists:

- CHTD ships as its **own package** exporting `presets.chtd`, installed and trusted through the normal package flow. No harness file is edited.
- The Profile then names it: `packages` gains the CHTD entry, `presets.allowed` gains `chtd`.
- Because presence and integrity are lockfile facts, **the exact track a run used is attributable after the fact** — which is the property the experiment needs.

### 4.1 Roles

| Role | Model | Weights | Responsibility |
| --- | --- | --- | --- |
| Teacher | Qwen (large) | Frozen | Decides at the student's states; supplies supervision |
| Student | Qwen (small) | LoRA | Runs tasks on the frozen track; receives the distillation |

Both are served behind the same harness through different deployment routes, so the track stays one track. Model sizes are open — see §6.

### 4.2 Pipeline

```
[1] Task set             one Agnes Harness session per task

[2] Student rollout      student on the frozen track -> trajectory B

[3] Teacher at B's states  teacher answers B's decision points, via same-prefix fork -> answers A

[4] Divergence extraction  (state, teacher decision) where A and B differ, outcome-validated

[5] LoRA fine-tune        student base weights -> tuned weights

[6] Co-harness re-eval    same track, before vs after, held-out tasks

[7] Harness-stripping     repeat [6] under a different preset, no tools, single turn

[8] Distractor analysis   decompose the result; strip non-causal co-movers with reasons
```

Note the ordering, which is the whole point: **the student runs first**. Supervision is collected on the states the student actually visits, not on the teacher's own path. This is DAgger's correction applied through the harness, and it is why the harness must be shared — the teacher has to be dropped into the student's exact prefix, which requires both to be running the same loop against the same ledger.

Hard constraint: **[2], [3] and [6] must use the identical frozen preset.** Enforced by `resolvedPresetHash`, not by convention.

### 4.3 Interface to Agnes Harness

CHTD does not modify harness core code. The integration points:

1. **Preset** — `harness/presets/chtd.yaml`, frozen by `harness/freeze.py`, delivered as a package `presets` export (§4.0).
2. **Routes** — teacher and student are separate routes in the Profile's `provider.routes`. `RouteDecl` carries a `baseUrl` (visible in `packages/host/test/assemble/assemble.test.ts`), so a locally served fine-tuned student and a remotely served teacher can run the same track. This is configuration, not a core change.
3. **Task driver** — AGH CLI and SDK to launch sessions and run tasks in batch.
4. **Trace reader** — trajectories from the append-only ledger; same-prefix fork for teacher-at-student-states.

Qwen is a first-class provider in the harness: `packages/ai/src/adapters/pi/api-key-providers.ts` declares a `qwen` provider (`ApiKeyProviderId` includes `'qwen'`, served over `openai-completions`). The teacher side is therefore a configuration entry, not new code. The student side needs a route pointing at wherever the fine-tuned weights are served.

## 5. The comparison that decides whether co-harness earns its name

The single most important experiment is not "does the student improve" but "does sharing the harness cause the improvement":

| Arm | Teacher data comes from | Everything else |
| --- | --- | --- |
| **Baseline** | the teacher's own trajectory (off-policy imitation) | identical track, tasks, budget, LoRA recipe |
| **CHTD** | the student's states, teacher-answered (on-policy) | identical track, tasks, budget, LoRA recipe |

If the CHTD arm does not beat the baseline — especially under the harness-stripping tests of §3.5 — then co-harness is a control, not a contribution, and the honest report says so.

## 6. Open questions

These must be settled before implementation, or the work gets redone.

- **Target capability.** Exactly one. Candidates: error recovery, tool selection, multi-step decomposition, verification behaviour. Recommendation: **error recovery** — divergence points are easy to produce and failure modes are legible.
- **Model sizes.** Which Qwen variants. Constraints: the teacher clearly stronger than the student; the student small enough for single-GPU LoRA; both from the same family, to rule out tokenizer and pretraining differences.
- **Compute.** LoRA training needs a GPU. Type and VRAM are unconfirmed, and this is the largest unknown in the project.
- **Task set.** Source (AGH examples, self-built, public benchmark) and size. Tasks need enough structure that teacher and student share an early prefix, or there is nothing to align.
- **Compliance.** The hackathon registration carries a technical commitment that models are limited to Agnes, while this project uses Qwen. The boundary must be confirmed through official channels.

## 7. Non-goals

- **No changes to harness core.** Integration goes through the preset, the routes, the CLI and the ledger.
- **No general distillation framework.** CHTD is one capability, one comparison, one set of reproducible evidence.
- **No multi-round online RL loop in the first experiment.** DAgger in principle iterates; here it runs one round (collect, teach, train) and stops. Aggregation across rounds is future work once one round holds.
- **No reporting of metrics without a causal basis.**
