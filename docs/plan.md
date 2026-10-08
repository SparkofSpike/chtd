# CHTD Experiment Plan

> Delivery plan for the L2 experiment. Ordering is by dependency, not by preference.

## 0. Critical path

Two unknowns gate everything downstream. Settle them first.

- **Compute** — which GPU, how much VRAM. Without it there is no LoRA step and no L2.
- **Model sizes** — teacher and student variants, chosen against that GPU's budget.

Until both are fixed, the work below can only proceed as far as data collection.

## 1. Milestones

### M0 — Decisions (blocks everything)

- [ ] Confirm available GPU and VRAM.
- [ ] Pick teacher and student Qwen variants.
- [ ] Pick the target capability for the first experiment.
- [ ] Confirm the Agnes-model compliance boundary for the hackathon.

Exit: the four items are written down and no longer open.

### M1 — Frozen track — DONE

- [x] Author `harness/presets/chtd.yaml`: minimal, explicit, one tool set, fixed step budget.
- [x] Freeze its hash (`harness/freeze.py`), mirroring the harness's own freeze gate.
- [x] Prove the gate fails on an edited byte and passes on the frozen file.
- [x] Prove the bytes survive git: the stored blob, the worktree and the recorded hash are identical.

Measured: `chtd.yaml` is `4663943a4802f4559b6e0c73f91e242e8bb7b2aff051608a5a4afe80ef2e3252`; a one-character comment edit is refused.

### M2 — Delivery, routes and rollout collection

- [x] Package the frozen track as a CHTD package exporting `presets.chtd`. Author and verified: the package is `packages/chtd/`, its load-time gate refuses drifted bytes, and an end-to-end run against `assemble()` mounts `preset:chtd` from a snapshot install. The track is delivered as a JS module rather than the YAML because a non-builtin package is installed without `node_modules` — see `packages/chtd/README.md`. Not yet exercised: installing through `agnes package add` against a built harness, which is the next item's prerequisite.
- [ ] Add the package to a Profile's `packages` and `presets.allowed`; confirm the track resolves. Authored in `harness/profile/chtd.yaml` (the package entry and `presets: { default: chtd, allowed: [chtd] }`); not yet confirmed on a running instance — that needs a built harness.
- [ ] Register a teacher route and a student route in that Profile, each with its own `baseUrl`. Both routes are declared, with the teacher first so the `default` sentinel resolves there; the two `models` catalogues and the teacher's `baseUrl` wait on M0.
- [ ] Run both against the `chtd` preset and confirm from recorded evidence that they ran the same track: the CHTD package's `integrity` identical in both lockfiles, and the two `resolvedProfileHash` values differing only through `provider.routes`.
- [ ] Drive AGH from its CLI/SDK, one session per task; collect student trajectories from the ledger.

Exit: a pilot task set produces student trajectories that replay, on a track provably identical to the teacher's, with the track's integrity recorded in the lockfile.

### M3 — Teacher-at-student-states dataset

- [x] For each student decision point, fork the session at that boundary and have the teacher answer there. The fork and the pairing are implemented and tested: `harness/trajectory.py` gives every decision a prefix digest, and `distill/divergence.py` pairs the two sides by it. The mechanical driving of forks waits on M0.
- [x] Extract divergences; emit `(state, teacher decision)` pairs. `distill/dataset.py` emits them with a digest and a manifest.
- [x] Apply the signal-quality gate: keep only divergences whose teacher choice the outcome validates. `distill/gate.py`; fail-closed, and every refusal is recorded with its reason.
- [ ] Sanity-check the set: divergences should reflect the target capability, not noise. Needs real rollouts.

Exit: a dataset whose entries are individually inspectable and traceable to a ledger event, with privileged teacher information excluded from the model input. Implemented — an entry carries no field that could hold gate evidence, and a test asserts the key set. Not yet exercised on real data.

### M4 — LoRA fine-tune

- [ ] Training run on the student base weights. `distill/train.py` emits the plan (recipe, provenance, dataset digest) and refuses without an accelerator; the loop itself is a GPU-side step.
- [x] Record hyperparameters, dataset hash, base-model revision. `distill/train.py` freezes the recipe with the plan, and the plan names the dataset digest and the base revision.
- [ ] Confirm the tuned student loads and serves behind a route.

Exit: tuned weights exist, are reproducible from recorded inputs, and run inside the harness on the frozen track.

### M5 — Evaluation and the deciding comparison

- [ ] Same track, same tasks, before vs after, on held-out tasks.
- [x] **Baseline arm**: teacher data from the teacher's own trajectory (off-policy). Same track, tasks, budget and LoRA recipe. `eval/arms.py` refuses a comparison whose arms differ on any of those four.
- [x] **CHTD arm**: teacher data from the student's states (on-policy). `distill/divergence.py` pairs on state, so this is what the pairing means.
- [x] Harness-stripping tests: different preset, no tools, single turn (§3.5 of the design). `eval/stripping.py`; a gain that does not survive is reported as `scaffold-overfit`, and a ladder with no stripped rung is refused rather than called survived.

Exit: a result that either supports or refutes the claim, with the arm comparison and the stripping ladder attached. The machinery is built and tested; the numbers wait on real runs.

### M6 — Distractor analysis and write-up

- [x] Correlation check on each candidate distractor. `eval/distractors.py`, over arm-to-arm deltas rather than raw values.
- [x] Decompose the result into capability / distractor / transfer / regression groups. The gate keeps refusals with reasons, the ladder keeps its rungs, and the arm comparison splits gained from lost.
- [x] Strip non-causal co-movers from the report; keep causal ones with a stated reason. A kept co-mover without a reason is refused, and the report prints the reason.
- [x] Write the result, limitations and reproduction steps. `eval/report.py` generates it from the objects above, so a report cannot omit what the pipeline found.

Exit: a reader can rerun the pipeline and reach the same numbers. The dataset digest and the track digest are the two things to compare; `distill/pipeline.py` is the command, and a test asserts the same inputs give the same digest.

## 2. Deliverables

| # | Artifact | Location |
| --- | --- | --- |
| 1 | Frozen preset + freeze gate | `harness/presets/`, `harness/freeze.py` |
| 2 | Route configuration and rollout collection | `harness/profile/`, `harness/rollout.py` |
| 3 | Teacher-at-student-states dataset builder | `distill/divergence.py`, `distill/gate.py`, `distill/dataset.py` |
| 4 | Training scripts and configs | `distill/train.py` |
| 5 | Evaluation, arm comparison, stripping ladder | `eval/` |
| 6 | Result write-up with reproduction steps | `eval/report.py`, `docs/` |
| 7 | The frozen track as an installable package | `packages/chtd/` |

## 3. Risks

**Compute is the top risk.** If no suitable GPU is available, L2 cannot be delivered and the project falls back to L1 while stating plainly that the weight-level claim was not tested.

**Alignment difficulty.** Teacher and student may diverge so early that few usable decision points exist. Mitigation: choose tasks with a structured early phase, and measure the divergence distribution before committing to a dataset size.

**Signal quality.** A divergence is not automatically a place where the teacher is right. Without the outcome gate, the dataset teaches the teacher's quirks along with its capability.

**Scaffold overfit.** A student can learn to behave well inside this harness without acquiring the capability. The stripping ladder in M5 is the guard; if the gain does not survive it, that is the finding.

**The baseline may win.** If off-policy imitation matches the on-policy arm, co-harness is a control rather than a contribution. The plan must be able to report that outcome as a result.

**Single-capability scope.** Proving a transfer on one capability says nothing about others. That limitation belongs in the write-up, not in a hedge.

## 4. What "done" means

The experiment is done when a reader can, from this repository alone: rebuild the frozen track, regenerate the dataset, reproduce the fine-tune, rerun both arms and the stripping ladder, and get the same numbers — including the negative ones.
