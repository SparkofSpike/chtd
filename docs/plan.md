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

- [ ] Package the frozen track as a CHTD package exporting `presets.chtd`; install and trust it through the normal package flow.
- [ ] Add the package to a Profile's `packages` and `presets.allowed`; confirm the track resolves.
- [ ] Register a teacher route and a student route in that Profile, each with its own `baseUrl`.
- [ ] Run both against the `chtd` preset and confirm from recorded evidence that they ran the same track: the CHTD package's `integrity` identical in both lockfiles, and the two `resolvedProfileHash` values differing only through `provider.routes`.
- [ ] Drive AGH from its CLI/SDK, one session per task; collect student trajectories from the ledger.

Exit: a pilot task set produces student trajectories that replay, on a track provably identical to the teacher's, with the track's integrity recorded in the lockfile.

### M3 — Teacher-at-student-states dataset

- [ ] For each student decision point, fork the session at that boundary and have the teacher answer there.
- [ ] Extract divergences; emit `(state, teacher decision)` pairs.
- [ ] Apply the signal-quality gate: keep only divergences whose teacher choice the outcome validates.
- [ ] Sanity-check the set: divergences should reflect the target capability, not noise.

Exit: a dataset whose entries are individually inspectable and traceable to a ledger event, with privileged teacher information excluded from the model input.

### M4 — LoRA fine-tune

- [ ] Training run on the student base weights.
- [ ] Record hyperparameters, dataset hash, base-model revision.
- [ ] Confirm the tuned student loads and serves behind a route.

Exit: tuned weights exist, are reproducible from recorded inputs, and run inside the harness on the frozen track.

### M5 — Evaluation and the deciding comparison

- [ ] Same track, same tasks, before vs after, on held-out tasks.
- [ ] **Baseline arm**: teacher data from the teacher's own trajectory (off-policy). Same track, tasks, budget and LoRA recipe.
- [ ] **CHTD arm**: teacher data from the student's states (on-policy).
- [ ] Harness-stripping tests: different preset, no tools, single turn (§3.5 of the design).

Exit: a result that either supports or refutes the claim, with the arm comparison and the stripping ladder attached.

### M6 — Distractor analysis and write-up

- [ ] Correlation check on each candidate distractor.
- [ ] Decompose the result into capability / distractor / transfer / regression groups.
- [ ] Strip non-causal co-movers from the report; keep causal ones with a stated reason.
- [ ] Write the result, limitations and reproduction steps.

Exit: a reader can rerun the pipeline and reach the same numbers.

## 2. Deliverables

| # | Artifact | Location |
| --- | --- | --- |
| 1 | Frozen preset + freeze gate | `harness/presets/`, `harness/freeze.py` |
| 2 | Route configuration and rollout collection | `harness/` |
| 3 | Teacher-at-student-states dataset builder | `distill/` |
| 4 | Training scripts and configs | `distill/` |
| 5 | Evaluation, arm comparison, stripping ladder | `eval/` |
| 6 | Result write-up with reproduction steps | `docs/` |

## 3. Risks

**Compute is the top risk.** If no suitable GPU is available, L2 cannot be delivered and the project falls back to L1 while stating plainly that the weight-level claim was not tested.

**Alignment difficulty.** Teacher and student may diverge so early that few usable decision points exist. Mitigation: choose tasks with a structured early phase, and measure the divergence distribution before committing to a dataset size.

**Signal quality.** A divergence is not automatically a place where the teacher is right. Without the outcome gate, the dataset teaches the teacher's quirks along with its capability.

**Scaffold overfit.** A student can learn to behave well inside this harness without acquiring the capability. The stripping ladder in M5 is the guard; if the gain does not survive it, that is the finding.

**The baseline may win.** If off-policy imitation matches the on-policy arm, co-harness is a control rather than a contribution. The plan must be able to report that outcome as a result.

**Single-capability scope.** Proving a transfer on one capability says nothing about others. That limitation belongs in the write-up, not in a hedge.

## 4. What "done" means

The experiment is done when a reader can, from this repository alone: rebuild the frozen track, regenerate the dataset, reproduce the fine-tune, rerun both arms and the stripping ladder, and get the same numbers — including the negative ones.
