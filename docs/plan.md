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

### M1 — Frozen track

- [ ] Author `harness/presets/chtd.yaml`: minimal, explicit, one tool set, fixed step budget.
- [ ] Freeze its hash; add the verification gate (mirroring `minimal-rl`).
- [ ] Prove the gate fails on an edited byte and passes on the frozen file.

Exit: two runs on the same task produce the same preset hash, and an edit is refused.

### M2 — Task driver and rollout collection

- [ ] Drive AGH from its CLI/SDK, one session per task.
- [ ] Collect teacher and student trajectories on the identical frozen track.
- [ ] Confirm trajectories land on the append-only ledger and can be read back whole.

Exit: a pilot task set produces matched teacher/student trajectories that replay.

### M3 — Divergence dataset

- [ ] Align teacher/student steps under a shared prefix.
- [ ] Extract divergence points; emit `(state, teacher decision)` pairs.
- [ ] Sanity-check the set: divergences should reflect the target capability, not noise.

Exit: a dataset whose entries are individually inspectable and traceable to a ledger event.

### M4 — LoRA fine-tune

- [ ] Training run on the student base weights.
- [ ] Record hyperparameters, data hash, base model revision.
- [ ] Confirm the tuned student loads and serves.

Exit: tuned weights exist, are reproducible from recorded inputs, and run inside the harness.

### M5 — Co-harness evaluation

- [ ] Same task set, same frozen track, before vs after.
- [ ] Held-out tasks the training set never contained.
- [ ] Report: target-capability success rate and failure-mode change — nothing else.

Exit: a result that either supports or refutes the approach, with the evidence attached.

### M6 — Distractor analysis and write-up

- [ ] Correlation check on each candidate distractor.
- [ ] Strip non-causal co-movers from the report; keep causal ones with a stated reason.
- [ ] Write the result, limitations and reproduction steps.

Exit: a reader can rerun the pipeline and reach the same numbers.

## 2. Deliverables

| # | Artifact | Location |
| --- | --- | --- |
| 1 | Frozen preset + hash gate | `harness/presets/` |
| 2 | Task driver and rollout collection | `harness/` |
| 3 | Dataset builder | `distill/` |
| 4 | Training scripts and configs | `distill/` |
| 5 | Evaluation and distractor analysis | `eval/` |
| 6 | Result write-up with reproduction steps | `docs/` |

## 3. Risks

**Compute is the top risk.** If no suitable GPU is available, L2 cannot be delivered and the project must fall back to L1 while stating plainly that the weight-level claim was not tested.

**Alignment difficulty.** Teacher and student may diverge so early that few shared prefixes exist. Mitigation: choose tasks with a structured early phase, and measure the divergence distribution before committing to a dataset size.

**Signal quality.** A divergence point is not automatically a place where the teacher is right. Without a filter, the dataset teaches the teacher's quirks along with its capability. Mitigation: keep only divergences where the teacher's choice is validated by the task outcome.

**Single-capability scope.** Proving a transfer on one capability says nothing about others. That limitation belongs in the write-up, not in a hedge.

## 4. What "done" means

The experiment is done when a reader can, from this repository alone: rebuild the frozen track, regenerate the dataset, reproduce the fine-tune, rerun the evaluation, and get the same numbers — including the negative ones.
