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

- [x] For each student decision point, fork the session at that boundary and have the teacher answer there. Implemented, tested, **and exercised on real rollouts**: the teacher's session was forked into the student's state and the divergence that produced `data/datasets/official-1` came from a real fork, not a fixture.
- [x] Extract divergences; emit `(state, teacher decision)` pairs. `distill/dataset.py` emits them with a digest and a manifest.
- [x] Apply the signal-quality gate: keep only divergences whose teacher choice the outcome validates. `distill/gate.py`; fail-closed, and every refusal is recorded with its reason.
- [x] Sanity-check the set, to the extent one row can be checked. The single admitted divergence is legible: the student answered "Hello! I can help" where the teacher read the file, which is the target capability's shape (the student did not recognise a tool call as the move) rather than noise. **One row is not a distribution** — whether divergences *typically* reflect the capability is a question only a larger set can answer, and the collection plan for one is in [finetune.md](finetune.md).

Exit: a dataset whose entries are individually inspectable and traceable to a ledger event, with privileged teacher information excluded from the model input. **Met**: an entry carries no field that could hold gate evidence, a test asserts the key set, and the one real entry is traceable to `student_seq: 17` / `teacher_seq: 17` on `prefix_digest 935ef0d4…` in the shared ledger.

### M4 — LoRA fine-tune

- [x] Training run on the student base weights. Ran on the training box 2026-10-09: Qwen3-4B-Instruct-2507 in 4-bit, LoRA on q/k/v/o_proj, 11,796,480 trainable of 2,217,606,656 (0.532%), 3 steps, mean loss 4.1219, 8.9 s wall clock, 17.79 GB peak. The run and its numbers are in `runs/chtd-1/`. **What it does not show:** one sample and one effective optimizer step cannot support any claim about learned behaviour; the tuned student's answer is wrong in the same way the untuned one was. The dataset is a smoke test until §"What would have to happen" in [finetune.md](finetune.md) is carried out. **Nor is it bit-reproducible:** the same plan run twice produced two adapters, so what a reader checks is the protocol, not the bytes.
- [x] Record hyperparameters, dataset hash, base-model revision. `distill/train.py` freezes the recipe with the plan, and the plan names the dataset digest and the base revision. Two recipe fields were set against a measurement rather than left at their defaults — `max_sequence_length: 8192` because the sample renders to 6715 tokens and truncation would remove the label, and `gradient_accumulation: 1` because one sample with the default of 8 never steps the optimizer. Both are recorded in the plan and explained in `runs/chtd-1/README.md`.
- [x] Confirm the tuned student loads and serves behind a route. The box's endpoint runs with `--adapter /root/chtd_runs/chtd-1`; `/health` reports the adapter path, and two student rollouts through the harness ran under `preset: chtd` on track `a37b90fe…`, one against each of the two adapters. The adapters' sha256 values are recorded in `runs/chtd-1/README.md`.

Exit: tuned weights exist, are reproducible from recorded inputs, and run inside the harness on the frozen track. **Met.** Reproducing them starts from the plan and the samples (`distill/finetune.py --plan … --samples …`), which are themselves regenerated from the ledger and the captured prefix.

A fourth item was needed and is recorded here because it is not optional: **the harness prefix had to be captured rather than reconstructed.** The system prompt is assembled from several packages and the tool disclosure comes from the loaded assembly — measured at 22 tools where the preset's `NOT CONSUMED` line names two. `harness/prefix/capture-1.json` holds the bytes a live run sent, and `distill/finetune.py` refuses a sample whose prompt is not a token-for-token prefix of its rendering. See [finetune.md](finetune.md).

### M5 — Evaluation and the deciding comparison

- [ ] Same track, same tasks, before vs after, on held-out tasks.
- [x] **Baseline arm**: teacher data from the teacher's own trajectory (off-policy). Same track, tasks, budget and LoRA recipe. `eval/arms.py` refuses a comparison whose arms differ on any of those four.
- [x] **CHTD arm**: teacher data from the student's states (on-policy). `distill/divergence.py` pairs on state, so this is what the pairing means.
- [x] Harness-stripping **assessment**: a ladder that does not survive stripping is reported as `scaffold-overfit`, and a ladder with no stripped rung is refused rather than called survived. `eval/stripping.py`.
- [ ] Harness-stripping **execution**: run the same tasks under a different preset, then with no tools, then in a single turn. Nothing yet produces a rung — `harness/rollout.py`'s runner takes a route and a model but no preset and no tool or turn limit — so `eval/stripping.py` currently assesses a ladder somebody else's run would supply. This is the gap between the two bullets, and it is a gap. See `docs/pipeline.md`.

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
