# Control experiment audit — 2026-10-09

## Question

Does a student trained on teacher decisions at the student's reached states outperform
the same base student trained on teacher decisions from the teacher's own trajectories?
The two adapters must use the same base weights, LoRA recipe, training task set, frozen
AGH track, step budget, and held-out evaluation tasks. Only the source of training
states may differ.

## Verified in this checkout and on the training box

- The 29 task fixtures pass `python harness/tasks/build_workspaces.py --check-only`:
  each broken fixture fails its check and its known repair passes.
- The Python test suite passes after replacing a host-dependent CUDA assumption in
  `tests/test_train.py` with a simulated no-CUDA runtime.
- The training box has a 48 GB-class RTX 4090 and the `chtd-4` adapter file. Its
  SHA-256 matches `runs/chtd-4/README.md`:
  `f5ebe23860d90bd615d0d7319d229c93784af30302cefab067aedb3160754bb0`.
- The model endpoint reports that it has loaded `chtd-4`.
- The box has CHTD SFT sample files with 1, 12, and 19 rows. No baseline SFT
  samples or baseline adapter were found there.
- The committed five-task holdout file had checker commands pointing at the machine
  that generated it. `harness.round.load_round_tasks` now resolves those checks to
  this checkout and verifies that the expected phrases have not changed.

## What has not been measured

The repository's `0/5` base-versus-`2/5` tuned comparison is a recorded observation,
not a rerun here. It is also **base weights versus CHTD**, not the deciding
teacher-own-trajectory versus CHTD control. The original AGH session ledgers,
per-run outcomes, and the teacher-own-trajectory dataset were not found in this
checkout or the inspected training-box directories. The five-task phrase checker has known false
negatives, so its score needs manual review with the full responses.

No new model-performance lift should be reported until both adapters have been
trained and evaluated from those records under matched conditions. The test suite
checks the comparison code; synthetic test outcomes are not experimental data.

## Competition rule to resolve

The current official rule 3.2 at https://agnes-ai.cn/hackathon limits **all** model
calls to Agnes models. Existing student calls use Qwen, so this run cannot by itself
establish compliance for a competition submission. The former statement in
`docs/agnes-integration.md` that third-party models are allowed has been corrected.

## Inputs required for the actual control

1. AGH `sessions.db`, rollout `records.jsonl`, and observed `outcomes.jsonl` for
   the training and held-out tasks, or access to the AGH host to recollect them.
2. A successful teacher-own-trajectory dataset, matched in task coverage and
   admitted sample budget to the CHTD dataset.
3. Two independently trained adapters from the same pinned Agnes student base,
   same recipe and seed, then AGH runs on the same untouched held-out tasks.
4. A reviewed outcome table, including normal, boundary, and failure cases.

This file records readiness, not a positive or negative CHTD result.
