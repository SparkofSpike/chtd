# Reviewed all-tool anchor: exploratory follow-up

This follow-up tests whether supervising only the first `ls` call was too narrow after the first
candidate made an invalid later `read` call. It is an iterative diagnostic chosen after observing
that candidate, so the reused 15-task set cannot provide independent validation.

The same 11 manually accepted teacher final diagnoses are paired with the three student-sourced
tool calls actually observed before each final teacher answer. The 44 rows therefore contain
33 student tool-call labels and 11 teacher diagnosis labels. No student final diagnosis is a target.
Each tool action is its own last-turn sample, with the preceding conversation as context.

The base model and QLoRA parameters match the first-tool run, except 8 epochs are planned instead
of 14. This gives approximately the same number of optimizer updates (88 planned versus 84 in the
first candidate); it is **not** an exact matched-budget method comparison. Plan and generated
sample digests are in this directory; raw samples, logs, and weights stay under
`/mnt/proj/chtd-experiments/aux-all-tools-20261011/`.

The 44-row tokenizer preflight passed with no prefix drift or over-length sample. Training then
completed 88 updates in 486.1 seconds; the final training loss was 0.2172. The exact base weight
digest matches the first candidate (`e36d2f3848f1b704d35f4fa91a4983137448f76d0ad8a22c87378a5b0f655c7b`).
The adapter SHA256 is `08d4c66b3ca005556e3d3768ebec0b90b5542f410de7d3094a8145d871680f9b`;
its weight file stays in the cloud run directory. The full record is `train-run.json`.

The paired serving evaluation completed all 15 tasks on each arm without runtime failures.
The automatic keyword proxy passed 13/15 base answers and 15/15 adapter answers. Raw session
records are archived at `/mnt/proj/chtd-experiments/aux-all-tools-20261011/aux-all-tools-eval-20261011-a.tar.gz`
(SHA256 `6692ad40669e0375e06b442d0c9234a62c0d2f9c8ecb8f93972e7afe21f09741`), and
the small per-task report is `eval-summary.json`.

Manual spot checks found errors despite the 15/15 proxy score. On `fix-none-cap`, the adapter
claims `min(5, None)` returns `None`, although Python raises `TypeError`. On `fix-broad-except`,
it recommends catching `TypeError`, which would preserve the bug of swallowing that exception.
On `fix-string-case`, its suggested assertion change would not repair the case handling.
These tasks were inspected in earlier rounds and this candidate was designed after seeing the
first candidate fail; the result is an iterative diagnostic, not an independent holdout score or
a controlled estimate of a CHTD method effect. Further gains require a fresh task set and an
executable or human-checked correctness measure.
