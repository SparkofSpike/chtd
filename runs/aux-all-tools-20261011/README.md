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

Results will be added after training and serving checks. Any result here is diagnostic only.
