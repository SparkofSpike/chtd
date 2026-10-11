# Reviewed first-tool anchor: exploratory run

This run tests a practical repair for the last-turn adapter's loss of tool use. It is an exploratory
model-improvement run, not the controlled CHTD versus teacher-own comparison proposed by the method.

## Training input and provenance

- Source: 13 mixed-history CHTD samples from the `round-20261011` collection. The original raw
  student and fork ledgers for this round were unavailable on the current machine; the sample file
  and plan are preserved under `/mnt/proj/chtd-experiments/round-20261011/`.
- Manual review against fixture defects accepted 11 teacher final diagnoses and rejected two.
  The single-reviewer decisions are recorded in `review.json`.
- Each accepted task contributes two supervised last turns: one **student-sourced** first `ls` tool
  call, and one reviewed **teacher-sourced** final diagnosis after the student-reached history.
  The training file therefore has 22 rows. The tool calls do not become teacher demonstrations.
- The source, review, generated samples, plan, and source code snapshot are frozen by SHA256 in
  the cloud run directory `/mnt/proj/chtd-experiments/aux-first-tool-20261011/`.
- Base: Qwen3-4B-Instruct-2507. QLoRA recipe: rank 32, alpha 64, dropout 0.05, learning rate
  1e-4, batch 1, accumulation 4, maximum length 8192, seed 0, 14 epochs. The actual update count
  and base weight-shard hashes are written by the trainer to `adapter/run.json`.

The 22-row tokenizer preflight passed, with no prefix drift or over-length examples. The cloud
trainer completed **84 updates** in 445.3 seconds with a training loss of 0.2257. This loss is not
an evaluation score. The recorded SHA256 digest of the base weight shards and rendering files is
`e36d2f3848f1b704d35f4fa91a4983137448f76d0ad8a22c87378a5b0f655c7b` (full per-file
hashes in `train-run.json`). The adapter SHA256 is
`bb0388085bcc7e80fdb84d7081288ed4e3bb06bd3553c3c0897b809da0b095b9`; its 264 MB weight
file stays on the cloud under the run directory.

## Diagnostic serving result

The same 15 tasks were run sequentially against the unadapted base and this adapter, using the
same bf16 server implementation and frozen task files. All 30 raw sessions are preserved at
`/mnt/proj/chtd-experiments/aux-first-tool-20261011/eval-20261011-b/`. The corresponding archive
SHA256 is `463fcae2d71e5f8b4786baaf67122c1ed0887398c4eed51e534d66f844276453`.

The **automatic keyword proxy** passed 13/15 for the base and 12/15 for the adapter. The adapter
made a malformed later `read` call on `fix-closure-late-binding` (exit 4), then diagnosed its own
tool error instead of the Python defect. It also hallucinated a circular import on
`fix-enumerate-start`. The compact per-task record is in `eval-summary.json`.

Manual inspection exposed scorer errors in both directions: the adapter correctly diagnosed
`fix-none-cap` but failed the gate because the phrase “there is no” appeared in an explanation;
it passed `fix-or-default-falsy` by naming `0` and `8080` while wrongly advising that the test
expectation be changed. The base passed `fix-enumerate-start` while concluding that the test was
wrong, despite the implementation's default zero-based numbering. Several other answers mix a
correct root cause with inconsistent repair advice. These counts are therefore not reliable human
accuracy estimates. This first candidate has **no demonstrated performance gain**.

## Interpretation limits

The `final-holdout` tasks have already been used in earlier sweeps and are diagnostic only. The
keyword scorer is a proxy: it has both false positives and false negatives. Runtime failures count
as failures, and material claims require review of the final answer against the fixture. Any
observed gain here would need a new untouched test set and a matched teacher-own control before it
could support a CHTD method claim.
