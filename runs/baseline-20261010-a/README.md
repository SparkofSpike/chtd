# Control-arm collection, 2026-10-10

This is the observed teacher-own-trajectory control collection for the paired
CHTD experiment. It is **not a trained adapter or a model-performance result**.

- Tasks: 24 training fixtures from `harness/tasks/error-recovery.jsonl`; the five
  `harness/tasks/final-holdout.jsonl` fixtures were not run during collection.
- Model: `agnes-3.0-flash` through Agnes Harness commit
  `a39272bb6bcb3109584e1bd3abac595789ae365d`, builtin `minimal-rl` preset.
- Observed track digest: `d403e1e12186a4021cc9d20bbc2c8770cf8366e9fccfc67621a64fde307967ba`.
- Automatic phrase check: 21/24. Manual answer and trajectory review: 18 tasks
  admitted. The six exclusions and their reasons are in `review.json`.
- Candidate labels: 76 teacher decisions. Eleven exclusions are recorded in
  `sft/refusals.jsonl` (six rejected tasks and five nonportable tool paths).
- `sft/manifest.json` names the exact candidate sample digest. The selection
  remains unmatched until the experimental arm provides its task coverage and
  admitted sample budget.

`collection/sessions.db` is an SQLite backup of the append-only AGH ledger. It
also contains two earlier pilot sessions; the 24 task records in
`collection/tasks/` identify the sessions used by this collection. Paths in
those records are portable. The complete sample set can be rebuilt with:

```sh
python -m distill.baseline \
  --collection runs/baseline-20261010-a/collection \
  --prefix runs/baseline-20261010-a/prefix.json \
  --review runs/baseline-20261010-a/review.json \
  --out /tmp/baseline-rebuilt
```

The checked-in prefix came from an unscored mock endpoint using the same AGH
commit and builtin preset; the mock changed only the student route URL. It has
11 disclosed tools. The older `harness/prefix/minimal-rl.json` capture had 22,
so it cannot be used for this run. A direct serving-side request capture remains
necessary before claiming token-for-token train/serve parity.

The collection used the `chtd` commit and collector SHA in
`collection/collection-manifest.json`. The Windows stdout decoding fix in
`harness/rollout.py` was applied after collection; it does not change AGH
requests or the ledger, but the emitted stderr log contains GBK decoding
warnings. The task outcomes and candidate builder read the ledger itself.
