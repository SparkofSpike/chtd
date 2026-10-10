# Cloud storage for the 2026-10-10 control pilot

The canonical cloud location is `/mnt/proj/chtd-experiments/` on the training
account. It contains:

| Path under that root | Contents |
| --- | --- |
| `baseline-20261010-a/` | Collection manifest, SQLite ledger, 24 task records, manual review, 76 candidate samples, training plan, preflight, 30-step trainer log and run record, LoRA adapter and tokenizer files. The `raw-rollouts/` subdirectory preserves all 76 local collector files. |
| `models/Qwen3-4B-Instruct-2507/` | Frozen local Qwen base-model snapshot used for the run. |
| `source-1e08d66.tar` and `source-1e08d66/` | Git archive and extracted source at fork-PR commit `1e08d66`. |
| `student-server.log` | Current unadapted student endpoint log. |

The copied LoRA weights hash to
`f119c29e3dcf972ac3ab47ecc533301bcdf5e33a7f2b8aa31c7e5a739a661eb2`.
The three copied base weight files hash to
`75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`,
`0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`,
and `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`.
The source archive SHA256 is
`e5b4d0b55a550486b15eb68cd5f495568d3bfd1a865acb17a7100d6aea507054`.
The raw-rollout archive at `baseline-20261010-a/raw-rollouts.tar` hashes to
`9b16d70227447b2e4ba91892718a232ab3da1e8d50e90d673c57e504338ea941`.
All of these hashes were checked on `/mnt/proj` after copying.

The unadapted student server now loads the model and code from `/mnt/proj` and
listens on port 8000. The historical training plan and run record retain their
original `/root` execution paths; the artifact above is the verified saved copy.
The `/mnt/user/chtd-control-backup/baseline-20261010-a/` duplicate was kept.

This is an exploratory training-path result. The CHTD arm has not been paired,
and the five holdout tasks have not been evaluated. No competition credential
is stored in these artifacts.
