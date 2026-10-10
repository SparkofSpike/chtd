# After-turn recovery pilot, 2026-10-10

## Scope and audit trail

The frozen `minimal-rl` AGH track was `d403e1e12186a4021cc9d20bbc2c8770cf8366e9fccfc67621a64fde307967ba`. The base model was Qwen3-4B-Instruct-2507. The 24 teacher-own-trajectory training tasks yielded 18 manually accepted tasks and 76 candidate labels. The base student then ran all 24 training tasks on the same track; its `d` batch passed 17 keyword checks. The 24 after-turn fork records were reviewed individually. Seven student failures produced three teacher-win branches, but only the aligned `fix-type-error` teacher action passed the label review. The other two teacher-win branches had an incorrect intermediate statement or a non-diagnostic aligned action.

AGH supported forks at completed `turn/end` boundaries for this run. The teacher answered a continuation from a student-reached, completed-turn state. This is **after-turn recovery**, not a fork before every student decision inside a turn.

The selection manifest was frozen before opening the five holdout tasks. It selected one final assistant label from each arm for `fix-type-error`. Both adapters started from the same model fingerprint and used the same 30 optimizer updates, seed 0, rank 32, alpha 64, dropout 0.05, learning rate 2e-4, and source commit `3799db5`. The baseline example had 262 supervised tokens and the CHTD example 381, so the supervised-token budget was **not** matched. The source states also differ between the teacher-own and student-reached trajectories. This pilot cannot establish a clean causal effect of CHTD.

The untouched holdout tasks were `fix-ordinal-index`, `fix-shallow-copy`, `fix-single-name`, `fix-sort-key`, and `fix-string-case`. The same frozen AGH track and fresh workspaces were used for the unadapted base, baseline adapter, and CHTD adapter, with one service active at a time. The automatic phrase check is a diagnostic only; it has known false positives and negatives and requires manual review of the actual diagnosis.

## Holdout results

The phrase checker only searches for expected words. The manual column below counts an answer when it identifies the actual Python file/function and the causal defect from the fresh fixture. This is a single-reviewer, non-blinded reading of the final assistant message, not a pre-registered blind adjudication. `Y/N` means phrase check / core diagnosis.

| Holdout task | Base | Teacher-own adapter | After-turn CHTD adapter | Review note |
| --- | --- | --- | --- | --- |
| `fix-string-case` | Y/Y | Y/N | Y/Y | Teacher-own adapter invented JS API files; base and CHTD found the exact-match bug. |
| `fix-sort-key` | Y/Y | Y/N | N/N | Teacher-own adapter invented a JS contract; CHTD did not inspect `sorting.py`. Base named the bad `key=len(words)` expression but called it a syntax error. |
| `fix-shallow-copy` | N/Y | Y/N | Y/Y | The base described the in-place row mutation without the checker's `mutat` substring. Teacher-own adapter discussed a nonexistent JS file. |
| `fix-ordinal-index` | Y/Y | Y/N | Y/Y | CHTD named the off-by-one indexing bug and fix, but incorrectly said position 1 itself raises `IndexError`. Teacher-own adapter described missing JS paths. |
| `fix-single-name` | N/Y | N/N | N/Y | Both base and CHTD identified the unconditional `parts[1]` access; neither said the checker's word `length`. Teacher-own adapter described missing JS paths. |
| **Total** | **3/5 / 5/5** | **4/5 / 0/5** | **3/5 / 4/5** | Manual totals count core diagnosis, with the noted inaccuracies. |

All 15 runs completed and recorded the same track digest. The three services were loaded sequentially. The observed ranking depends on the review rubric and tiny five-task holdout. Training loss is not a task-success metric. This run offers no evidence that CHTD improves on the unadapted base; the single-example teacher-own adapter degraded badly, and the CHTD adapter also missed one task.

## Artifacts

The cloud experiment root is `/mnt/proj/chtd-experiments/`. Relevant directories and archives are `baseline-20261010-a/`, `baseline-20261010-b.zip`, `student-20261010-d/`, `forks-20261010-d-reviewed/`, `paired-20261010-d-reviewed/`, and `matched-20261010-a/`. The two pilot adapters and their `run.json` records are under `matched-20261010-a/baseline-adapter/` and `matched-20261010-a/chtd-adapter/`. The selection manifest, exact sample JSONL files, and validation records are in `matched-20261010-a.zip` (SHA256 `33f995f43381592e2b1113f930c69867857b99b14e9dc8b026befb1ec8d5eb56`). Model weights and credentials are not committed to Git.

| Raw artifact under cloud root | SHA256 |
| --- | --- |
| `baseline-20261010-b.zip` | `cb374ae93bb1b8ad37c0d60004c5b8c37c137e04039d417c8748bdb7e1182ae6` |
| `eval-base-20261010-a.zip` | `6521f9313675c937bdf20bcaf09579641d9b6111bfc687e19f624475221a328a` |
| `eval-baseline-20261010-a.zip` | `3f8747023338161692d0f81562f77c317f016d23deac785463a9c06cde29b857` |
| `eval-chtd-20261010-a.zip` | `9ab8db16610c6d42411efe3144bff104d9636e8856b99be558997feb322300cb` |
| `eval-20261010-a-sessions.db` | `df4e51bc4386a0634fc81153d907397e62ab8f7ff0279b22f9ee210db6671750` |
| `eval-review-20261010-a.json` | `29dd2e6cc2c61917c8fdc6b190a70729d394cc35848bdfe98e5793a98b659b2f` |

The SQLite file is a consistent backup of the AGH session ledger (10,181 events), including the evaluation session keys referenced by the review JSON. The three evaluation manifests agree on fixture hashes, AGH commit, profile hash, and the observed track. All five tasks in each arm have a recorded final message. Local credential scans of the archives and ledger found no Agnes Key bytes before upload.

## Interpretation

This is a one-task training pilot and five-task holdout, not the prespecified strict comparison in `docs/control-experiment-protocol.md`. More reviewed teacher actions on student-reached states, a matched supervised-token budget, multiple seeds, and a second AGH preset would be required for a defensible general claim. The Qwen student route is separate from the event's Agnes-model-use requirement; competition eligibility must be checked independently.
