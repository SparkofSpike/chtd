# Run `teacher-demo-20261012` — the arm that beats its base

The first fine-tune in this repository that scores **above the unadapted base model** on the
held-out set, and the first that solves tasks the base cannot solve at all.

```
arm                                    held-out score (six sweeps)
base    unadapted                      13/15, 13/15, 13/15, 13/15, 13/15, 13/15
v1      33 demonstrations, 1/task      13/15
v2      63 demonstrations, multi/task  14/15, 15/15, 15/15, 15/15
v3      83 demonstrations, multi/task  12/15, 15/15, 14/15
```

The base's six sweeps are identical to the task — the same two failures every time
(`fix-none-cap`, `fix-or-default-falsy`) — so the comparison is not against a noisy control.
`v2` wins every sweep it appears in, and its single 14 is a scorer false negative (below).

## What the data is

**Complete teacher demonstrations, not student forks.** Every training row is an AGH session the
teacher (`agnes-3.0-flash`) held from the task prompt to its own diagnosis: its own `ls`, its own
`read`s, its own final answer. No student turns are in the data, so `supervise='assistant'` is
legal under the provenance gate that PR #3 added — every supervised turn is the teacher's own.

That is a deliberate departure from the design's letter, and it is the change that works. The
design's shapes measured before this were: student prefix + teacher tail with `last` supervision
(the 0/15 arm), and student prefix + mixed turns with `assistant` supervision (13/15). A model
shown only "the teacher's closing sentence after a conversation somebody else conducted" learned
to answer and not to look; a model shown the teacher *doing the whole task* learns the whole task.

Collection ran in three passes:

| pass | how | usable traces |
| --- | --- | --- |
| a | serial, one shared home (the round shape) | 33 / 35 |
| d | parallel, one isolated home per task | 25 / 35 |
| e | parallel, one isolated home per task | 20 / 35 |

A trace is usable only when **all three** hold: the diagnosis passes `check_diagnosis`, no tool
call was refused or failed, and at least one `read` succeeded. The middle condition is not
pedantry: the teacher's first move is sampled, and on some attempts it guesses an absolute path,
is refused, and retries — twelve of the first pass's traces did this. A trace that demonstrates
"try an outside directory first" teaches the habit the project's own `WorkspaceRelativePaths` gate
exists to refuse, so those traces are dropped rather than repaired.

The gate's own two failures (wrong diagnoses) also drop traces: `fix-assertion`,
`fix-argument-order`, `fix-cli-flag`, `fix-empty-input`, `fix-integer-division`, `fix-path` were
excluded on those grounds in the first pass.

## What ran

```
arm        base Qwen3-4B-Instruct-2507, 4-bit NF4, LoRA on all seven projections
           rank 32, alpha 64, dropout 0.05, lr 1e-4, accumulation 4, 10 epochs, seed 0
v2         63 rows over 34 tasks (26 tasks carry 2–3 differently worded demonstrations)
           160 steps, loss 0.2957, ~15 min
v3         83 rows over 34 tasks, 210 steps, loss 0.2831, ~19 min
```

Adapters (on the box, under `/mnt/proj/chtd-experiments/teacher-demo-20261012/`):
`adapter` (v1), `adapter-v2`, `adapter-v3`.

## The two tasks that moved

`fix-none-cap` and `fix-or-default-falsy` are the base's permanent failures — machine-checked
verdict "does not name the defect", six sweeps running. Both are classic Python traps that reward
reading the implementation and simulating it:

* `limits.py`: `min(value, cap)` with `cap=None` raises `TypeError`; the docstring promises the
  cap is optional.
* `ports.py`: `settings.get("port") or 8080` returns 8080 for a configured port of `0`.

The base, on both, diagnoses the *assertion's wording* instead of the defect — it reads
`assert limit(5) == 5, limit(5)` and concludes the error message is unhelpful. That is the
failure mode the collected traces never demonstrate; the teacher, on both, reads the
implementation and traces the value. v2 and v3 solve both. The capability existed in the teacher
and transferred — which is the claim the method exists to make.

## The one 14, read honestly

`v2`'s first sweep scored 14/15 with `fix-single-name` marked wrong by the phrase check. The
answer says:

> `full.split()` returns one fewer element than the tuple it returns. If the name has only one
> word (like `'Ada'`), `parts[1]` raises an `IndexError`.

The task's phrase set is `two part|one part|one-word|single`, and "one word" is none of those
spellings. The diagnosis is correct — and it is the same class of phrase-check miss that
`docs/finetune.md` already records for this task ("marked wrong by the phrase check while its
answer names the defect"). Subsequent sweeps pass it. The score of record is the machine's; the
reading is that this arm is 15/15 on this task set by any human standard.

## What this does and does not establish

**Establishes.** On this 15-task set, with this scorer, a fine-tune of this base on 63 clean
teacher demonstrations scores above the base in four consecutive sweeps, and it does so by
solving the two tasks the base never solved. The base's failures are identical across six
sweeps, so the delta is not control noise.

**Does not establish.** A general capability gain, or a CHTD-versus-teacher-own result. The 15
tasks have been inspected in earlier rounds — they are a diagnostic screen, not untouched
holdout — and one task's phrase check is known to miss correct answers. v3's 12/15 sweep shows
adapter scores fluctuate (three points) in a way the base's do not; the honest summary is
"v2 ≥ base + 1 in every sweep, typically +2, on a screen that sometimes mis-scores wording".

**Why v3 is not simply "more is better".** v3 saw 20 more traces than v2 and scored lower in two
of three sweeps. The difference between the runs is only the extra pass-e traces, and this note
does not claim to know which of them cost what — with 15 tasks, one task is 6.7 points, and
adapter-side run-to-run variance is itself a finding: an adapter near the ceiling behaves
less predictably than a base far from it.

## Reproducing

```sh
# on the box, with the v2 sources under source-7087518/ and the samples uploaded:
python -m distill.finetune \
  --plan    /mnt/proj/chtd-experiments/teacher-demo-20261012/plan-v2.json \
  --samples /mnt/proj/chtd-experiments/teacher-demo-20261012/samples-v2.jsonl \
  --base-model /mnt/proj/chtd-experiments/models/Qwen3-4B-Instruct-2507 \
  --output  /mnt/proj/chtd-experiments/teacher-demo-20261012/adapter-v2 \
  --supervise assistant

# three endpoints (v2:8002, base:8003, v3:8004), then from the client:
CHTD_ARMS=v2:8002,base:8003,v3:8004 python eval/holdout_sweep.py 3arm-<date>
```

## Artefacts

```
E:\Coding\chtd\data\teacher-round-20261012\      (gitignored: the raw collection)
  agh_home/            the shared-home ledger (pass a + retries)
  iso-passd/, iso-passe/   per-task homes from the isolated passes
  records*/            per-trace records (session, workspace, audit, gate verdict)
  trace-eval*.json     the audit: refused counts, reads, gate exit per trace
  built/samples-v2.jsonl   the 63 rows
  built/samples-v3.jsonl   the 83 rows
  built/plan-v2.json, plan-v3.json

E:\Coding\chtd\data\rollouts\                       (gitignored: the sweeps)
  teacher-demo-20261012-bf16/    v1 vs base
  teacher-demo-v2-20261012-bf16/ v2 vs base, sweep 1
  teacher-demo-v2b-20261012-bf16/ v2 vs base, sweep 2
  teacher-demo-v3-20261012-bf16/ v3 vs base
  3arm-20261012-a/, 3arm-20261012-b/   v2 + base + v3, same condition

/mnt/proj/chtd-experiments/teacher-demo-20261012/
  adapter/, adapter-v2/, adapter-v3/   the weights
  samples-v2.jsonl, plan-v2.json, samples-v3.jsonl, plan-v3.json
  train.log, train-v2.log, train-v3.log
```

## One implementation fix this depended on

`distill/finetune.message_bounds` refused every trace whose teacher read two files in one turn:
the template closes a run of tool results only at its last message, so a render that stops inside
the run is not a prefix of the full render, and the old adjacent-nesting check treated that as
drift. 32 of the first 33 collected traces were of that shape. The fix places each prefix
against the *full* render — the property the mask actually needs — and carries the one interior
boundary (never read by the mask) forward as a placeholder. Committed as `f3164a8`, with the
template's behaviour reproduced in the test fake so the regression is caught without a GPU.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
