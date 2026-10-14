# Run `holdout-4-20261014` — the chtd-seed0 "directory does not exist" answers, traced

The observation this record exists for: in the holdout-4 sweep, CHTD seed 0 answered that the
requested directory (or `src/`) does not exist on 29 of its 48 cells, while its session logs show
the tool calls executing normally — a typical session makes 7 tool calls (14 call+result events),
all returning well-formed results. The gate scored that arm 7/48 against 34/48 for both controls.
This record traces the mechanism from the session ledger and states what the numbers can and
cannot mean.

**The finding, stated first.** The seed-0 adapter never lists the workspace. It opens every
session with `find test*.py` and assumes a `src/` project layout that does not exist in the
fixtures; 96 of its 109 `read` calls fail with `ENOENT`, and only 13 of 48 sessions ever
successfully read a file. The "directory does not exist" answers are the model reporting its own
failed exploration, not a missing directory. Every one of the 7 passes is inside the 13 sessions
that did read; all 41 sessions that never read failed. This is a policy regression under a
changed prompt, not a diagnostic-capability result.

## What the records show

From the private archive `holdout-4-20261014-v3-final` (`sessions.db` + per-arm
`records.jsonl`; the census is checked in as `tool-call-census.json`):

| arm | sessions | gate passes | first tool call | read ok / error | sessions that ever read | answers saying "directory does not exist" |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| **chtd-seed0** | 48 | **7** | `find {"pattern":"test*.py"}` ×48 | **13 / 96** (12%) | **13 / 48** | **29** |
| chtd-seed1 | 33 | 25 | `ls {"path":"."}` ×33 | 37 / 0 | 33 / 33 | 0 |
| base | 48 | 34 | `ls {"path":"."}` ×48 | 48 / 0 | 48 / 48 | 0 |
| control-seed0 | 48 | 34 | `ls {"path":"."}` ×48 | 48 / 0 | 48 / 48 | 0 |

(seed1 covers 33 of 48 task ids in this archive; the remaining 15 rows are the separately
recorded resume shard, as the audit notes.)

The within-seed0 cross-tab, which is the load-bearing part:

- `passed` ∩ `ever_read` = **7** — every pass is a session that read a file.
- `passed` − `ever_read` = **0**.
- `ever_read` − `passed` = 6 — reading was necessary but not sufficient.

A representative session (`h4-unicode-nfc-1`, 20.5 s): `find test*.py` → "no matches";
`grep assert ./` → "no matches"; `read ./src/evaluate.py` → `ENOENT ...\student\src\evaluate.py`;
`read ./src/utils.py` → `ENOENT`; `find evaluate.py ./src` and `find *.py ./src` → "no matches
[1 path(s) could not be read]"; `ls ./src` → `ENOENT`. The final answer: "The requested directory
does not exist, so I cannot inspect the project files…". It never ran `ls` on the workspace root.
The fixture for that task is flat — `check.py` and `lib_unicode_nfc.py` — with no `src/` at all.

## The mechanism

Three facts, each checkable:

**1. The first move is a seed-0-only policy.** All 48 seed-0 sessions open with
`find "test*.py"`; every other arm — including the untrained base — opens with `ls "."`.
The seed-0 adapter also references `src/` paths in 48 of 48 sessions. Nothing in the fixtures or
the task file suggests a `src/` layout; the fixtures are flat single-module workspaces.

**2. The training data contains neither behavior.** All 93 `samples-v5.jsonl` rows (and all 9
`teacher-state` rows) open with `ls {"path":"."}` (87×, plus `./` and empty-path variants); the
string `find` appears **0** times across every tool call, and `src/` appears **0** times. The
teacher demonstrations are consistent: list the directory, then read the files it contains, then
diagnose. The seed-0 policy is therefore not learned from the demonstrations — it is the adapter's
own regression, present at this seed and not at seed 1 (nor in the base).

**3. The prompt changed between holdout-3 and holdout-4.**

- Training and holdout-3 prompt: "Something in this directory is broken. Read the files and tell
  me what the defect is. Do not run any commands and do not change any files…"
- Holdout-4 prompt: "A user reports that this small utility gives an incorrect result on the
  documented inputs. Inspect the project and explain the root cause. Do not use a terminal or edit
  files. End with one final line shaped as: `DEFECT: <relative-file>::<symbol> | MECHANISM: …`."

The same seed-0 adapter has no recorded prompt failure of this kind on holdout-3 — there the CHTD
seeds read normally (61 read calls per 30 tasks at each seed, gate 23/22/21; see
`runs/holdout-3-20261014/` and `runs/teacher-state-20261014/README.md`) — while under the
holdout-4 prompt seed 0 degrades. One caveat, consistent with the audit: the holdout-4 records do
not pin adapter artifact hashes, so the identity of "chtd-seed0" across the two sweeps rests on
the arm labels, not on verified weights.

The comparison available here cannot separate "the prompt wording" from "the required output
format" from "the new task family" as the trigger; what it establishes is that the seed-0 policy
is fragile to a prompt change that leaves every other arm — trained or not — unharmed.

**Failure to recover.** The seed-0 sessions do eventually try `ls` (31 of 48), but only after the
guessed `src/` reads fail, and the final answer still reports the directory as missing rather
than retrying the root listing. Failing to recover from an unexpected tool result is the opposite
of the error-recovery capability this project exists to measure.

## What this is not

- **Not a capability measurement.** Seed 0 answered 7 of the 13 tasks where it actually read the
  files (~54%), and its `read` behavior on the earlier holdout-3 prompt was normal. The 7/48 is
  dominated by exploration failure, not by diagnosis quality; both variables are entangled in
  this number.
- **Not evidence about CHTD's data by itself.** Both controls use the same architecture, the same
  9 rows, and the same recipe at other seeds, and neither shows the behavior. The cross-seed
  spread (7/48 vs 38/48) is a property of this training regime's stability, not a graded
  difference between data sources.
- **Not a harness defect.** Tool results were well-formed and correctly reported `ENOENT`; the
  workspace contained exactly the task files. (The `check.py` exposure that earlier holdout-4
  runs had is a separate, already-fixed issue — see `reports/EXPERIMENT_AUDIT.md`.)

## Relation to earlier findings

This is the third recorded instance of the same family, and it should be read with them:

1. `docs/why-finetuning-can-hurt.md` §3 — the control arm "invented `src/student/src` paths and
   JS contracts that do not exist in the fixtures", and the 4-bit collapse looped on 89
   `web_fetch` calls against a non-existent URL. The document already says: failure by *looping*
   rather than by *being wrong* "stays unexplained and recorded as such".
2. `runs/teacher-state-20261014/README.md` — the teacher-state arm dropped to 11 reads/30 tasks
   at one seed with 25/30 sessions never reading, and "invented file contents". The conclusion
   there: the same architecture that is stable on student-state rows is unstable on teacher-state
   rows.
3. This record — the student-state shape itself, at one seed, under a prompt shift.

Common root across all three: nine rows with `supervise='last'` teach the final answer, not the
exploration policy; the policy is inherited unstably, and what the model does with an unexpected
tool result is outside the training signal entirely.

## Consequences for reading the holdout-4 numbers

- The seed-0 column of the reported table measures exploration robustness under a new prompt as
  much as it measures diagnosis. Comparing it against seed 1 (same data, same recipe, different
  seed) or against the controls as a capability difference is not supported by these records.
- A clean diagnostic comparison would hold the prompt fixed to the training distribution, or
  report "did the session read the fixture" as a separate, first-class metric alongside the gate —
  it is exact, mechanical, and independent of answer prose.
- The failed-exploration mode converts directly into gate failures, so any future sweep should
  record per-session read health (this census is the template) next to the pass count.

## Re-derivation

The private archive keeps `sessions.db` and the per-arm `records.jsonl`. The census in
`tool-call-census.json` was produced by reading, for each arm, every `tool/call` and `tool/result`
event from `events` for the recorded `session_key`s, and classifying: first call; `read` results
by `isError`; whether any session ever produced a non-error `read`; whether any call argument
contains `src/`; and whether the final answer matches /directory does not exist/i. The training
claim (all-`ls` first moves, zero `find`, zero `src/`) was verified over
`teacher-demo-20261012/samples-v5.jsonl` (93 rows) and
`node4-recollection-20261012/prepared/matched-v2/teacher-state-samples.jsonl` (9 rows).

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
