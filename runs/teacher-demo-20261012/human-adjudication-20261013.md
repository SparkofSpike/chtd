# Human adjudication, 2026-10-13 — what the phrase gate counted vs what the answers say

The claim this repository has been building toward is "the fine-tuned arm beats its base". On
2026-10-12 the phrase gate said it did: `v4` scored 15/15, 15/15, 14/15 against a base of 13/15
across ten sweeps. This note is what happened when every answer behind those numbers was read by
hand.

**The short version: on human reading, the fine-tuned arm and the base are level.** The gate's
lead for the adapter is mostly the gate's own error. This is a correction of this run's earlier
record, and it is recorded here rather than quietly deleted.

## Method

Two task sets:

- **The original 15 held-out tasks** (`harness/tasks/final-holdout.jsonl`), which the adapter's
  training data covers in kind — the same defect families appear in the 34 training tasks.
- **A fresh 12-task set** built for this check (`data/holdout-2-20261013/`), of defect types the
  training data does *not* contain: banker's rounding, `str.strip` as a character set,
  `os.path.join` with an absolute right side, deleting from a dict while iterating its view,
  `finally: return`, `//` versus truncation toward zero, and so on. Teacher calibration: the
  teacher solves 11 of 12 (the exception, `h2-unbound-local`, is a prompt-ambiguity case and is
  noted, not hidden).

For every answer, the arm's **final message** was read in full and judged on one question: does
it name the actual mechanism the fixture contains? Three grades:

- **correct** — names the mechanism; a reader could fix the defect from it.
- **partial** — right location and mechanism in outline, with a material detail wrong or an
  ambiguous verdict.
- **failed** — wrong location, wrong mechanism, or no verdict.

The gate was run unchanged alongside, so the two readings can be compared per task.

## Original 15 tasks — one sweep, both arms, same conditions (`4arm-20261012-a`)

```
task                        gate(base/v4)   human base        human v4
fix-string-case             pass / pass     correct           correct
fix-sort-key                pass / pass     correct           partial   (says key=len(words) returns a list)
fix-shallow-copy            pass / pass     partial           partial   (v4 says the mutation lands after the assert)
fix-ordinal-index           pass / pass     correct           correct
fix-single-name             pass / pass     correct           correct
fix-strip-whitespace        pass / pass     correct           correct
fix-closure-late-binding    pass / pass     correct           correct
fix-generator-exhaust       pass / pass     correct           FAILED    (v4 claims doubled() returns a tuple)
fix-broad-except            pass / pass     correct           correct
fix-none-cap                FAIL / pass     FAILED            partial   (v4 names min(5,None) -> TypeError)
fix-enumerate-start         pass / pass     partial           correct
fix-zip-truncates           pass / pass     partial           correct
fix-or-default-falsy        FAIL / pass     FAILED            partial   (conclusion right, one sentence self-contradicts)
fix-row-alias               pass / pass     correct           correct
fix-augmented-assign        pass / pass     correct           correct
                                      ---------------------------------
                                      11 correct        10 correct
                                      +3 partial        +4 partial
                                      +2 failed         +1 failed
```

**Gate margin: +2 for the adapter. Human reading: −1 for the adapter.** The two readings disagree
in both directions:

- `fix-generator-exhaust`: the adapter's answer says `doubled()` "returns a tuple", which is false
  (`(x * 2 for x in values)` is a generator expression) and even contradicts its own next
  sentence ("happens to work here"); the gate passed it on the words *generator* and *twice*.
- `fix-none-cap`: the adapter correctly names `min(5, None)` raising `TypeError` — a better answer
  than the base's ("limit() is correctly implemented") — and the gate **failed** it, because the
  forbid phrase `there is no` matched the harmless clause "when there is no cap".

## The fresh 12 tasks — same sweep conditions

```
task                        gate(base/v4)   human base        human v4
h2-setdefault-none          FAIL / pass     partial           correct
h2-re-match                 pass / pass     correct           correct
h2-round-half-even          pass / pass     correct           FAILED    (v4: "halves do go up, which is correct")
h2-join-absolute            FAIL / pass     FAILED            FAILED    (both misread os.path.join)
h2-naive-aware              pass / pass     partial           partial   (neither says TypeError)
h2-floor-toward-negative    pass / pass     correct           correct
h2-strip-charset            FAIL / pass     FAILED            FAILED    (both: strip('N/A') removes the substring)
h2-dict-mutation-view       FAIL / pass     FAILED            FAILED    (neither says RuntimeError)
h2-unbound-local            pass / pass     correct           partial   (v4 suggests nonlocal for a module global)
h2-class-attr-shared        pass / pass     correct           correct
h2-finally-return           pass / pass     correct           partial
h2-min-key-string           pass / pass     partial           correct
                                      ---------------------------------
                                      6 correct         5 correct
                                      +3 partial        +3 partial
                                      +3 failed         +4 failed
```

**Gate margin: +3 for the adapter. Human reading: −1.** Four of the adapter's twelve "passes" are
false positives — `round`, `join`, `strip`, `dict` — every one of them a confident answer with the
wrong mechanism:

- `h2-round-half-even`: "halves *do* go up, which is correct" — `round(0.5)` is 0.
- `h2-join-absolute`: claims the join "returns '/srv/www/img/logo.png', which is technically
  correct" — it returns `/img/logo.png`.
- `h2-strip-charset`: claims `strip('N/A')` "leaves it unchanged" — it returns `'pple'`.
- `h2-dict-mutation-view`: claims the loop "skips the next entry" and works — it raises
  `RuntimeError`.

The gate passed all four on generic words (`half`, `absolute`, `characters`, `iterat`). The
answer-length and fluency this arm produces make its prose *more* likely to contain those words
than the base's flatter output — which is the mechanism by which the gate manufactures a margin.

## What survives, stated carefully

1. **The adapter and the base are level on human reading, on both sets** — 10–11 of 15 and 5–6 of
   12, with the same pass bands. The "beats its base" claim from 2026-10-12 does not survive as
   stated and is withdrawn.
2. **One real, narrow improvement.** On value-semantics traps (`min(5, None)`, `0 or 8080`), the
   base fails in every sweep and the adapter gets them right in every sweep. That is 1–2 tasks of
   genuine gain, in the family the training data covers heavily.
3. **One real, narrow regression.** `fix-generator-exhaust` is answered wrongly by the adapter in
   both sweeps read here, correctly by the base in both. Net across the two sets: approximately
   zero.
4. **Shared blind spots.** `join`-absolute, `strip`-as-charset, dict-mutation-during-iteration
   defeat *both* arms — these are base-model weaknesses that nineteen-to-sixty demonstrations of
   other defect types did not transfer to. A result that appears on them is a difference in the
   teacher's coverage, not in the method.
5. **The phrase gate is not a measurement instrument at this margin.** Its errors run both ways
   and are arm-correlated; every comparison in this repository that rests on its counts alone —
   including the adapter scores recorded here on 2026-10-12, and the base's 13/15 — carries the
   same uncertainty.

## What this does not overturn

The infrastructure findings stand on their own and are unaffected: the `message_bounds` fix for
parallel tool calls (`f3164a8`), the collection provenance discipline (gate + zero-refusal +
successful-read), the supervision-mode observation that `lastonly` arms lose tool use entirely,
and the fact that fine-tuning at this data scale neither helps nor hurts much — which is exactly
what `docs/why-finetuning-can-hurt.md` predicts, and this corrected reading is now consistent
with it.

## Artefacts

```
data/holdout-2-20261013/
  tasks.jsonl, workspaces/         the fresh 12 tasks
  teacher-calibration.json         teacher 11/12
  records-{base,v4}.json           the runs
  scores-{teacher,base,v4}.json    the gate's counts
D:\CodeWhaleData\tmp\             the export and adjudication scripts used above
  build_holdout2.py, verify_holdout2.py, run_arm_h2.py, score_holdout2.py,
  export_arm_answers.py, digest_answers.py, judge_answers.py
```

Raw answers per arm per task were read from the ledgers under
`data/rollouts/{4arm-20261012-a,2arm-v4c-20261012}/**/_agh_home/data/sessions.db`, one home per
task-run, as that sweep wrote them.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
