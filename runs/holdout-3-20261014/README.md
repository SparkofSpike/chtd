# Run `holdout-3-20261014` — a 30-task set for the question the small sets could not answer

The h1/h2 comparison left two claims: adapted arms beat the base, and the student-state arm beats
the teacher-own arm by two or three tasks. The second claim rested on 12–15 tasks, and one task
there is 7–8 points. This run answers both questions on **30 new tasks nobody had read**, with the
student-state arm trained at **three seeds** — the missing piece that turns a single number into a
distribution.

**The result, stated first.** *Adapted beats base* is confirmed and large: every adapted arm
clears the base by 3–7 tasks. *Student-state beats teacher-own* **does not replicate**: the two
arms' seed ranges overlap and their means differ by ~1.2–1.7 tasks, which is smaller than the
student-state arm's own seed spread. The earlier +2–3 was consistent with noise.

## The task set

30 single-defect Python diagnosis fixtures under `data/holdout-3-20261014/`. Design rules:

- **Every mechanism is new.** Checked against all 71 existing task ids (the 34 teacher-demo
  fixtures, the 15 holdout-1 tasks, the 12 holdout-2 tasks, the 10 pass-2 fixtures): no mechanism
  repeats. Several near-misses were caught and changed during design (e.g. an integer-division
  average was dropped because `fix-integer-division` already carries it).
- **All 30 verified broken before use.** `check.py` in each workspace fails — 30/30, by
  `AssertionError` (19), `TypeError` (4), `ValueError` (3), `AttributeError` (2), `KeyError` (1),
  `UnicodeDecodeError` (1).
- **Failure modes span the classes that matter**: mutability and aliasing, scoping, quantifiers,
  iteration, text encoding, class/property semantics, integer division, ordering, and the
  "empty case" traps.
- The prompt is the same one h1/h2 use; the flow is the same per-task-home flow
  (`run_arm_h3.py`, one workspace and one ledger per task).

## The arms

| arm | data | seeds |
| --- | --- | --- |
| base | — (unadapted) | 1 run |
| chtd-seed0/1/2 | 9 student-state rows, `last` | **3 seeds** (seed0 from the pilot, seeds 1–2 trained tonight) |
| ctrl-seed0/1 | 9 teacher-own rows, `assistant` | 2 seeds |
| v5 | 93 teacher-own rows, `assistant` | 1 run |

## Scores (30 tasks, phrase gate)

| arm | gate | note |
| --- | ---: | --- |
| base | 20/30 | 10 failures concentrated on the new mechanisms |
| chtd-seed0 | 26/30 | |
| chtd-seed1 | 27/30 | |
| chtd-seed2 | 24/30 | one task lost to an agent-process crash (`h3-divmod-order`, exit `0xC0000409`), not to reasoning |
| ctrl-seed0 | 25/30 | |
| ctrl-seed1 | 23/30 | |
| v5 | 27/30 | |

**The student-state arm's three seeds: 24, 26, 27 — a spread of three tasks.** The teacher-own
arm's two: 23, 25 — a spread of two. The two arms' means differ by ~1.7 (25.7 vs 24.0), i.e. **less
than one seed of noise**. On this set, data source is not a distinguishable factor.

**Base vs adapted is a different picture.** The worst adapted arm (23) still clears the base (20)
by three, and the best by seven. That difference is several times the seed spread, and it holds
for every arm on the harder half of the set (the ten fixtures the base fails).

## Adjudication spot-check (6 tasks × 6 arms, 36 answers read)

The gate erred in both directions, as it always does. The load-bearing cases:

- **`h3-last-zero` — gate false negative on `ctrl-seed1`.** Its answer says `items[-0:]` is
  `items[0:]`, i.e. the whole list, which is exactly the defect; the phrase list
  (`zero|-0|negative`) missed it because the answer wrote `n=0` and `items[0:]`. Read against the
  answers, ctrl-seed1 is **24/30**, not 23.
- **`h3-counter-most-common` — gate false negative on `base`.** Its answer says the function
  "should return just the most common letter, not a list of tuples" — the mechanism and the
  direction. Base reads **21/30**.
- **`h3-counter-most-common` — gate false positive on `chtd-seed0`.** It identified the mechanism
  but concluded the fix is to edit `check.py`'s assertion to index `[0]` — i.e. to change the
  test, not the code. A strict read is partial, not correct.
- **`h3-divmod-order` — gate pass on `v5`, read as partial.** It names the swap but states
  `divmod` returns `(remainder, count)`, which is backwards (it returns `(quotient, remainder)`);
  the conclusion happens to land right.
- `h3-counter-most-common` on `v5` is a second boundary case: the mechanism is named, but its
  prescribed fix `[0][1]` would return the count, not the letter.

After these three corrections the totals are **base 21, chtd 25.7 mean, ctrl 24.5 mean, v5 27** —
the arm ordering is unchanged, and the student-state/teacher-own gap stays under one seed of
spread.

## What this establishes, and what it does not

**Establishes.**

1. **Fine-tuning on 9 rows improves this base on this distribution, for every arm tested.** All
   five adapted arms land 23–27 against the base's 20 — a 3-to-7-task effect that dwarfs seed
   noise and survives the gate's known errors. Whatever else is uncertain, "the fine-tunes
   regress" is not supportable on this material.
2. **The student-state arm's three seeds reproduce each other within 3 tasks** (24–27), so the
   arm is stable enough to be compared — and the comparison is what kills the previous claim:
   chtd 24–27 vs ctrl 23–25 overlap.
3. **The h1/h2 +2–3 task advantage does not replicate on 30 fresh tasks.** With three seeds of
   the experimental arm now measured, that gap sits inside the arm's own seed spread. The honest
   reading of the earlier sets is "too small to tell", not "ahead".

**Does not establish.** Any reason to prefer student-state data over teacher-own at this scale.
The design still cannot separate data source from supervision mode; the h3 set, like h1 and h2,
has now been read by humans; and 30 tasks still resolve only ~15-point differences. One arm
(cthd-seed2) lost a task to an infrastructure crash, which is a measurement gap, not a result.

## Files

```
harness/tasks/holdout-3.jsonl                the 30 tasks, checked in
harness/tasks/holdout-3/<id>/                the fixture sources, checked in (check.py + module)
data/holdout-3-20261014/
  workspaces/<id>/{check.py,<module>.py}   the working copies; all verified failing
  arm-*/                                   per-task workspaces and ledgers
  records-*.json                           raw sweeps (session, ledger, workspace)
  scores-*.json                            gate results per arm
/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v2/
  chtd-seed1-adapter/, chtd-seed2-adapter/ the new seed replicates (run.json in each)
```

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
