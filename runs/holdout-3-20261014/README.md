# Run `holdout-3-20261014` — a 30-task set for the question the small sets could not answer

The h1/h2 comparison left two claims: adapted arms beat the base, and the student-state arm beats
the teacher-own arm by two or three tasks. The second claim rested on 12–15 tasks, and one task
there is 7–8 points. This run answers both questions on **30 new tasks nobody had read**, with the
student-state arm trained at **three seeds** — the missing piece that turns a single number into a
distribution.

**The result, stated first.** *Adapted beats base* holds, but smaller than this record first
claimed: by hand read the gap is **1 to 5 tasks** (base 22, arms 23–27), not the "3-to-7" written
at first pass — see the revision note below. *Student-state beats teacher-own* **does not
replicate**: the two arms' seed ranges overlap and their means differ by ~1.7 tasks, which is
smaller than the student-state arm's own seed spread. The earlier +2–3 was consistent with noise.

> **Revision note, 2026-10-14 (adversarial review).** SpikeBot 003 reviewed this record against
> the raw material (comment on PR #8) and the review landed three findings that this file now
> carries. (1) Three of the base's ten recorded failures are **correct diagnoses the gate
> missed**, and one of its twenty passes is a wrong conclusion the gate rewarded — hand-read, the
> base is **22/30**, not 20. The "3-to-7-task effect" was therefore overstated; the surviving
> effect is 1–5. (2) Two cells the record scored as failures (`h3-counter-most-common` on v5 and
> on chtd-seed2) were lost to a **phrase bug**, not to the answers — the phrase
> `index|unpack|element|first|\[0\]` demands a literal backslash under the scorer's own semantics
> and matched nothing; seven phrases carried that defect. Fixed in `harness/tasks/holdout-3.jsonl`
> and re-scored; see "The phrase fix" below. (3) chtd-seed2 lost **two** tasks to infrastructure,
> not one — the crash this record names, plus a 180 s timeout it did not. All three findings were
> re-verified against the records before being applied; the corrections are in place below.

## The task set

30 single-defect Python diagnosis fixtures under `data/holdout-3-20261014/`. Design rules:

- **No mechanism is identical to the 71 pre-existing tasks.** Checked against the 34 teacher-demo
  fixtures, the 15 holdout-1 tasks, the 12 holdout-2 tasks and the 10 pass-2 tasks: no mechanism
  repeats. Several near-misses were caught and changed during design (e.g. an integer-division
  average was dropped because `fix-integer-division` already carries it). **The review showed the
  guard was not complete:** three fixtures re-use an existing surface with a changed symptom —
  `h3-slice-negative-step` shares the negative-slice family with `fix-slice-direction` (the h3
  task adds the reversed step), `h3-max-empty-raise` breaks on a **byte-identical line**
  (`return max(scores)`) in a same-named `scores.py` as `fix-max-returns-key` (different
  mechanism: empty-input vs key-vs-value), and `h3-stem-find-vs-rfind` is a third member of the
  `tp2-find-missing-index` / `tp2-suffix-boundary` extension-parsing cluster. "No mechanism is
  identical" is the defensible claim; "every mechanism is new" was stronger than the material.
- **All 30 verified broken before use.** `check.py` in each workspace fails — 30/30, by
  `AssertionError` (19), `TypeError` (4), `ValueError` (3), `AttributeError` (2), `KeyError` (1),
  `UnicodeDecodeError` (1). The review re-ran all thirty and reproduced the exact distribution;
  a scan for planted failures (`raise NotImplementedError`, TODO markers, syntax errors) found
  none.
- **Failure modes span the classes that matter**: mutability and aliasing, scoping, quantifiers,
  iteration, text encoding, class/property semantics, integer division, ordering, and the
  "empty case" traps.
- The prompt is the same one h1/h2 use; the flow is the same per-task-home flow
  (`eval/run_arm_h3.py`, one workspace and one ledger per task).

## The arms

| arm | data | seeds |
| --- | --- | --- |
| base | — (unadapted) | 1 run |
| chtd-seed0/1/2 | 9 student-state rows, `last` | **3 seeds** (seed0 from the pilot, seeds 1–2 trained tonight) |
| ctrl-seed0/1 | 9 teacher-own rows, `assistant` | 2 seeds |
| v5 | 93 teacher-own rows, `assistant` | 1 run |

## Scores

Three readings of the same 30 tasks. `as run` is what the sweeps recorded; `phrase fix` re-applies
the corrected phrase lists to the same answers; `hand read` applies the confirmed gate errors in
both directions (each cell below is named).

| arm | as run | phrase fix | hand read | hand-read basis |
| --- | ---: | ---: | ---: | --- |
| base | 20 | 20 | **22** | +3 false negatives, −1 false positive |
| chtd-seed0 | 26 | 26 | **25** | −1 false positive (change-the-test answer) |
| chtd-seed1 | 27 | 27 | 27 | not fully re-read — the least certain row here |
| chtd-seed2 | 24 | **25** | **25** | +1 phrase-bug false negative; 2 tasks have no answer (see below) |
| ctrl-seed0 | 25 | 25 | 25 | not fully re-read |
| ctrl-seed1 | 23 | 23 | **23** | +1 false negative, −1 false positive |
| v5 | 27 | **28** | **27** | +1 phrase-bug false negative; that cell is a partial on read |

**The student-state arm's three seeds: 25, 27, 25 (hand read).** The teacher-own arm's two: 23, 25.
The means differ by ~1.7, and a two-mean comparison on n=3 vs n=2 puts the standard error of that
difference at ~1.36 (t≈1.2) — nowhere near significant, which agrees with the range test this
record used at first pass, though the range is not the right instrument (see below).

**cthd-seed2 lost two tasks to infrastructure, not one.** `h3-divmod-order` crashed
(`0xC0000409`, empty answer, 3.7 s) and `h3-string-count-overlap` hit the 180 s task timeout
(43-character answer). No other arm has a nonzero exit. On the 28 tasks that produced an answer,
seed 2 is 25/28.

**Base vs adapted is the number the review moved.** By the corrected gate the worst adapted arm
(23) clears the base (20) by three and the best by eight. By hand read the base is 22 and the arms
are 23–27: the gap is **1 to 5 tasks**, and the base sits one task from ctrl-seed1. The claim that
survives is "every adapted arm beats the base by hand read"; the size this record first printed
does not.

**One statistical correction.** The record argued "seed spread (3) ≥ mean gap (1.7), so no paired
effect". The conclusion stands — a range is not a standard error, and on n=3 vs n=2 the SE of the
difference is ≈1.36 — but the same range argument cannot be applied to base-vs-adapted as written,
and the material bounds base variance far more tightly than it first appears: on h1 the base ran
ten sweeps, identical 13/15 every time. Base is near-deterministic on this project's evidence; the
three-task spread is the *adapted* arms' noise. Using it to shrink the base gap would be
anti-conservative, so the hand-read gap above is stated without it.

## The phrase fix

The scorer's phrase matching treats every character except the `.?` operator **literally**
(`harness/tasks/check_diagnosis.py:_spells`). Seven holdout-3 phrases were written with regex-style
escapes — `\[0\]`, `str\(`, `\.value`, `\[2\]`, `\.`, `grid\[0\]`, `\x27\x27` — which under that
semantics demand a *literal backslash* and can never match the text they were meant to match. The
`\[0\]` case cost two real cells: `h3-counter-most-common` on v5 and chtd-seed2, whose answers
name the mechanism and write `[0]` but no other accepted spelling.

Fixed on 2026-10-14 in `harness/tasks/holdout-3.jsonl` (and the working copy the scorer reads),
each escape corrected to its literal spelling (`[0]`, `str(`, `.value`, `[2]`, `join|int|string|''`
, `grid[0]`; the lone `\.` was dropped rather than kept as a bare any-character literal). Re-scored
against the same answers: exactly two cells flip, both to pass — v5 27→28, chtd-seed2 24→25. Every
other gate cell in the batch is unchanged, verified by re-running all 210. One of the two flipped
cells still reads as a partial by hand (v5's prescribed fix `[0][1]` returns the count, not the
letter), which the hand-read column carries. No other task set in the repo has backslash phrases —
scanned `error-recovery.jsonl`, `final-holdout.jsonl`, `example.jsonl`.

## Adjudication, after two reviews

The original spot-check read 36 answers (6 tasks × 6 arms). The adversarial review re-read all of
the base's ten failures and a further 3 × 7 cells of its own choosing, and separately re-ran the
gate over all 210 cells. The confirmed gate errors:

**False negatives — correct answers the gate refused:**

- `h3-counter-most-common` on **base** (already recorded at first pass): names the mechanism and
  the direction.
- `h3-counter-most-common` on **v5** and **chtd-seed2** (found by the review): both answers name
  the mechanism and write a fix; both were lost to the `\[0\]` phrase bug, now fixed.
- `h3-split-unbounded` on **base** (found by the review): names the mechanism ("`line.split(" ")`
  … split on any occurrence … should instead split on the first space character") and prescribes
  the fix. Its worked example mis-states the crash as a wrong return value — the diagnosis and fix
  are nonetheless right, so it reads correct with that caveat.
- `h3-string-count-overlap` on **base** (found by the review): names non-overlapping counting
  explicitly and prescribes the sliding-window fix. Unambiguous.
- `h3-last-zero` on **ctrl-seed1** (already recorded at first pass): `items[-0:]` is `items[0:]`
  — the defect, named.

**False positives — wrong conclusions the gate rewarded:**

- `h3-enum-not-str` on **base** (found by the review): concludes the assertion "is actually
  correct — it should pass" and blames maintainability. The real defect (`"status: " + status` on
  an Enum member) is never named. Where this record's earlier table put base at 21, this cell
  brings it back down.
- `h3-property-no-setter` on **ctrl-seed1** (found by the review): says `reading.celsius += delta`
  "changes the object's state" and the assertion "passes temporarily". The property has no setter;
  the operation raises `AttributeError`. Confidently wrong.
- `h3-counter-most-common` on **chtd-seed0** (already recorded at first pass): prescribes editing
  `check.py`'s assertion — the test, not the code.

**Reliability of the two reads.** The review's own re-grade of the three h1 arms differed from the
first-pass table by one C/P boundary cell per arm and moved no ordering; its h3 findings here were
re-verified against the full answer texts before being applied. Two rows in the table above
(cthd-seed1, ctrl-seed0) were not fully re-read by either pass — their hand-read numbers are the
as-run gate numbers carried forward, and they are the residual uncertainty in this ledger.

## What this establishes, and what it does not

**Establishes.**

1. **Fine-tuning on 9 rows improves this base on this distribution, for every arm tested.** By
   hand read the arms land 23–27 against the base's 22 — a 1-to-5-task effect. It survives the
   gate's known errors in both directions, but it is smaller than this record first said, and the
   base's own answers (seven conclusions wrong in substance, three correct diagnoses the gate
   refused) are the strongest competing account of part of it. "The fine-tunes regress" remains
   unsupportable on this material.
2. **The student-state arm's three seeds reproduce each other within 2 tasks** (25–27 hand read),
   so the arm is stable enough to be compared — and the comparison is what kills the previous
   claim: chtd 25–27 vs ctrl 23–25 overlap.
3. **The h1/h2 +2–3 task advantage does not replicate on 30 fresh tasks.** With three seeds of
   the experimental arm now measured, that gap sits inside the arm's own seed spread. The honest
   reading of the earlier sets is "too small to tell", not "ahead".

**Does not establish.** Any reason to prefer student-state data over teacher-own at this scale.
The design still cannot separate data source from supervision mode; the h3 set, like h1 and h2,
has now been read by humans; and 30 tasks still resolve only ~15-point differences. One arm
(cthd-seed2) lost two tasks to infrastructure (a crash and a timeout) — a measurement gap, not a
result, and one that cuts *for* the arm on the answered set (25/28).

## Files

```
harness/tasks/holdout-3.jsonl                the 30 tasks, checked in (phrase lists fixed 2026-10-14)
harness/tasks/holdout-3/<id>/                the fixture sources, checked in (check.py + module)
artifact-binding.md                          adapter sha256s for every arm of this batch,
                                             with the evidence fixing each to its run
eval/run_arm_h3.py                           the per-task-home sweep script, checked in
data/holdout-3-20261014/
  workspaces/<id>/{check.py,<module>.py}   the working copies; all verified failing
  arm-*/                                   per-task workspaces and ledgers
  records-*.json                           raw sweeps (session, ledger, workspace)
  scores-*.json                            gate results as run
  scores-rescored-20261014.json            all 210 cells re-graded after the phrase fix
/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v2/
  chtd-seed1-adapter/, chtd-seed2-adapter/ the new seed replicates (run.json in each)
```

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
