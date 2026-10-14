# Run `holdout-3-20261014` — a 30-task set for the question the small sets could not answer

The h1/h2 comparison left two claims: adapted arms beat the base, and the student-state arm beats
the teacher-own arm by two or three tasks. The second claim rested on 12–15 tasks, and one task
there is 7–8 points. This run answers both questions on **30 new tasks nobody had read**, with the
student-state arm trained at **three seeds** — the missing piece that turns a single number into a
distribution.

**The result, stated first.** Under the corrected instruments (third pass, below), this batch
**does not support either claim it was built to test**. *Student-state beats teacher-own* is not
supported in either direction. *Adapted beats base* survives only in a weaker, differently-shaped
form: by the mechanically fixed gate every adapted arm clears the base (20) by 1–3 tasks except
v5 (93 rows) by 8, and by a full hand read of all 210 answers the CHTD-specific arms **do not**
clear it (17–19 against the base's 17) while the teacher-own control arms do (21 and 25). The
gate-based table this record was built on — base 20, chtd 24–27, ctrl 23–25, v5 27 — is an
instrument artefact: the phrase gate passes answers that conclude there is no defect, and the
CHTD arm's answers did that most often.

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

> **Revision 2, 2026-10-14 (the third pass).** This record went back to the raw answers a third
> time, now reading **all 210** — every pass, not only the failures — and the reading found the
> instrument's largest defect so far. The phrase gate asks whether an answer says the right
> *words*; it never asks whether the answer concludes the fixture is broken. Twenty-four answers
> in the batch contain "no defect"; sixteen of the gate's passes flip to fail once a
> defect-denial phrase list is added to `forbid` (the fix is in `harness/tasks/holdout-3.jsonl`,
> and the per-cell flips are named below). After that fix the CHTD arms' lead over the base
> shrinks to 1–3 tasks and the student-state/teacher-own pair reads as a wash. A complete hand
> read of all 210 answers (`handread-20261014.json`, rubric stated there) goes further: it puts
> the CHTD arms at 17–19 correct — the base's level — and the teacher-own arms at 21 and 25.
> The claims this record was built to test do not survive that read. The ledger is one reader's
> judgement, per-cell and auditable; the reviewer's re-grade of it is the open item.

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

## The third pass: the deny-list fix and the full hand read

The phrase gate is a one-sided test in a second way the earlier passes had not priced in. It
requires the expected words and refuses a few named ones, but it never refuses an answer whose
**conclusion** is that the fixture is fine. An answer that walks through the code and ends
"there is no defect in the code as written" contains the same words a correct diagnosis does. On
this batch, 24 answers contain "no defect" and 16 of them were gate passes.

**The fix.** Every h3 task's `forbid` now carries a defect-denial list (`no defect|no bug|is
correct as written|functionally correct|no obvious defect|no actual defect|nothing broken|both
files are correct|is correct and|code works|no issues|no problems|nothing wrong|no error in the
code|passes as expected`), chosen by testing each candidate against all 210 answers: any phrase
that matched a cell the hand read calls correct was dropped ("works correctly", "works as
intended" and "code is correct" all failed that test and are not in the list). The rows live in
`harness/tasks/holdout-3.jsonl`; the flips are reproducible with the checked-in scorer.

**Three instruments, one table.** `as run` is what the sweeps recorded; `phrase fix` re-applies
the corrected phrase lists; `deny fix` adds the defect-denial list; `hand read` is the full
per-cell read (below).

| arm | as run | phrase fix | deny fix | hand read (correct) |
| --- | ---: | ---: | ---: | ---: |
| base | 20 | 20 | 20 | **17** |
| chtd-seed0 | 26 | 26 | 23 | **19** |
| chtd-seed1 | 27 | 27 | 22 | **18** |
| chtd-seed2 | 24 | 25 | 21 | **17** (of 28 answered) |
| ctrl-seed0 | 25 | 25 | 22 | **21** |
| ctrl-seed1 | 23 | 23 | 21 | **25** |
| v5 | 27 | 28 | 28 | **21** |

Cells the deny fix flips (all were gate passes whose answers conclude the code is fine):
`chtd-seed0` — enum-not-str, join-nonstring, max-empty-raise; `chtd-seed1` — capitalize-vs-title,
enum-not-str, join-nonstring, round-returns-float, sum-no-start; `chtd-seed2` —
capitalize-vs-title, enum-not-str, list-index-raises, max-empty-raise; `ctrl-seed0` —
enum-not-str, join-nonstring, round-returns-float; `ctrl-seed1` — enum-not-str,
round-returns-float. The fix catches the denial class; it does **not** catch answers that name
the wrong mechanism in the right words, or that blame the test, which is why the hand-read
column is the one this record's conclusions are set by.

**The hand read.** All 210 answers were read against the fixture sources on 2026-10-14;
the per-cell grade and a one-line reason are in `handread-20261014.json` beside this file. The
rubric: *correct* — points at the defective code, and any repair it offers would make the
fixture pass (minor mis-statements of the failure mode are notes, not demotions); *partial* —
right area, but the account does not explain the observed failure, or the repair is wrong or
hedged toward editing the test; *failed* — concludes there is no defect, blames the test, or
names an unrelated mechanism. Two chtd-seed2 cells have no answer at all (a crash and a timeout)
and are counted as neither.

What the read changes, in one line each: the gate's pass/fail ordering (CHTD arms on top) is not
reproduced by reading (CHTD arms at the base's level; the teacher-own arms above both). The
largest single error class is denial (24 cells), followed by "the test is wrong" (the answer
relocates a real mismatch into `check.py`, 9 cells) and by wrong-mechanism-with-right-words.
Every cell the earlier passes had flagged is confirmed by this read; no earlier finding is
withdrawn by it.

**Caveats on the read.** One reader (the author) — no blind protocol, and the grades are
contestable at the correct/partial boundary, where a stricter reader would move several cells
down and a more lenient one up; the ledger names every cell so the boundary can be redrawn. The
asymmetry to watch for: two arms (chtd-seed1, ctrl-seed0) were read from full texts and five from
700-character tails, and more text can only find more errors — the reviewer's independent
grade of the tails-read arms is the check on that.

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

**Reliability of the reads.** The review's own re-grade of the three h1 arms differed from the
first-pass table by one C/P boundary cell per arm and moved no ordering; its h3 findings here were
re-verified against the full answer texts before being applied. The two rows this paragraph used
to carry as "not fully re-read" (cthd-seed1, ctrl-seed0) were read in full by the third pass, and
both moved — see the table above. The third pass's own residual is stated where it belongs: one
reader, contestable at the C/P boundary, and five arms read from tails rather than full texts.

## What this establishes, and what it does not

**Establishes.**

1. **The instrument was the largest single factor in the earlier numbers.** The phrase gate
   passes defect-denial conclusions; adding a defect-denial `forbid` list (chosen by testing every
   candidate phrase against all 210 answers) flips 16 cells, unevenly across arms — five from
   chtd-seed1, four from chtd-seed2, three each from chtd-seed0 and ctrl-seed0, two from
   ctrl-seed1, none from base or v5. Any comparison drawn from the as-run gate is a comparison of
   answer styles as much as of models.
2. **On the corrected instruments the paired comparison is a wash or worse for the student-state
   arm.** Deny-fixed gate: chtd 21–23 vs ctrl 21–22. Hand read: chtd 17–19 vs ctrl 21 and 25. The
   earlier +2–3 did not just fail to replicate at n=30; on this batch it reads backwards.
3. **"Adapted beats base" survives only as a weak, mostly teacher-own result.** Deny-fixed gate:
   every adapted arm ≥ base+1, v5 ≥ base+8. Hand read: the CHTD arms tie the base (17–19 vs 17)
   and the teacher-own arms lead (21, 25), with v5 at 21. Neither reading supports a claim about
   the CHTD-specific ingredient; what survives is that *some* fine-tunes on *some* data shapes
   beat the base by a few tasks.
4. **The h1/h2 +2–3 task advantage still does not replicate on 30 fresh tasks** — and the third
   pass is a further reason not to lean on the h1/h2 numbers either: they were read before this
   batch's lesson about unread passes, and the lesson is not specific to h3.

**Does not establish.** Any reason to prefer student-state data over teacher-own at this scale, in
either direction — or the reverse. The design still cannot separate data source from supervision
mode; the h3 set, like h1 and h2, has now been read by humans; and 30 tasks still resolve only
~15-point differences. Two chtd-seed2 cells have no answer at all (a crash and a timeout), which
is a measurement gap and the reason its row is stated over 28.

**What would change this record's mind.** An independent re-grade of `handread-20261014.json`
that moves the correct/partial boundary enough to restore the CHTD arms' lead — the cells to argue
about are named, and the tails-read arms (base, chtd-seed0, chtd-seed2, ctrl-seed1, v5) are the
ones where a fuller read could still find more, not fewer, errors.

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
  scores-denyfix-20261014.json             all 210 cells re-graded after the deny-list fix
handread-20261014.json                     the third pass: every cell, graded, with a reason
/mnt/proj/chtd-experiments/node4-recollection-20261012/matched-v2/
  chtd-seed1-adapter/, chtd-seed2-adapter/ the new seed replicates (run.json in each)
```

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
