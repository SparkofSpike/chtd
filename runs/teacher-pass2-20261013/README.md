# Run `teacher-v5-20261013` — the first arm that beats its base on human reading

This is the arm that follows the 2026-10-13 withdrawal. `v4`'s "beats its base" claim was
manufactured by the phrase scorer; this run targets the weaknesses the hand read found, and the
same hand read says it clears the base. It is also the first run here whose every decisive answer
was adjudicated by a human before any number was written down.

```
arm            data                                    held-out score (gate / human)
base           —                                       13/15 · 9/12   /   ~12-13 · 4 correct
v4             v2's 63 rows, seed 1                    15/15 · 12/12  /   ≈level (withdrawn)
v5             v3's 83 rows + 10 new demonstrations    14/15 · 10/12, 11/12  /   8 correct
```

## What changed between v4 and v5

The adjudication named three shared blind spots — path joining, `strip`-as-charset, mutating
during iteration — that defeated **both** arms, and two mechanisms the adapter got wrong on its
own (banker's rounding, generator exhaustion). Ten new training tasks were written for exactly
those mechanisms, as different fixtures from the holdout-2 set so holdout-2 stays a transfer test:

| task | mechanism | source of the lesson |
| --- | --- | --- |
| `tp2-join-log-path` | `os.path.join` discards the left side on an absolute right | shared blind spot |
| `tp2-rstrip-charset` | `strip`/`rstrip` take a character set, not a suffix | shared blind spot |
| `tp2-loop-remove-skips` | removing during iteration shifts the walk | shared blind spot |
| `tp2-round-half-even-format` | `format`/`round` are half-to-even | v4's regression |
| `tp2-sort-returns-none` | in-place mutators return None | value semantics |
| `tp2-find-missing-index` | `find` returns -1, not an error | sentinel indexing |
| `tp2-default-mutable` | mutable default arguments are shared | classic trap |
| `tp2-suffix-boundary` | substring test vs extension boundary | string semantics |
| `tp2-hex-without-base` | `int()` is base 10 unless told otherwise | parsing |
| `tp2-bool-key-collision` | `True == 1` in dict keys | equality/hash |

All ten fixtures were verified broken before use (a task whose check passes measures nothing).
The teacher (`agnes-3.0-flash`) solved all ten; demonstrations were collected under the same
discipline as before (isolated home per task, gate + zero-refusal + successful-read). v5 trains
on v3's 83 rows plus these 10, same recipe and seed as v4, so the only variable against v4 is
data. 240 steps, loss 0.2838.

## holdout-2 (the 12-task transfer test) — human adjudication

One fresh sweep per arm, then every answer read in full and graded: **correct** (names the
mechanism), **partial** (right direction or fix, with a material error), **failed**.

```
task                     gate v5/base2    v5                  base2
h2-setdefault-none       pass / pass      correct             partial  (says TypeError; it is a silent None)
h2-re-match              pass / pass      correct             correct
h2-round-half-even       pass / pass      correct             correct
h2-join-absolute         pass / FAIL      partial  (fix works, mechanism wrong)   failed
h2-naive-aware           pass / pass      correct             partial  (says wrong value; it is a TypeError)
h2-floor-toward-negative FAIL / pass      correct (gate false negative)          correct
h2-strip-charset         pass / FAIL      correct  ← blind spot cleared          failed
h2-dict-mutation-view    pass / FAIL      partial  ← blind spot improved         failed
h2-unbound-local         pass / pass      partial             partial
h2-class-attr-shared     pass / pass      correct             correct
h2-finally-return        pass / pass      correct             partial  (misses that the success path is also swallowed)
h2-min-key-string        pass / pass      partial             partial
                         -------------    ----------------    ----------------
                         gate 10/12 vs 9  correct 8           correct 4
                                          partial 4           partial 5
                                          failed  0           failed  3
```

**v5 clears the base with a margin that survives the scoring the gate cannot do**: 8 correct
answers against 4, and no failed answer against 3. Every v5 answer is at minimum actionable; three
of the base's twelve conclude wrongly. This is the opposite pattern from v4's withdrawal, where
the gate said +2 and the answers said −1.

The blind-spot task that fully transferred is `strip`-as-charset — v5 says "removes *any*
characters from the strip set from both ends" and prescribes `removeprefix`. `dict`-mutation and
`join` improved to partial but did not fully transfer: on `join` v5 still fabricates a
double-slash story (the fix it prescribes works, its explanation of why is wrong); on `dict` it
knows "mutating while iterating" but claims silent skipping rather than the `RuntimeError` the
fixture actually raises.

A second v5 sweep scored 11/12 (the one loss is `unbound-local`, where the gate's phrase `assign`
misses a correct-in-substance answer). The arm is stable; its two sweeps differ by one task.

## holdout-1 (the original 15-task screen) — regression check

```
v5    14/15   the one failure is fix-none-cap, a genuine misdiagnosis this sweep
base  13/15   fails fix-none-cap (genuine) and fix-or-default-falsy (genuine)
```

The decisive task is `fix-or-default-falsy`: v5 names the falsy-`0` defect and the fix; base
writes "the port() function correctly returns the specified port value when provided… The code is
otherwise correct" — a false conclusion on a defect it was shown. And `fix-generator-exhaust`,
v4's regression, is now **correct** in v5 ("once the first sum() exhausts the generator… returns
(3, 0) instead of (3, 12)"), where v4 had called the generator expression "a tuple".

## What this establishes, and what it does not

**Establishes.** With human adjudication as the instrument: on the 12-task transfer set v5 scores
8 correct to the base's 4 with no failed answers; on the 15-task screen it scores 14 to the base's
13, winning on tasks where the base's text is confidently wrong. Two mechanisms that both arms
previously failed (`strip`, `round`) and one that v4 regressed (`generator-exhaust`) are now
solved. Two more blind spots (`join`, `dict`) improved from "not found" to "found but explained
wrongly".

**Does not establish.** Generality beyond these fixtures. The two sets are 27 tasks of Python
defect diagnosis, both have now been read by a human, and the next honest test is a set nobody has
looked at. The scorer still errs both ways (one false negative each way in these very sweeps), so
any future gate-only comparison inherits this document's uncertainty.

## Files

```
data/teacher-pass2-20261013/     training pass 2 (gitignored)
  tasks.jsonl, workspaces/       the 10 fixtures
  records/                       teacher demonstrations (10/10 usable)
  built/samples-v5.jsonl         the 93 rows
  built/plan-v5.json             the plan (digest recomputed, not copied)
data/holdout-2-20261013/
  records-{v5,base2,v5b}.json    the sweeps; scores-*.json the gate's counts
data/rollouts/{v5-h1-20261013,base-h1-20261013}/
                                 the holdout-1 regression checks under PR #5's integrity gates
data/holdout-1-check-20261013/selection-manifest.json
                                 the train/holdout split this run checked against
/mnt/proj/chtd-experiments/teacher-demo-20261012/adapter-v5
                                 the weights
```

One bookkeeping defect to note: `distill/finetune.message_bounds` was fixed in `f3164a8`, and
plan-v4 had been produced by copying plan-v2 without recomputing its digest — it stayed correct
only because the box's `samples-v2.jsonl` was still the 63-row version at training time. Plan-v5
recomputes. A local copy of `samples-v2.jsonl` that differs from the box's (83 rows vs 63) is
noted here so nobody diffs it and wonders.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
