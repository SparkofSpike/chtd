# Run `round-20261011` — a collection round, and the supervision rule it exposed

The first round built from the enlarged task set, and the first whose samples supervise a **diagnosis**
instead of a tool call. It produced a clean negative result and one finding worth more than the score:
**which assistant turns carry the loss decides whether the fine-tune can use tools at all.**

**Correction, 2026-10-11:** CHTD samples contain earlier student assistant turns and a final teacher
label. The `allturns` run below supervised both sources, so it is an exploratory mixed-provenance run,
not a complete teacher trajectory. Its difference from `lastonly` cannot be attributed to teacher
trajectory supervision. The trainer now refuses all-turn supervision without explicit proof that
every assistant turn came from the teacher.

## What was wrong with the previous arm, stated mechanically

`runs/chtd-20261011-a` supervised the last turn only. That is what the design calls for — the label is
the teacher's decision at a divergence, and the turns before it are the state — and it produced an
adapter that **could not do the task**:

```
arm            tool calls per run        mean wall clock     score
base           3.0 (every task)          19 s                14/15
that arm       0.0 (every task)           7 s                 0/15
```

Fifteen identical answers — *"I don't see any files listed or a path provided"* — because the only
behaviour it had ever been shown was prose placed after a conversation somebody else had conducted. It
had learned to answer, and never to look.

## What this round changed, and what it cost

Three changes, each with a measured reason:

**1. The follow-up asks for the diagnosis.** The original read *"Revisit your diagnosis … Read relevant
files if needed"*, and the "if needed" clause decided what the samples taught: with reading left open
the first divergence between student and teacher landed on a `tool_call`, so the label became "whether
to read again" — ten to nineteen supervised tokens. Asking for a diagnosis puts the divergence where
the measurement is. Sixteen of the previous nineteen rows had supervised a tool call; none of these do.

**2. Every assistant turn carries the loss, not just the last.** `--supervise assistant`. This is the
change that fixed the tool calls, and it is a real departure from the design's letter: the samples are
now imitation of earlier student turns together with the teacher's final answer. This run does not
isolate the effect of teacher trajectory supervision.

**3. The scorer was repaired where it was failing correct answers.** `collect_forks._score` compared
literal strings, so an expectation written `mutable|shared|default argument` could never match anything:
four tasks were reported as teacher failures on that basis alone, and three of the four were correct
diagnoses.

## The result

Fifteen held-out tasks, three arms, all behind bf16 serving on the same track:

```
arm                                            tool calls   score
base            unadapted                      3.0          14/15
allturns        every turn supervised          3.0          13/15
lastonly        last turn only supervised      0.0           0/15
```

**The observed gain over `lastonly` is not attributable to teacher-only supervision.**

The pair `allturns`/`lastonly` differs in which turns carry the loss, including student tool calls
in `allturns`. It shows that the extra labels preserved tool use in this run; it does not validate
the intended CHTD comparison.

Against the base, though, the repaired arm is **one task behind**, and both arms fail the same way:

- `fix-none-cap`: both diagnose the *assertion's error message text* instead of the missing `<`
  comparison;
- `fix-or-default-falsy`: `allturns` diagnoses the error message again; the base gets this one.

Neither is a method result. They are the same weakness in the same base model, and one of the two arms
happened to clear one of the two tasks.

## Why this round cannot show a gain, stated plainly

**The base scores 14/15.** There is one task of headroom, and a single task is 6.7 points of the score.
A method that improved half its opportunities would move the number by less than one task — inside the
run-to-run range this project has already measured for the *same* unmodified model (0/5, then 5/15,
then 8/15 across earlier sweeps; 13/15 and 14/15 on this task set). **A ceiling is not evidence of
absence, and it is not evidence of presence either.**

What the round does establish:

- the collection chain works end to end at this scale — 35 student runs, 35 teacher forks, 13 admitted
  labels, a trained adapter, a scored sweep;
- the supervision rule is a real variable with a large effect;
- thirteen diagnosis-supervised rows are enough to *preserve* a base model's behaviour through 20
  epochs of LoRA, which the equivalent tool-call-supervised rows were not.

What it does not establish: that the method improves anything. That claim needs a task set the base
does **not** already solve.

## The next experiment, named concretely

The bottleneck is no longer the pipeline, the scorer, or the supervision rule. It is that the held-out
tasks are too easy for the base model to show a gain. The next round should either:

- **add held-out tasks the base fails**, by difficulty rather than by defect kind — candidates are the
  four fixtures the teacher itself missed (`fix-accumulator-reset`, `fix-max-returns-key`, `fix-path`,
  `fix-shadowed-total`) and the two the base missed (`fix-none-cap`, plus `fix-or-default-falsy` under
  the repaired scorer), which suggests the hard band exists but the set does not sample it; or
- **choose tasks by measured base difficulty** rather than by category, so the set has headroom by
  construction.

Until then, a "the method works" claim from this task set would be measuring noise.

## Artefacts

```
E:\Coding\chtd\data\chtd-round-20261011\      (gitignored: the raw round)
  student/          35 student runs
  forks/            35 teacher forks
  review.json       13 teacher wins, 18 both-right, 4 teacher-failed-too
  built/            dataset/ and samples.jsonl
  prefix.json       captured from a live request (11 tools, 454 system tokens)
  plan.json         the recipe, unchanged from chtd-4

/mnt/proj/chtd-experiments/round-20261011/
  adapter/              supervise=last   (the 0/15 arm, kept for comparison)
  adapter-allturns/     supervise=assistant
  logs/train*.log
```

Adapter `adapter-allturns` is `8b32cd5d13963c4fb8c35ef2beff592b511fa61286031d396840b87806dbe42a`,
252 MiB, 80 steps, mean loss 0.168.
