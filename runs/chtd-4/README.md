# Run 3 — the complete LoRA, and the first correct diagnoses on held-out tasks

The adapter exists on the training box at `/root/chtd_runs/chtd-4`. The weights are not in this
repository. What is here is the record, the comparison that makes it worth anything, and the two
defects this run found.

```
adapter_model.safetensors  f5ebe23860d90bd615d0d7319d229c93784af30302cefab067aedb3160754bb0  (253 MB)
adapter_config.json        (tracked in this directory)
run.json                   (tracked in this directory)
```

## What ran

```
base model   Qwen3-4B-Instruct-2507, 4-bit NF4
adapter      LoRA on all SEVEN projections: q/k/v/o + gate/up/down
             66,060,288 trainable of 4,022,468,096  (1.6423%)
rank 32, alpha 64, dropout 0.05
dataset      c031479db37bdff5cce4ee339d1075039102109c2415b4f0d9cf0d6ca2c1bf7f  (19 entries)
samples      8de7e26147d14f78fae884d593803a62638d1eb26358925d287f720dc080dcbd  (19 rows)
track        1eb6b3ffaee3fb07228abebf4d3ce67112debb525c49e2dd0bc1eb363be0a97a
steps        100   (19 samples × 20 epochs ÷ accumulation 4)
loss         3.57 → 2.66 → 0.75 → 0.19 → … → 0.077   (trainer's mean 0.2022)
grad norms   47.7 → 18.6 → 4.95 → … → 0.33..0.66     (settled)
wall clock   1344 s
peak VRAM    18.3 GB
```

The ladder is in `docs/finetune.md`. It falls three orders of magnitude and then oscillates between
0.01 and 0.17 for the last sixty steps — a plateau, which is what twenty epochs was for. Run 2
trained nine steps and its loss was still falling.

> **Correction.** `run.json` records `logical_parameters: 3739343360` and
> `trainable_share_of_logical_pct: 1.7666`, both from the function that assumed `q_proj` was square.
> The model is **4,022,468,096** and the share is **1.6423%**. The record is kept as recorded; the
> correction is in `runs/chtd-2/README.md` and in the code, which now reproduces the checkpoint to
> the parameter.

## The held-out comparison, which is the point

Five tasks were written for this measurement and appear in no training sample. Same track, same
server, same prompts. The only difference between the columns is which adapter the endpoint loaded.

```
                     base          chtd-3            chtd-4
                     weights       (22 rows,         (19 rows,
                                   3 bad labels)     labels gated)
fix-string-case      greeting      ls read read ✗    run failed
fix-sort-key         greeting      ls read ls ✗      ls read read edit ✓
fix-shallow-copy     greeting      ls read ls ✗      ls read read ✗
fix-ordinal-index    greeting      ls read ls ✗      ls read read ✓
fix-single-name      greeting      ls read ls ✗      ls read read ✗

correct diagnoses    0 / 5         0 / 5             2 / 5
```

**Three things changed, and each is a separate finding.**

**The behaviour changed in run 2 already** — the base model answers "Hello! How can I help you
today?" and stops, one decision point, on all five. Every tuned version walks: `ls`, then `read`,
then answers. That transfer was established in run 2 and holds here.

**A wrong behaviour was trained and then un-trained.** `chtd-3` trained on twenty-two rows, three of
which told the model to `read` an absolute path — `/answers.py`, `/check.py`, a `/tmp/...`
directory. The outcome gate admitted them, because the teacher's *run* succeeded. The result:
`chtd-3` calls `read({"path": "/answers.py"})` in a workspace whose file is `answers.py`, is refused
by the sandbox twice, and then says *"I'll inspect the directory structure and list the files in the
workspace"* — a run that looks like a failed diagnosis and is a path-format failure learned from its
own training data. `WorkspaceRelativePaths` now refuses those rows and `chtd-4` trains on the
nineteen that remain. It reads `answers.py`.

**The correct diagnoses are new.** Two of five, and both are a *named* defect rather than a
description: `fix-ordinal-index` reads *"check.py tests the nth function … nth(['a','b','c'], 1)
should be 'a'"*, and `fix-sort-key` goes furthest — it spots the defect and calls `edit`, which is the
first time any run in this project has tried to repair rather than describe.

## What two of five does and does not license

It licenses: a complete LoRA, on nineteen gated rows, trained to a plateau, produces on held-out
tasks both the correct search behaviour and — twice — the correct answer.

It does not license calling error recovery acquired. Three of five hold a **wrong** answer, and one
of those is worth naming: on `fix-shallow-copy` the model says *"it creates a copy of the table"*
about a function that mutates the caller's rows. That is the defect, read backwards. A wrong answer
stated confidently is the failure mode this whole method exists to reduce, and two of five leaves it
very much present.

One more caveat against the score itself: `fix-single-name` is marked ✗ by the phrase check and its
answer reads *"I'll fix the `names.py` file to handle edge cases where a full name might only have
one part"* — which names the defect. The check wants `index` or `length`. So the machine-scored 2/5
is probably 3/5 by reading, and the score is a screen rather than a measurement. That is stated in
`docs/finetune.md` as a limitation of the method rather than corrected by hand here.

## What was needed to get here

```
24 training tasks collected, 2 arms each      48 runs   14 min
dataset: 23 aligned, 1 skipped, 22 admitted, 4 refused
first training (chtd-3)                      100 steps 26 min
second training on gated labels (chtd-4)     100 steps 22 min
```

The first training was not wasted: it is what showed the gate hole, and `chtd-3` is kept because
comparing it to `chtd-4` is the evidence that the hole was real.
