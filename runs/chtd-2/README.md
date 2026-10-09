# Run 2 — twelve samples, and the first behaviour that moved

The adapter exists on the training box at `/root/chtd_runs/chtd-2`. The weights are not in this
repository. What is here is the record, plus what the tuned model did on tasks it never trained on.

```
adapter_model.safetensors  4c9247e815ddd4efac87249fb39d69554c692be3d5085227718746bfacae692c
adapter_config.json        e574a8c1cbaf8c5ea59628e99da38289472fbe35d2f86896bf0f4546aebc1e61
run.json                   6892ba94257238da8f7768ef4aef4e083c552e8d3c3241870f07965441f845a7
```

## What ran

```
base model   Qwen3-4B-Instruct-2507, 4-bit NF4, LoRA on q/k/v/o_proj
adapter      11,796,480 trainable of 3,739,343,360 logical parameters (0.3155%)
dataset      bd090f51d3db8c7380f02bbb5cad227b9364c457c0f74b79c951c495cb3f68cf  (12 entries)
samples      7ad38c32926a47e678afaf5e00baec6cf382028b54bddd4222d57ac1a7149c51  (12 rows)
track        1eb6b3ffaee3fb07228abebf4d3ce67112debb525c49e2dd0bc1eb363be0a97a
              — observed from each session's own `session/start`, on `minimal-rl`
steps        9   (12 samples × 3 epochs ÷ accumulation 4)
loss         4.058 → 3.873 → 2.505 → 2.312 → 0.297 → 0.825 → 0.529 → 0.342 → 1.111
train_loss   1.7612
wall clock   113 s
peak VRAM    17.84 GB
```

Compare run 1: three steps, twelve samples fewer, and a mean loss that barely moved (4.13). Here the
ladder falls from 4.06 to a minimum of 0.30 and settles at 1.76.

## Reproduction

```sh
python harness/tasks/build_workspaces.py                      # write the 16 workspaces
python -m distill.rounds --round data/rollouts/round-f \
    --out data/datasets/round-f                               # round -> dataset
python -m distill.sft --dataset data/datasets/round-f \
    --student-ledger data/agh_home/data/sessions.db \
    --prefix harness/prefix/minimal-rl.json \
    --out data/sft/round-f.jsonl                              # dataset -> samples
python -m distill.finetune --plan data/sft/plan-round-f.json \
    --samples data/sft/round-f.jsonl \
    --base-model <the local Qwen3-4B-Instruct-2507 directory> --output <somewhere writable>
```

The round itself needs a harness and two endpoints; `harness/round.py` is the module that drives it,
and the profile it needs is described in `docs/finetune.md`.

## The held-out result

Four tasks were written **after** the dataset was built and are in no sample. Same track, same
server, same prompts; the only difference between the two columns is whether the adapter was loaded.

```
                    base weights          + this adapter
fix-argument-order  greeting              ls, read, read, answer
fix-empty-input     greeting              ls, read, read, answer
fix-missing-prefix  greeting              ls, read, read, answer
fix-slice-direction greeting              ls, read, read, answer

diagnosis correct   0 / 4                 1 / 4
```

**The behaviour changed on every task, and the change is the trained one.** The base model answers a
request to inspect a broken directory with "Hello! I'm ready to assist you" and stops — one decision
point in all four runs. The tuned model runs `ls`, then reads both files, then answers, in all four.
That sequence is what every training sample demonstrated, and it transferred to directories it had
never seen.

**The diagnosis is right once.** Only `fix-slice-direction` names its defect. The other three
describe the code and conclude it is fine — `fix-empty-input` says "there are no apparent issues or
bugs in the implementation" about a function that raises `IndexError` on an empty string. So this is
a behaviour change with a small accuracy gain, not a capability, and one correct diagnosis out of
four is a single data point rather than a rate.

## What that does and does not license

It licenses: a fine-tune of this size, on this data, changes what the student *does* at the start of
an error-recovery task, on held-out tasks, in the direction the teacher demonstrated.

It does not license: a claim that error recovery was acquired. Three of four answers are wrong, and
the training signal was 12 rows of ~16 supervised tokens each — about 190 tokens. The model learned
a sequence, which is what 190 tokens can carry.

## Two defects this run exposed

**Whitespace is missing from every recorded answer.** Every assistant message in the ledger reads
`"Hello!I'mreadytoassistyou"` — the words are there, the spaces are not. It is not the fine-tune:
the base model's replies are stored the same way, so it is in the serving path or in what the harness
keeps. It does not change the result above, and it does mean every check that matches a phrase
containing a space is matching against text that has none. The checks here use single words.

**A phrase check can pass an answer that says the opposite.** The first held-out check used two
phrases per task, and `fix-empty-input` passed on an answer claiming there was no bug, because that
answer contained "split" and "empty". The four held-out phrases were tightened to name the defect
rather than the subject. One of those was then wrong the other way: `slice` is not a substring of
`slicing`, so the check failed the answer that was plainly correct. Both directions are recorded
because both are the reason a phrase check is a screen and not a measurement.
