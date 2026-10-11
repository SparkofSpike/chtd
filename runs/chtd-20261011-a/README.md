# Run chTD-20261011-a — the experimental arm, retrained after the box's `/root` was lost

This is a **re-run of the experimental (`chtd`) arm**, not a new experiment. It produces the adapter
`runs/chtd-4/` recorded and the training box lost — same sample set, same recipe, same track, same
base weights, each verified by digest rather than by name.

## Why it exists

`/root/chtd_runs/chtd-4` was written to a container path. The box was reset, `/root` did not survive,
and the adapter went with it. The rule that would have prevented it was already recorded — *all
experiment data goes on `/mnt/proj`, the 100 TB NFS* — and the control-arm work had followed it.
This run is that rule applied to the experimental arm.

What survived locally is what makes the re-run checkable rather than approximate:

| Asset | Where | Verified against |
| --- | --- | --- |
| 19 gated samples | `data/sft/round-g.jsonl` | `samples_sha256 = 8de7e261…` — the digest `runs/chtd-4/run.json` recorded |
| The dataset behind them | `data/datasets/round-g/` | `dataset_sha256 = c031479d…` |
| The recipe | `data/sft/plan-round-g2.json` | rank 32, alpha 64, dropout 0.05, 20 epochs, accumulation 4, lr 1e-4, max length 8192 |
| The base weights | `/mnt/proj/chtd-experiments/models/Qwen3-4B-Instruct-2507` | `base_weights_fingerprint = c790f354…` |

Both digests were recomputed from the local files before the run and matched. The sample digest is
the one that matters most: it covers the exact bytes the trainer reads, so a match means the re-run
trained on the recorded labels rather than on a rebuilt approximation. The base-model path moved
from `/root/models/…` to `/mnt/proj/…`; the fingerprint, which is over `config.json`,
`tokenizer_config.json` and `generation_config.json`, is byte-identical to the one `chtd-4` recorded,
so the weights under the new path are the weights the recorded run used.

## What ran

```
arm          chtd
base model   Qwen3-4B-Instruct-2507, 4-bit NF4, from the /mnt/proj copy
adapter      LoRA on all seven projections: q/k/v/o + gate/up/down
             66,060,288 trainable of 4,022,468,096  (1.6423%)
rank 32, alpha 64, dropout 0.05, lr 1e-4, 20 epochs, accumulation 4
samples      8de7e26147d14f78fae884d593803a62638d1eb26358925d287f720dc080dcbd  (19 rows)
dataset      c031479db37bdff5cce4ee339d1075039102109c2415b4f0d9cf0d6ca2c1bf7f
track        1eb6b3ffaee3fb07228abebf4d3ce67112debb525c49e2dd0bc1eb363be0a97a
steps        100   (20 epochs × ceil(19 samples / 4 accumulation) = 100)
loss         2.4553 → 0.1656 → 0.1010 → 0.0566 → … → 0.0823   (trainer's mean 0.1546)
wall clock   1354 s
peak VRAM    18.4 GB
adapter      fcc7de5d76d19dbf2fa4bb6a99a60764476eb7ac2223ec72b2dd1d04b787bed1  (252 MiB)
```

`run.json` in this directory is the trainer's own record, copied off the box unedited: its
sha256 (`8e09ae41…`) is identical to the copy on the box, and it carries no CR bytes, so the bytes
tracked here are the bytes the trainer wrote.

The step count is 100 and not 95, which is worth stating because the short form of the arithmetic —
19 samples × 20 epochs ÷ 4 accumulation — gives 95. The trainer rounds the last partial accumulation
window up, so it is `ceil(19 / 4) = 5` update steps per epoch. The repository's own pilot record
shows the same convention in the other direction: 76 samples × 3 epochs ÷ 8 is recorded as 30 steps,
which only `ceil` produces.

The adapter was checked to load rather than assumed to: its safetensors header holds 504 tensors —
seven projections across 36 layers, `lora_A` and `lora_B` each — and the file loads through `peft`
onto the base model with `r = 32, alpha = 64`. The weights themselves stay on `/mnt/proj`; what is
tracked here is the record, as with every other run in this repository.

The box-side evidence, so that a reader can check rather than take it on faith:

```
/mnt/proj/chtd-experiments/chtd-20261011-a/
  adapter/adapter_model.safetensors   fcc7de5d76d19dbf2fa4bb6a99a60764476eb7ac2223ec72b2dd1d04b787bed1   264,308,896 bytes
  adapter/run.json                    8e09ae4195f09193c1c5f5d6d4cf37880f969cf67e9e08ae39afe475bf921019
  env/chtd-control-venv.tar.gz        2b9dc6a8326fc1ba70e144e75beba021488743c3b13d976245ee07616157032f
  logs/train.log                      the trainer's full output, loss ladder included
  round-g.jsonl                       the 19 samples; file sha256 4eb9c305…, samples_digest 8de7e261…
```

The two sample numbers measure different things and both are quoted above on purpose. The file's own
sha256 covers its bytes as stored. `samples_digest` (`distill/sft.py`) is what the trainer records as
`samples_sha256` and what `chtd-4` recorded as `8de7e261…`: it hashes each row's text in order, so it
names the rows rather than the file, which is why a reader rebuilding them from a different checkout
can arrive at the same value.

The `run.json` tracked in this directory carries the same sha256 (`8e09ae41…`) as the box's copy and
no CR bytes, so it is the trainer's record and not a re-typed one.

## The one number that does not reproduce, and why it is not the data

`runs/chtd-4/run.json` records `token_lengths: {min: 6705, max: 6714}`. The same sample file,
rendered here, gives **6744–6753**. The offset is exactly **+39**, and it is not a property of the
samples.

The check that establishes that: the control arm's sample set carries its own recorded rendering
(preflight over 76 rows: 2414–3071 tokens). Re-rendered in this environment it gives 2453–3110 —
**also exactly +39**. Two datasets, built weeks apart by different code paths, displaced by the same
constant. And in both, the **supervised token count is unchanged to the token** (10–19 for the
experimental arm; 16–479 for the control arm; deltas of zero).

A constant offset confined to the prompt and absent from the labels is a rendering difference, not a
data difference — and the sample digests, which cover the bytes actually written, are identical.
The most likely carrier is the tokenizer library: this environment has `tokenizers` 0.22.2 behind
`transformers` 4.57.6, while the recorded run predates the box's reset and its exact versions are not
recoverable. That is stated as the most likely explanation and not as a demonstrated one. What *is*
demonstrated is that both arms are displaced identically, so an arm-to-arm comparison is unaffected.

`PrefixDrift` still passes on all 19 rows — every prompt renders as a strict prefix of its full
conversation — so training and serving agree with each other here, which is what that check covers.

## The held-out comparison: four arms, fifteen tasks, and no arm that wins

The adapter was measured against three others on the fifteen held-out tasks, every arm behind bf16
serving, same track, same prompt. Two of them are the pair the method is about: the 2026-10-10
experiment and control arms, trained on one sample each and differing only in where the teacher's
data came from.

```
                          passing when run    re-scored
base   unadapted                8/15           13/15
matched_baseline  2026-10-10 control   9/15     13/15
matched_chtd      2026-10-10 experiment  8/15   12/15
experiment_19     retrained today, 19 samples   5/15   10/15
```

**Read the second column.** The first is what the sweep printed with a scorer that required phrases
and asked nothing else; the re-score is the same answers judged by the current one, which accepts a
synonym for the same mechanism and refuses an answer citing a file the fixture does not hold. The
under-count was **not equal across arms** — base lost five passes to wording, the retrained arm lost
five, the two matched arms four each — and an unequal under-count moves the comparison, which is the
only thing this project measures. `eval-4arms.json` carries both, per task, with every answer.

**Four legs of evidence, one direction.** The fine-tune did not help, on either serving path, at
either data size:

- the retrained experiment arm is **last**, three behind the base, on the repaired scorer;
- it was last on the broken one too — the ordering is not an artefact of the scorer;
- its failures are not close calls: **four of the five are answers that announce they are about to
  look at the files and then stop** ("I'll inspect the code in both files."), which is a run that
did not do the task rather than a run that did it slightly worse;
- and the two matched arms — 1 sample each — also fail to beat the base, so this is not "19 samples
  were too few": **one sample is not enough either, and neither is nineteen.**

**Why the retrained arm is the weak one is visible in what it was trained on.** Each of its nineteen
samples supervises one tool call, ten to nineteen tokens, with the label being the *first* move
(`ls`, then `read`). Twenty epochs at a learning rate of 1e-4 fit that distribution hard — the
recorded loss fell to 0.077 — and what it demonstrably learned is the opening, not the diagnosis.
An arm that says it will look and then stops is what a first-move dataset produces when asked for a
last move. That is hypothesis from the training record and the answers, not a measured mechanism;
what is measured is the behaviour.

## The serving path: a second defect, found in the same sweep

Behind 4-bit NF4 the *same* adapter scores 0/5 and two of its five runs never finish: one spends 89
tool calls in a `web_fetch` loop against a URL that does not exist, the other 18 messages ending in a
`<tool_call>` the loop never accepts. Behind bf16 the same tasks finish in 11–23 s. The base model on
the same 4-bit path is unharmed (3/5).

So the quantised path degrades the *adapter* specifically — the opposite of what one would expect of
a QLoRA-trained adapter served under the same `BitsAndBytesConfig` it was trained with. That is
recorded as a measurement and not as an explanation; **anyone serving this adapter should use bf16**
until it is.

## What this run does not establish

- **It is not a comparison.** No baseline arm was trained against these 19 rows, and no held-out
  evaluation was re-run. The result `runs/chtd-4/` reports — two correct diagnoses of five — belongs
  to that run and is not restated or inherited here. What is restored is the ability to run that
  comparison, not the comparison.
- **It is not bit-reproducible, and was never going to be.** 4-bit quantisation and bfloat16
  accumulation on a GPU are not order-deterministic; `docs/finetune.md` records the same protocol run
  three times to three different adapters. The mean loss here (0.1546) differs from `chtd-4`'s
  (0.2022), and with a differently-tokenised prompt (the +39 above) plus a different sampling path,
  the difference is expected rather than diagnostic. The adapter hash differs too, for the same
  reason.
- **It does not show the method working.** The comparison above has every fine-tuned arm at or below
  the base. `runs/chtd-4` reports the one positive result this project has; nothing here reproduces
  it, and the base's changed behaviour says the setup that produced it is not the setup available
  now. Fifteen tasks put one task at 6.7 points, so the ordering is firmer than it was at five — but
  a 13-versus-10 gap is still three tasks, and a claim that the method *fails* would need more than
  that too.
- **It does not separate "the method" from "the data budget".** Every arm here was trained on one or
  nineteen samples, and none of them beat a base that was trained on none. The literature in
  `docs/why-finetuning-can-hurt.md` says a fine-tune at this scale is expected to do nothing or
  harm; this run is consistent with that and does not distinguish it from a defect in CHTD's design.
  Separating them needs the collection round that produces hundreds of admitted decision points —
  tasks are now 35 training fixtures instead of 24, which is the lever, but the rollouts have not
  been collected.

## Reproducing it

```sh
python -m distill.finetune \
  --plan    /mnt/proj/chtd-experiments/chtd-20261011-a/plan-chtd.json \
  --samples /mnt/proj/chtd-experiments/chtd-20261011-a/round-g.jsonl \
  --base-model /mnt/proj/chtd-experiments/models/Qwen3-4B-Instruct-2507 \
  --output  /mnt/proj/chtd-experiments/chtd-20261011-a/adapter
```

The environment is archived beside the run (`env/chtd-control-venv.tar.gz`) because `/root` — where it
was built — does not survive a reset either, and a recipe without its pinned packages is a recipe
that drifts.

## Reproducing the sweep

The held-out comparison is driven by `eval/holdout_sweep.py`, checked in with this run. It starts
nothing: the arms have to be serving already, one port each, and it points the student route at
them.

```sh
# on the box, one endpoint per arm (bf16 is the path that works; see the 4-bit note above)
cd /mnt/proj/chtd-experiments/chtd-20261011-a/source-48101c4
python serve_openai_bf16.py --model-path /mnt/proj/chtd-experiments/models/Qwen3-4B-Instruct-2507 \
  --adapter-path /mnt/proj/chtd-experiments/chtd-20261011-a/adapter --host 0.0.0.0 --port 8002
python serve_openai_bf16.py --model-path /mnt/proj/chtd-experiments/models/Qwen3-4B-Instruct-2507 \
  --host 0.0.0.0 --port 8003

# on the machine with node and agnes-harness
CHTD_ARMS=adapter:8002,base:8003 python eval/holdout_sweep.py chtd-20261011-a-bf16
```

Scoring calls `harness/tasks/check_diagnosis.py`, the same reader the training gate uses, so a
pass means here what it means there. `CHTD_EVAL_HOST` overrides the box's address.
