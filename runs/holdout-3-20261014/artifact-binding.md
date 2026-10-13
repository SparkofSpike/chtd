# Artifact binding for the 2026-10-13/14 runs

The adversarial review of `control-recovery-20261014` (SpikeBot 003, comment on PR #7) named the
sharpest open item in this batch: four of five h1 runs carry no `artifact_sha256`, and nothing
durable binds the served adapter to the records. `eval/holdout_sweep.py` already reads
`CHTD_MODEL_ARTIFACT_ID` / `CHTD_MODEL_ARTIFACT_SHA256` from the environment — the run scripts of
the day set only the id — and the per-task-home flow recorded neither. This file closes that gap
for the whole batch, h1, h2 and h3, as far as it can be closed after the fact.

## What this is, and what it is not

The hashes below were computed on cgpai-node3 on 2026-10-14, **after every run closed**, from the
adapter directories the box served. Three things support them as the bytes each run used:

- every `adapter_model.safetensors` mtime precedes the serve start and the sweep that used it
  (mtime column below), so no adapter was rewritten between its run and this hash;
- the switch command for every arm is in the run's session records on this machine (e.g.
  `bash /tmp/switch_generic.sh chtd-seed1-adapter serve-8003-chtd1.log` for h3's seed 1), naming
  the adapter and the serve log the box wrote;
- the two control hashes reproduce, digit for digit, the values the **training** side recorded
  independently in `runs/control-recovery-20261014/README.md` (`e0cd4d74…`, `9d4670a4…`).

They are **not** a run-time record, and they cannot detect a box-side edit that preserved mtimes.
They are the strongest binding available after the fact. From this batch forward, the hash is
recorded at run time — see "For future runs".

## The adapters

All paths under `/mnt/proj/chtd-experiments/` on cgpai-node3. `sha256` is over that adapter's
`adapter_model.safetensors` (264,308,896 bytes for every row).

- **base** — unadapted, no adapter.
- **chtd-adapter** (the h1/h2 `chtd arm`, h3 `chtd-seed0`)
  `node4-recollection-20261012/matched-v2/chtd-adapter`
  `c1b4ebfdbef62a17c52ca059ff0887d191fd02ddb14c5e56f44768d52fab436c` — mtime 2026-10-13 14:52
- **chtd-seed1-adapter** (h3 seed 1)
  `node4-recollection-20261012/matched-v2/chtd-seed1-adapter`
  `793685aee0b3022f11435bacd9c8f8b08a808096ae4c8019a2cf758672502a80` — mtime 2026-10-14 03:39
- **chtd-seed2-adapter** (h3 seed 2)
  `node4-recollection-20261012/matched-v2/chtd-seed2-adapter`
  `0d61e500076cba6f8d2ed4133ae04dee616f5740c0a5b3040c9e92ff61f9e9c9` — mtime 2026-10-14 04:08
- **baseline-assistant-adapter** (ctrl seed 0, h1/h2/h3)
  `node4-recollection-20261012/matched-v2/baseline-assistant-adapter`
  `e0cd4d74589c41c88d011172028276543e34a588bef2388cfe2da4a268057f7a` — mtime 2026-10-14 01:38
- **baseline-assistant-seed1-adapter** (ctrl seed 1, h1/h2/h3)
  `node4-recollection-20261012/matched-v2/baseline-assistant-seed1-adapter`
  `9d4670a4d9dbecd3af94d5a6ad98dca9fbe3fc7adc140c26b0c11cabe8093c61` — mtime 2026-10-14 02:03
- **baseline-adapter** (the control as previously trained — the 0/15 collapse)
  `node4-recollection-20261012/matched-v2/baseline-adapter`
  `1694dd43758e2586df1d8eea822c21c6431229e197df807ec870e377b4607624` — mtime 2026-10-13 14:34
- **adapter-v5** (the 93-row teacher-demo arm, h1/h2/h3)
  `teacher-demo-20261012/adapter-v5`
  `027552e8abccd2cdb67dfedb59d4aaf75c384b49c9f63d570b38d6f03e951ba3` — mtime 2026-10-14 00:19

## What each run record carries today

| run | record | artifact_id | artifact_sha256 |
| --- | --- | --- | --- |
| ctrl-assistant-h1-20261014 | `data/rollouts/…/run.json` | `matched-v2-baseline-assistant-adapter` | `e0cd4d74…` (at run time) |
| ctrl-seed1-h1-20261014 | `data/rollouts/…/run.json` | `matched-v2-baseline-assistant-seed1-adapter` | null |
| chtd-arm-h1-20261014 | `data/rollouts/…/run.json` | `matched-v2/chtd-adapter` | null |
| v5-h1b-20261014 | `data/rollouts/…/run.json` | `teacher-demo-20261012/adapter-v5` | null |
| h2 (5 arms) | `data/holdout-2-20261013/records-*.json` | — (per-task records only) | — |
| h3 (7 arms) | `data/holdout-3-20261014/records-*.json` | — | — |

`data/` is a working tree on this machine (`.gitignore` keeps it out of the repository): the
right-hand column is its state on 2026-10-14, and this file is the durable copy.

One h1 run does carry the complete pair: `ctrl-assistant-h1-20261014` was swept with both
variables set, and its `e0cd4d74…` matches the hash computed for this file two days later — the
path works when the script sets it.

## For future runs

`eval/holdout_sweep.py` (the h1 flow) and `eval/run_arm_h3.py` (the per-task-home flow, the
script the h3 runs used, now checked in) both read:

```
CHTD_MODEL_ARTIFACT_ID=<name>
CHTD_MODEL_ARTIFACT_SHA256=<sha256 of the adapter's adapter_model.safetensors>
```

Take the hash on the box before starting the serve:

```
ssh root@100.84.151.62 "sha256sum /mnt/proj/chtd-experiments/<adapter>/adapter_model.safetensors"
```

The server's `/health` response also names the adapter path it loaded, so the run-time
confirmation can be checked against the record by eye — but the record is what persists.
Checked end-to-end on 2026-10-14: one h3 task (`h3-counter-most-common`) run through
`eval/run_arm_h3.py` with both variables set; the record carries id and hash, and the existing
scorer reads it unchanged.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
