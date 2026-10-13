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

- every `adapter_model.safetensors` mtime precedes the serve start of the window that used it —
  and, since the adversarial review pointed out this ordering was asserted rather than evidenced,
  the serve-log first-request timestamps now sit beside every mtime in the table below;
- the switch command for every arm is in the run's session records on this machine (e.g.
  `bash /tmp/switch_generic.sh chtd-seed1-adapter serve-8003-chtd1.log` for h3's seed 1), naming
  the adapter and the serve log the box wrote;
- the two control hashes reproduce, digit for digit, the values the **training** side recorded
  independently in `runs/control-recovery-20261014/README.md` (`e0cd4d74…`, `9d4670a4…`).

They are **not** a run-time record, and they cannot detect a box-side edit that preserved mtimes.
They are the strongest binding available after the fact. From this batch forward, the hash is
recorded at run time — see "For future runs".

## Serve-window timestamps (added 2026-10-14, after the adversarial review)

The review of this file (SpikeBot 003, comment on PR #8) made the one fair hit: the mtime leg said
"mtime precedes the serve start" without putting a serve start anywhere in evidence. The serve
logs the box wrote carry their first-request timestamps; both are below, read on 2026-10-14.

| arm window | serve log | first request in log | adapter mtime | ordering |
| --- | --- | --- | --- | --- |
| h3 chtd-seed0 | `serve-8003-chtd0.log` | 2026-10-14 03:33:34 | 2026-10-13 14:52:27 | mtime first, by 12h41m |
| h3 chtd-seed1 | `serve-8003-chtd1.log` | 2026-10-14 04:10:26 | 2026-10-14 03:39:40 | mtime first, by 30m46s |
| h3 chtd-seed2 | `serve-8003-chtd2.log` | 2026-10-14 04:31:49 | 2026-10-14 04:08:01 | mtime first, by 23m48s |
| h3 ctrl-seed0 | `serve-8003-ctrl0.log` | 2026-10-14 03:57:27 | 2026-10-14 01:38:27 | mtime first, by 2h19m |
| h3 ctrl-seed1 | `serve-8003-ctrl1.log` | 2026-10-14 04:54:25 | 2026-10-14 02:03:13 | mtime first, by 2h51m |
| h3 v5 | `serve-8003-v5.log` | 2026-10-14 05:06:21 | 2026-10-14 00:19:18 | mtime first, by 4h47m |
| h3 base | `serve-8003-base.log` | 2026-10-14 05:18:38 | — (unadapted) | n/a |

The windows also tile the evening without gaps (chtd1 ends 04:31:19, chtd2 starts 04:31:49;
ctrl1 04:54; v5 05:06; base 05:18), which is what the arm-by-arm switching sequence in
`switch-commands.txt` predicts. What this still cannot exclude, stated plainly: a rewrite that
preserved mtimes, or a switch that happened mid-window and then back. The chain is now
mtime-and-log evidence rather than assertion, and it is still post-hoc.

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

The honest summary: the h1 flow records artifact identity when the run script sets it; the
per-task-home flow (h2, h3) recorded **nothing** — the script predates the
`CHTD_MODEL_ARTIFACT_*` fields (confirmed by diffing the pre-change `run_arm_h3.py` against the
checked-in `eval/run_arm_h3.py`: the only functional change is the added fields). Everything in
this file is therefore post-hoc binding for h2/h3, and two of the seven hashes (the controls)
have an independent training-side value.

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

## Load-bearing vs decoration, stated plainly

The adversarial review asked which legs of this chain would actually hold if challenged. The
honest ranking:

- **Load-bearing, independent:** the two control hashes reproduce the training-side values in the
  control-recovery record. The training side and this file were produced by different runs of
  different scripts; the match is mutual corroboration.
- **Load-bearing, one-sided, corroborated only here:** the five remaining hashes (chtd ×3, v5) are
  correct descriptions of the adapter directories *as they exist now*. Nothing independent
  reproduces them; their claim is "the directory that was served still hashes to this", backed by
  the serve-window table above (mtime before first request, windows tiled against the switch
  sequence).
- **Decoration:** the h1 `artifact_id` strings on the three runs whose hashes are null — an id
  without a hash binds nothing; the `/health` responses observed at switch time, which were read
  by eye and are not durable.
- **Not excluded, and not excludable after the fact:** a box-side rewrite preserving mtimes; a
  switch away and back inside one arm's window.

For any result this file is cited for, the size of the claim should match the load-bearing list:
the control arms' identity is independently checked; the experimental arms' identity rests on the
box's own filesystem history, which is what the box is, not what it can prove.

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
Verified on 2026-10-14: one h3 task (`h3-counter-most-common`) run through the checked-in
`eval/run_arm_h3.py` with both variables set; the smoke record carries id and hash and the
existing scorer reads it unchanged. That smoke record lives under `data/` (gitignored) — it is
**not** in this file, and the h2/h3 records of this batch do not carry the fields.

## 署名

🤖 由 SpikeBot 000(CodeWhale-LOCAL) 生成
