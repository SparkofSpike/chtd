# Where the artefacts live, and what each copy protects against

This file exists because the project lost four adapters once: they were written to `/root`, the
container was reset, and `/root` does not survive that. The lesson was recorded as a rule — *all
experiment data goes on `/mnt/proj`* — and the rule was followed until a second question turned up that
the rule does not answer: **what happens when the box itself is gone?**

So there are now three copies, and they are not equivalent. Stating which failure each one covers is
the point of writing this down; a backup whose scope is guessed at is a backup that fails at the
moment it is needed.

| Copy | Location | Survives |
| --- | --- | --- |
| working | `cgpai-node3-5:/mnt/proj/chtd-experiments/` | a container reset — the failure that already happened |
| second | `cgpai-node3-5:/mnt/user/chtd-experiments-backup/` | the project directory being deleted, or its mount being lost |
| local | `E:\Coding\chtd\data\` | the box and both of its exports being unavailable |

**All three of the first two sit on the same storage server** (`172.25.70.4`), so neither protects
against that server failing. The local copy is the only one off that machine, and it is the slowest to
maintain — the link between here and the box is a Tailscale DERP relay (the box is behind a symmetric
NAT, so no direct path can be established) and it manages 100–200 KB/s, and intermittently *stalls*
mid-transfer. That is why the local copy is made in 4 MB chunks with a stall detector rather than with
one `scp` per file.

## What cannot be rebuilt, and what can

Training is not deterministic. `docs/finetune.md` records the same plan against the same samples run
three times, producing three different adapters with three different hashes. So:

- **The weights cannot be recreated.** They are the only genuinely irreplaceable artefact, and the
  reason the local copy exists at all.
- **Everything else can.** The fixtures are in git, the round inputs (ledger, forks, review, prefix)
  make the samples rebuildable, and the samples plus a plan make a *new* adapter reproducible in about
  seven minutes of GPU time — a different one, but one that answers the same experimental question.

That asymmetry sets the priority: the small files are what make a rebuild possible, so they are kept
in more places than the weights are.

## The second copy

```sh
# on the box — it is NFS to NFS, so this is local disk speed rather than anything over the relay
SRC=/mnt/proj/chtd-experiments
DST=/mnt/user/chtd-experiments-backup
tar --exclude='./models' -cf - -C "$SRC" . | tar -xf - -C "$DST"

# then a manifest sha256sum can read, and a check
cd "$DST"
find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256 --quiet && echo "ALL MATCH"
```

`models/` is excluded on purpose: 7.6 GB of `Qwen3-4B-Instruct-2507` that ModelScope will serve again,
in a location that a second copy would triple for no protection.

**The manifest must exclude itself.** A first version wrote it with `find … > MANIFEST.sha256`, so the
shell created the file before `find` ran, `find` listed it, and it was hashed while empty — the check
then compared a non-empty file against the hash of nothing and failed. A self-referential manifest is
a small version of the problem this whole arrangement is for.

## The local copy

```powershell
# 4 MB chunks, because a chunk that stalls costs forty seconds instead of half an hour
python tools\pull_artefacts.py

# just one, when the link is slow and only one matters
python tools\pull_artefacts.py --only round20261011-allturns
```

The script verifies each assembled adapter against the sha256 **the box reports for the original**,
not against a hash of the bytes it just wrote — an integrity check that compares a copy to itself
checks nothing. It is resumable: a file already present with the right digest is skipped, so an
interrupted run continues where it stopped.

Writing it surfaced the mode the link fails in: a single 264 MB `scp` was measured frozen at 12,353,536
bytes for over a minute, and eight parallel transfers of 33 MB each moved **zero** bytes in 140
seconds. Smaller chunks with a stall detector fixed it; the same link then ran at 114–171 KB/s
sustained. The capacity was never the problem — the long-lived connection was.

## Checking a copy

```sh
# on the box
cd /mnt/user/chtd-experiments-backup && sha256sum -c MANIFEST.sha256 --quiet

# locally: every adapter, against the hash recorded in its own run.json
python tools\pull_artefacts.py          # reports "already local and correct" per artefact
```

Each adapter's `run.json`, kept beside it in `runs/`, records the sha256 of the weights that run
produced, so a local file can be checked against the record rather than against itself.
