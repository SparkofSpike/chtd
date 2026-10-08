# @chtd/track — the delivered co-harness track

This package delivers CHTD's frozen track to Agnes Harness and does nothing else. It contributes
one preset, `chtd`, which both the teacher and the student run.

```
index.mjs              the package entry: freeze gate, `presets` export, inert marker row
presets/chtd.mjs       the track itself — the bytes that are frozen
presets/chtd.sha256    their recorded sha256
tools/freeze.mjs       records or verifies the hash
tools/check-parity.mjs proves this package and the readable YAML source say the same thing
```

## Why the track is a JS module and not the YAML

`harness/presets/chtd.yaml` at the repository root is the source a person reads, and the file
`harness/freeze.py` freezes. It cannot be what a run loads.

A non-builtin package is installed **without its `node_modules`** (`packages/package-manager/src/sources.ts`
excludes it), and its entry is loaded through Host's own loader, whose virtual module table exposes
only `@agnes/extension-api`, `@agnes/protocol` and TypeBox
(`packages/host/src/ext-host/loader.ts`). A package that imported a YAML parser would therefore
resolve it by luck on a developer machine and fail in a real installation.

So the delivered form needs no parser and no dependency: it is evaluated, not parsed. Every import
in this package is a `node:` builtin.

Two representations of one track can drift, so `tools/check-parity.mjs` compares them key by key
and refuses on the first difference. Run it whenever either side changes.

## Why there is an inert plugin row

`trackMarker` registers no tool, subscribes to no hook and provides no service. It exists because a
package loaded from an immutable runtime snapshot is skipped before it is imported unless its
`package.json` declares an `agnes.plugins` entry — so a package exporting only `presets` never
contributes them. That is measured behaviour rather than a precaution:

```
packages/package-manager/src/package-plugin-loader.ts   no declarations -> returns before importing
packages/host/src/assemble/packages.ts                  no loaded rows  -> returns undefined
packages/host/src/assemble.ts                           falls back to { id, extensionEntry }, no presets
```

Reported upstream as AgnesAI-Labs/agnes-harness#420, with an end-to-end reproduction against
`assemble()`. The row runs nothing; the package's weight is still data. If upstream starts reading
presets before that early return, this export can go.

## The freeze gate

The gate runs at module load, mirroring `@agnes/code`'s own `verifyFrozenPreset`. A drifted track
therefore stops the package being imported rather than being reported by a check somebody can skip.

```sh
node tools/freeze.mjs check    # verify
node tools/freeze.mjs write    # re-record, after a deliberate change to the track
```

The comparison is over raw file bytes, not a parse: a reordered key, a changed comment and a
changed value all fail alike. Re-recording is a deliberate act — do it in its own commit and say in
the message why the track changed.

## Tests

```sh
node --test test/*.test.mjs
```

The suite covers the gate's refusals: a one-character comment edit, a changed value, and a missing
`chtd.sha256`. It uses only `node:` builtins, so it runs wherever Node does.

## Installing

The package is consumed through the harness's normal package flow — it is not published to a
registry. From a directory with a copy of this repository:

```
agnes package add file:./packages/chtd
```

The Profile then names it in `packages` and adds `chtd` to `presets.allowed`. Because presence and
integrity are lockfile facts, the exact track a run used is attributable afterwards.
