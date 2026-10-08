#!/usr/bin/env node
// Freeze and verify the delivered track's bytes.
//
// This mirrors `@agnes/code`'s own `tools/freeze.ts`, and `harness/freeze.py` at the repository
// root, which does the same job for the YAML source. The comparison is over raw file bytes, not a
// parse, so a reordered key, a changed comment and a changed value all fail alike.
//
// Usage:
//   node tools/freeze.mjs check     # verify the recorded hash matches
//   node tools/freeze.mjs write     # record the hash of the current bytes
//
// A drift means the experiment changed. Re-recording is a deliberate act: do it in its own commit,
// and say in the message why the track changed. Changing the track also requires re-running the
// source-vs-delivery parity check, because the two must not diverge.
import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const PRESETS_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'presets')
const TRACK = 'chtd'

const digest = (path) => createHash('sha256').update(readFileSync(path)).digest('hex')

function check() {
  const track = join(PRESETS_DIR, `${TRACK}.mjs`)
  const recorded = join(PRESETS_DIR, `${TRACK}.sha256`)
  let expected
  try {
    expected = readFileSync(recorded, 'utf8').trim()
  } catch {
    console.error(`FAIL: no recorded hash for ${TRACK}; run \`node tools/freeze.mjs write\``)
    return 1
  }
  const actual = digest(track)
  if (actual !== expected) {
    console.error(
      `FAIL: ${TRACK}.mjs has drifted.\n` +
        `  recorded: ${expected}\n` +
        `  actual:   ${actual}\n` +
        'The track changed, so the experiment changed. If that is intended,\n' +
        're-record in its own commit and state why.',
    )
    return 1
  }
  console.log(`ok: ${TRACK}.mjs matches its recorded hash (${actual.slice(0, 12)}...)`)
  return 0
}

function write() {
  const track = join(PRESETS_DIR, `${TRACK}.mjs`)
  let value
  try {
    value = digest(track)
  } catch {
    console.error(`FAIL: track missing: ${track}`)
    return 1
  }
  writeFileSync(join(PRESETS_DIR, `${TRACK}.sha256`), `${value}\n`)
  console.log(`recorded ${TRACK}.mjs -> ${value}`)
  return 0
}

const action = process.argv[2] ?? 'check'
if (action === 'check') process.exit(check())
if (action === 'write') process.exit(write())
console.error('usage: node tools/freeze.mjs [check|write]')
process.exit(2)
