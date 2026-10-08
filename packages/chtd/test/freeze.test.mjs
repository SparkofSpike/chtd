// The freeze gate's own tests. Zero dependencies, so they run anywhere Node does.
//
// The gate is the only thing standing between a drifted track and a run that is compared to
// another run as if it were the same experiment, so its refusals are tested rather than assumed.
import assert from 'node:assert/strict'
import { cpSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { PRESETS_DIR, TRACK_SHA256, verifyFrozenTrack } from '../index.mjs'

/** The gate takes a directory so it can be pointed at a copy; the delivered bytes stay untouched. */
const withCopy = (fn) => {
  const dir = mkdtempSync(join(tmpdir(), 'chtd-track-'))
  try {
    cpSync(PRESETS_DIR, dir, { recursive: true })
    return fn(dir)
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

const readTrack = (dir) => readFileSync(join(dir, 'chtd.mjs'), 'utf8')

test('the delivered track matches its recorded hash', () => {
  assert.match(TRACK_SHA256, /^[a-f0-9]{64}$/)
  assert.doesNotThrow(() => verifyFrozenTrack())
})

test('a one-character edit in a comment is refused', () => {
  withCopy((dir) => {
    // A comment, not a value: the gate compares bytes, so it cannot tell the difference and must
    // not try to.
    writeFileSync(join(dir, 'chtd.mjs'), readTrack(dir).replace('the frozen recipe', 'the frozen recipe!'))
    assert.throws(() => verifyFrozenTrack(dir), /E_PRESET_INVALID/)
  })
})

test('a changed value is refused', () => {
  withCopy((dir) => {
    writeFileSync(join(dir, 'chtd.mjs'), readTrack(dir).replace('max_steps: 60', 'max_steps: 61'))
    assert.throws(() => verifyFrozenTrack(dir), /E_PRESET_INVALID/)
  })
})

test('a missing recorded hash names the command that writes one', () => {
  withCopy((dir) => {
    rmSync(join(dir, 'chtd.sha256'))
    assert.throws(() => verifyFrozenTrack(dir), /tools\/freeze\.mjs write/)
  })
})
