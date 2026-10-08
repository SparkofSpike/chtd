#!/usr/bin/env node
// Proves the delivered track and its readable source say the same thing.
//
// The track exists twice on purpose. `harness/presets/chtd.yaml` is the source a person reads and
// the bytes `harness/freeze.py` freezes; `presets/chtd.mjs` is what a run actually loads. Two
// representations can drift, so this check is what keeps "the same track" a fact rather than an
// intention. Run it whenever either side changes.
//
// A YAML parser is needed and is deliberately not a dependency of this package - the package must
// install and load with no dependencies at all (see index.mjs). Point CHTD_YAML_FROM at any
// directory whose node_modules provides `yaml`; every Agnes Harness checkout has one:
//
//   cd /path/to/agnes-harness && node /path/to/chtd/packages/chtd/tools/check-parity.mjs
import { createRequire } from 'node:module'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { CHTD_TRACK } from '../presets/chtd.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const YAML_SOURCE = join(HERE, '..', '..', '..', 'harness', 'presets', 'chtd.yaml')

const requireFrom = createRequire(join(process.env.CHTD_YAML_FROM ?? process.cwd(), 'noop.js'))
let parse
try {
  parse = requireFrom('yaml').parse
} catch {
  console.error(
    'FAIL: no YAML parser available.\n' +
      '  This check needs one because the readable source is YAML; the package itself must not.\n' +
      '  Run it from a directory whose node_modules provides `yaml`, or set CHTD_YAML_FROM:\n' +
      '    CHTD_YAML_FROM=/path/to/agnes-harness node tools/check-parity.mjs',
  )
  process.exit(2)
}

const { readFileSync } = await import('node:fs')
const source = parse(readFileSync(YAML_SOURCE, 'utf8'))

/** Order-insensitive, and reports the first path that differs rather than a wall of JSON. */
function firstDifference(a, b, path = '') {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return `${path || '(root)'}: array vs non-array`
    if (a.length !== b.length) return `${path || '(root)'}: length ${a.length} vs ${b.length}`
    for (let i = 0; i < a.length; i += 1) {
      const d = firstDifference(a[i], b[i], `${path}[${i}]`)
      if (d) return d
    }
    return null
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)])
    for (const k of keys) {
      if (!(k in a)) return `${path}.${k}: only in the delivered track`
      if (!(k in b)) return `${path}.${k}: only in the YAML source`
      const d = firstDifference(a[k], b[k], path ? `${path}.${k}` : k)
      if (d) return d
    }
    return null
  }
  if (a !== b) return `${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`
  return null
}

const difference = firstDifference(source, CHTD_TRACK)
if (difference) {
  console.error(
    `FAIL: the YAML source and the delivered track disagree.\n  ${difference}\n` +
      'One of them changed without the other. Fix the pair, then re-record with\n' +
      '`node tools/freeze.mjs write` if the delivered bytes moved.',
  )
  process.exit(1)
}

console.log('ok: harness/presets/chtd.yaml and presets/chtd.mjs describe the same track')
console.log(`    keys compared: ${Object.keys(CHTD_TRACK).join(', ')}`)
