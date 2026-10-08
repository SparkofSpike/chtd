// CHTD's package entry: it delivers the frozen co-harness track to the harness, and nothing else.
//
// Two exports matter to Host:
//   - `presets`      the track, under the name a Profile's `presets.allowed` must list.
//   - `trackMarker`  an inert plugin row. See below.
//
// `trackMarker` carries no behaviour and exists for one reason: a package loaded from an immutable
// runtime snapshot is skipped before it is imported unless its `package.json` declares an
// `agnes.plugins` entry, so a package exporting only `presets` never contributes them. That is
// measured behaviour, not a precaution - the upstream report is AgnesAI-Labs/agnes-harness#420.
// The entry runs nothing; the package's weight is still data. If upstream starts reading presets
// before that early return, this export can go.
//
// No dependencies. A non-builtin package is installed without its `node_modules`, and the entry
// loader exposes only `@agnes/extension-api`, `@agnes/protocol` and TypeBox as virtual modules.
// Everything here is `node:` builtins, so the package installs and loads anywhere the harness does.
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { CHTD_TRACK } from './presets/chtd.mjs'

/** Where this package's recipes live. Both gate functions take a directory so a test can point them elsewhere. */
export const PRESETS_DIR = join(dirname(fileURLToPath(import.meta.url)), 'presets')

/** The one recipe whose exact bytes are frozen, because a track that drifts measures a different experiment. */
const FROZEN = 'chtd'

/**
 * The recorded hash. Missing or unreadable is a refusal that says how to fix itself: the gate runs
 * at module load, so this failure stops every import of the package, and an ENOENT naming a file
 * nobody has heard of is the wrong thing to hand whoever hits it.
 */
export function readFrozenSha256(dir = PRESETS_DIR) {
  const file = join(dir, `${FROZEN}.sha256`)
  try {
    return readFileSync(file, 'utf8').trim()
  } catch {
    throw new Error(
      `E_PRESET_INVALID: the frozen hash for ${FROZEN} is missing or unreadable at ${file}; regenerate it with \`node tools/freeze.mjs write\``,
    )
  }
}

/**
 * The freeze gate: the bytes on disk against the bytes that were signed. It is a byte comparison
 * and not a parse, so a reordered key, a changed comment and a changed value all fail it alike -
 * which is the point. A run measured against this track is comparable to another run only if the
 * track was identical, and "identical" cannot be a judgement call made per edit.
 */
export function verifyFrozenTrack(dir = PRESETS_DIR) {
  const expected = readFrozenSha256(dir)
  const actual = createHash('sha256')
    .update(readFileSync(join(dir, `${FROZEN}.mjs`)))
    .digest('hex')
  if (actual !== expected)
    throw new Error(`E_PRESET_INVALID: ${FROZEN} bytes changed (expected ${expected}, got ${actual})`)
}

/** The recorded hash, exported so a caller can attribute the track a run used. */
export const TRACK_SHA256 = readFrozenSha256()

// The gate runs before the track is handed out, and there is deliberately no way past it: this is
// what the package's named export calls at module load, so a drifted track stops the package being
// imported rather than being reported by a test somebody can skip.
verifyFrozenTrack()

/** The named export the host reads when it assembles this package's recipes. */
export const presets = { [FROZEN]: CHTD_TRACK }

/**
 * Inert by design. It registers no tool, subscribes to no hook and provides no service; it exists
 * so the package is not skipped before its `presets` are read (see the header). It is exported
 * rather than omitted because a declared export that is absent is refused at load.
 */
export const trackMarker = {
  name: 'chtd-track-marker',
  apply() {
    // Nothing to do. The track is data; this row only carries it past the loader's early return.
  },
}
