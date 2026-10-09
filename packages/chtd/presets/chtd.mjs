// CHTD's co-harness track: the frozen recipe both the teacher and the student run.
//
// This file is the delivered form of the track, and its exact bytes are frozen:
// `chtd.sha256` records their sha256 and the package refuses to load without a match. The source
// of the recipe, and the reasoning behind each value, is `harness/presets/chtd.yaml` at the
// repository root. Two representations of one track can drift, so `tools/check-parity.mjs`
// compares them key by key and refuses on the first difference. It is a command to run, not a gate
// that runs itself: nothing here invokes it, and the freeze below covers these bytes only.
//
// Why a JS module rather than the YAML: a non-builtin package is installed without its
// `node_modules` and loaded through Host's own entry loader, whose virtual module table exposes
// only `@agnes/extension-api`, `@agnes/protocol` and TypeBox. A package that imported a YAML
// parser would resolve it by luck on a developer machine and fail in a real installation. This
// form needs no parser and no dependency: it is evaluated, not parsed.
//
// The reason the bytes are frozen is the reason the experiment exists. A distillation result is
// only attributable to the model if everything else was identical, so "identical" cannot be a
// judgement call made per edit - it is a hash. CHTD refuses to compare two runs whose track bytes
// differ, and a drifted track fails to load at all.
//
// Every value says who reads it. Several keys have no reader in the harness today; they are kept
// because stating the whole intent of the track is the point, and a reader must not believe an
// unread key is in force.
export const CHTD_TRACK = {
  name: 'chtd',
  // NOT CONSUMED. No code reads `surfaces`; which surfaces may select a recipe is a Profile
  // decision.
  surfaces: ['cli', 'sdk'],
  // route/prompt_sections: read by core's readPreset and host's route materialization. `default` on
  // both fields of a target is the unconfigured sentinel - the deployment's adapter list decides,
  // which is what lets the same track run under a teacher adapter and a student adapter and stay
  // one track.
  // An empty prompt_sections is load-bearing here: no persona, no doctrine, no environment. Any
  // text added here would enter both trajectories and stop the difference between them being the
  // model.
  model: {
    contract_id: null,
    route: { primary: { route: 'default', model: 'default' } },
    prompt_sections: [],
  },
  // Read by core's readPreset. `standard` means the model emits tool calls directly.
  disclosure: 'standard',
  // tools.core: NOT CONSUMED. The tool registry is filled by the extensions the assembly loads,
  // not by this list; it states the intent of a two-tool track, and nothing enforces it yet. That
  // gap matters for CHTD and is recorded rather than papered over: if the deployed assembly ships
  // more tools than this line names, both models see them, so the comparison survives - the track
  // is merely wider than declared.
  // tools.timeout_ms: read by core's readPreset.
  tools: { core: ['shell', 'edit'], timeout_ms: 120000 },
  // NOT CONSUMED. No MCP client reads this yet. An empty server list is the intent: an MCP server
  // added on one side only would break co-harness.
  mcp: { servers: [], defer: false },
  // NOT CONSUMED. Nothing loads skills from a root list yet; the empty list is the intent.
  skills_roots: [],
  // Read by core's readPreset, all four fields.
  compaction: { enabled: true, reserve_tokens: 16384, keep_recent_tokens: 20000, agent_callable: false },
  // preflight / per_request_cap / on_exceed / max_steps: read by core's readPreset.
  // attribution: NOT CONSUMED.
  // max_steps is an experiment parameter, not a preference: it bounds how long a trajectory can
  // be, and therefore how many decision points are comparable. 60 is chosen to admit multi-step
  // work while keeping the tail short enough that divergent runs stay alignable; changing it
  // changes the experiment and requires re-freezing.
  budget: { preflight: 'estimate', per_request_cap: null, on_exceed: 'quote', max_steps: 60, attribution: 'own' },
  // on_unavailable and timeout_ms: read by core's readPreset, and on_unavailable again by base's
  // approval seam. command_policy: validated by the host when a session opens and evaluated by
  // base's approval seam. Empty means nothing is pre-approved, so every destructive call reaches a
  // human.
  // CHTD keeps it empty deliberately: a rule table here would have to name absolute paths, which
  // differ per machine and would defeat the byte freeze. What matters for the experiment is that
  // the policy is identical for teacher and student, and an empty table is identical everywhere.
  approval: { on_unavailable: 'deny', timeout_ms: 60000, command_policy: [] },
  // required: read by the host, which refuses to open a session when the platform cannot provide
  // L1. on_unavailable: read by core's readPreset. level: read by base's sandbox seam
  // (packages/base/extensions/sandbox/src/policy.ts, readSandboxConfig), which refuses a level
  // other than L0 or L1 — so this is a live knob, not a comment.
  sandbox: { level: 'L1', required: false, on_unavailable: 'deny' },
  // Read by core's readPreset as the depth limit. 0 means this track spawns no subagents: a
  // subagent would run its own trajectory and put decisions in it that neither teacher nor student
  // produced, which is exactly the contamination the track exists to remove.
  subagent: { max_depth: 0 },
  // Read by core's readPreset.
  verifier: { default_tier: 0 },
  // enabled: NOT CONSUMED. readPreset reads completion_gate.min_items and nothing else.
  completion_gate: { enabled: false },
  // NOT CONSUMED. No self-refinement loop reads this yet; `false` is the intent for a baseline.
  harness: { auto_refine: { enabled: false } },
  // consent: read by host's readTelemetryConsent (packages/host/src/session-hooks.ts), which
  // refuses a value outside DISABLED/LOCAL/ANON/FULL — so this is a live knob too. invariants and
  // timing are read by core's readPreset; neither is written here, so both take their defaults of
  // false. DISABLED is the stated intent.
  telemetry: { consent: 'DISABLED' },
  // NOT CONSUMED. Hook subscription comes from the extensions the assembly loads. A hook that
  // fired on one side only would be a harness difference wearing the model's name.
  hooks: [],
  // NOT CONSUMED.
  locale: 'en',
}
