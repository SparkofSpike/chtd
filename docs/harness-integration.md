# Running Agnes Harness against a real model

The harness is what CHTD measures both sides inside, so getting a real session to
run end to end is the gate on everything downstream. This file records what that
took on this machine, including three obstacles that are properties of the harness
rather than of CHTD.

Status: **a real session runs, against a real model, on another machine.** The
ledger it produced is readable by `harness/ledger.py`.

## The shape that works

```
this machine (MISTERVAN)                          training box (cgpai-node3)
  agh CLI  ──┐
             ├─ daemon (local, IPC)                 serve_openai.py
  AGNES_PROFILE=local-dev                          Qwen3-4B-Instruct-2507
  profile: local-dev/profile.yaml                   :8000
             └── provider route ── Tailscale ────►  http://100.113.249.5:8000/v1
```

The model is served from the box with an OpenAI-compatible surface; the harness
knows nothing about where it runs, because a route is a URL and a model id.

## The three obstacles

### 1. The daemon needs a private DACL on its home

`agh daemon start` failed with `local daemon startup failed: daemon child exited
before readiness (1)` and nothing in the audit log, because the child died before
it could write one. Running `daemon.mjs` by hand surfaced the real message:

```
Private directory validation failed
```

The harness requires `AGH_HOME` to carry a DACL whose allow entries name only the
current user, SYSTEM or Administrators. A directory created under a normal Windows
tree inherits `Authenticated Users` and `BUILTIN\Users`, which fail that check.

```powershell
icacls $AGH_HOME /inheritance:r
icacls $AGH_HOME /grant:r "$env:USERNAME:(OI)(CI)F"
icacls $AGH_HOME /grant:r "NT AUTHORITY\SYSTEM:(OI)(CI)F"
icacls $AGH_HOME /grant:r "BUILTIN\Administrators:(OI)(CI)F"
icacls $AGH_HOME /grant:r "$env:USERNAME:(OI)(CI)F" /T /C
```

### 2. A profile name must be a builtin template name

`name: chtd` is refused: the resolver passes the name through as the builtin
template to merge over, and only `local-dev` and `enterprise` exist. The file
belongs at `<home>/profiles/local-dev/profile.yaml`. The checked-in
`harness/profile/chtd.yaml` carries the CHTD content; its `name:` field is what a
deployment override replaces.

### 3. Installing a package needs a TTY

`agh package add` cancels itself when driven from a pipe — `confirmPackageInstall`
returns false unless stdin and stdout are both TTYs. Driving it through a ConPTY
and answering the prompt is what makes a scripted install possible:

```python
from winpty import PtyProcess      # pip install pywinpty
proc = PtyProcess.spawn(['node', AGNES, 'package', 'add', 'file:./packages/chtd'], ...)
# ... answer the `[y/N]` prompt
```

## The credential path, and why it goes through the environment

A route with no credential is refused before any request is sent:

```javascript
if (!keyless && !this.resolveCredential && this.credentialFor(route) === void 0)
  yield { code: 'AUTH', message: `route=${route}` }
```

and `keyless` alone is not enough, because the wire library refuses to send an
unauthenticated request — the failure then degrades to `AUTH status=?`, with no
HTTP status because no request left the machine.

So a credential is mandatory even for a server that ignores credentials. Declaring
`credentialRef: secret://student/default` satisfies the check, but the **file**
store cannot resolve it here: `readWindowsSecret` calls `hasPrivateDaclSync`, which
comes from the harness's native module, and on this machine that module exports
only `abiVersion`. Every DACL helper is absent, so the read throws, and because the
file store reports a *reason* (`private-file`) rather than a plain miss,
`composeSecrets` will not fall through to its environment store either.

The working configuration sidesteps the file store entirely:

```yaml
adapters:
  secrets: { kind: env }
```

```powershell
$env:AGNES_SECRET_STUDENT_DEFAULT = "ollama-local-no-auth"
```

`env` resolves `secret://student/default` from `AGNES_SECRET_STUDENT_DEFAULT` with
no filesystem call. The daemon inherits it from the shell that starts it.

### What this means for the native module

Functions that need it — private file reads, detached spawn, process identity — are
unavailable until it is rebuilt against a Node header cache:

```sh
export AGNES_NODE_HEADERS=/path/to/node-headers   # ...\include\node\node_api.h
pnpm --filter @agnes/system-node build:native
```

The symptom of a stale build is an export list containing `abiVersion` and nothing
else: the addon loads and passes its ABI check, so the failure appears late and in
an unrelated feature.

## Two details worth keeping

**`credentialRef` belongs to the route declaration, not the model record.** The
model record also carries a `route:` line, so a text edit that matches `route:
teacher` can land one level too deep and be silently ignored —
`resolveCredentials` reads `decl.credentialRef` off the route declaration only.

**A ledger from a real session reads back through CHTD's own pipeline.** Verified:
`open_ledger` + `read_trajectory` on the session this produced reports
`preset: standard`, one decision point, and the assistant message as its action,
with a prefix digest. That is the first non-synthetic ledger the CHTD code has
consumed.

## The chain, end to end on real models

With the two fixes below in place, one full pass works: two rollouts, an alignment,
a gate decision and a dataset with a digest.

```
student  qwen3-4b-instruct-2507   "Hello!HowcanIassistyoutoday?"
teacher  deepseek-v4-pro         read({"path":"notes.txt"})

same state: 935ef0d446090af60b5748a48b5cb847fa5440ff945ad705ad0e40f3546b0e1f

1 divergence, 0 agreements, 1 novel   →   admitted: 1, refused: 0
dataset digest e442209ef00125f9bf4dba472de1a9518ba6f547f367dcd16b2dad61d9c42fd9
```

Two bugs surfaced doing it, and neither was visible from the test suite — the
fixtures were too tidy for either.

### An empty `assistant/message` was counted as state

The harness writes `assistant/message` with `content: []` and
`stopReason: "tool_use"` immediately before a `tool/call`. `_action_of` already
declined to read it as a decision; `is_conversation` read it as state. deepseek
wrote the empty row and Qwen did not, so from that row onward every prefix digest
differed **while every state row before it was identical**:

```
state chain up to seq 15   student 3316f0c56118f5ae9910a4d9c81d
                           teacher 3316f0c56118f5ae9910a4d9c81d   MATCH
state chain up to seq 16   student 3316f0c56118f5ae9910a4d9c81d
                           teacher 4e0fe04c23c91ea191b38fdd939d   teacher gained seq 16
```

The symptom was a pipeline that found zero divergences for a reason unrelated to
the models. `harness/trajectory.py` now routes both questions through one
`_message_text`, so a row cannot be a state without being a decision again.

### `--teacher LEDGER:SESSION` split inside the session key

`rpartition(":")` divides at the last colon, and an Agnes key is full of colons:

```
agnes:local:local-dev:cli:workspace:8ad1758e07cde6e0
```

so the ledger path absorbed everything up to the final segment and the command
failed with `no ledger at <path>:agnes:local:...`. The split now takes the
rightmost colon whose left side names an **existing file**, which a session key's
prefix never is.

## The configuration that satisfies both constraints

Two requirements pull against each other, and the resolution is not obvious:

- **The prefix must align**, so nothing differing between the runs may enter the
  conversation.
- **The session keys must differ**, because `OutcomeTable` is keyed by
  `(session_key, prefix_digest)` and cannot record "the teacher's branch succeeded
  while the student's failed" if both branches share a key.

A session key is derived from the workspace path, so sharing a directory satisfies
the first and breaks the second. Separate directories satisfy the second — and
under the `standard` preset they break the first, because that preset injects a
`[runtime context]` message naming the cwd, the model, the route and the session
key. It describes the model to itself, which in a co-harness experiment is the
variable under test.

| preset | runtime context | cwd changes the prefix | notes |
| --- | --- | --- | --- |
| `standard` | injected | yes | unusable for a paired comparison |
| `minimal-rl` | none | no | what the rollouts above used |
| `chtd` | none (`prompt_sections: []`) | no | blocked on the package, below |

With `minimal-rl` the two runs take **separate workspaces**, so their keys differ
and the alignment holds anyway. This is also the closest available stand-in for
CHTD's own frozen track: `minimal-rl` is the shipped baseline whose
`prompt_sections` is empty for the same stated reason.

## Still open

**The frozen track cannot be loaded.** `@chtd/track` declares a preset, its
load-time freeze gate works, and it installs and trusts through the normal flow —
but a trusted, enabled package with no immutable runtime snapshot stops the daemon
at the `packages` step with `E_EXT_LOAD: trusted package has no immutable runtime
snapshot`, and every session becomes unrunnable. A snapshot laid down by hand is
discarded when the daemon next starts.

The consequence is stated rather than hidden: the rollouts above ran under
`minimal-rl`, so their `resolvedPresetHash` is **not** CHTD's frozen track, and a
dataset built from them is a demonstration of the pipeline rather than an
attributable result. Closing this is what makes the numbers mean something.

**The native module is incomplete.** It exports `abiVersion` and nothing else, so
`hasPrivateDaclSync` and its siblings are missing and the `file` secrets adapter
cannot work. Rebuilding needs a Node header cache:

```sh
export AGNES_NODE_HEADERS=/path/to/node-headers   # ...\include\node\node_api.h
pnpm --filter @agnes/system-node build:native
```

This is a prerequisite for the track package, not an alternative to fixing it.
