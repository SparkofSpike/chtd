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
`credentialRef: secret://student/default` satisfies the check, but the **file** store
could not resolve it here, and the reason is worth stating precisely because the
first diagnosis of it was wrong.

`readWindowsSecret` checks the store directory and the namespace directory:

```javascript
for (const path of [root, join(root, ns)]) {
  const stat = lstatSync(path)
  if (!stat.isDirectory() || stat.isSymbolicLink() || !hasPrivateDaclSync(path))
    throw Object.assign(new Error('unsafe secret directory'), { code: 'EACCES' })
}
```

`hasPrivateDaclSync` was reported as missing — that was a mistake, and the correction
is below — but it does return `false` for a directory that merely *looks* private:

```
hasPrivateDacl(<home>)              = true
hasPrivateDacl(<home>/secrets)      = true
hasPrivateDacl(<home>/secrets/student) = false     <-- the namespace directory
```

The check requires **`SE_DACL_PROTECTED`** on each directory, not just a short access
list. `icacls /inheritance:r` sets that flag; `icacls /grant:r` does not. The store
and its namespace were granted the right principals, but the namespace was left
inheriting, so the flag was absent and the read was refused. And because the file
store reports a *reason* (`private-file`) rather than a plain miss,
`composeSecrets` will not fall through to its environment store either — a refusal is
deliberately not a miss.

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

Fixing the file store instead means protecting **every** directory in the chain:

```powershell
icacls <dir> /inheritance:r
icacls <dir> /grant:r "$env:USERNAME:(OI)(CI)F"
```

for the store and each namespace beneath it — the flag is per-directory and does not
propagate.

### The native module was never missing anything

A probe once listed `Object.keys()` on `agnes-system.node`, saw only `abiVersion`, and
concluded the DACL helpers had not been compiled. **They are present but
non-enumerable.** `typeof native.hasPrivateDacl === 'function'`, and it answered
correctly for every path it was asked about. The conclusion was wrong; the code was
never the problem. What the probe should have reported is the `SE_DACL_PROTECTED`
requirement described above — a fact about the directories, not about the module.

Rebuilding is still worth having, and it needs no Visual Studio project system: the
script locates `vswhere`, calls `VsDevCmd.bat`, and invokes `cl.exe` with everything
stated explicitly.

```sh
export AGNES_NODE_HEADERS=/path/to/node-headers   # include/node/node_api.h + x64/node.lib
dist/native/agnes-system.node                    # 228864 bytes
```

On Windows the layout the script wants is exact: `<root>/include/node/node_api.h`
and `<root>/x64/node.lib`. Both come from nodejs.org for this Node version —
`node-v<version>-headers.tar.gz` and `win-x64/node.lib`.

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

## The frozen track loads, and what it took

`preset: chtd` now runs. The evidence, from the ledger itself:

```
student  session=...396121f2   preset=chtd   1 decision
 teacher  session=...5d6b1a2b   preset=chtd   3 decisions

shared state 935ef0d446090af60b5748a48b5cb847fa5440ff945ad705ad0e40f3546b0e1f
  student: Hello!I'm ready to assist you with your queries.
  teacher: I'll read the file and check the second line.

1 divergence, 0 agreements, 2 novel   ->   admitted: 1, refused: 0
track a37b90fe31115418c17c8294622df7eac37511abd9283356ea464828682b39d5
```

That digest is `packages/chtd/presets/chtd.mjs` — the bytes the package's own gate
checks at load — not the YAML the repository keeps beside it. The two are different
files describing one track; `packages/chtd/tools/check-parity.mjs` is what holds them
together.

Two things were in the way, and one of them was a conclusion of mine that was wrong.

### A hand-written runtime snapshot poisons the store

The snapshot store is created and pinned by the daemon, during activation. An attempt
to lay one down by hand — computing the hashes, copying the tree, writing the state
file — appeared to work and then failed in a way that looked like a missing file:

```
E_EXT_LOAD: trusted package has no immutable runtime snapshot
```

while the state file plainly listed the snapshot and validated. Enable failed. What
the hand-written entry also carried was a **guessed `requestHash`**, and
`pinRuntimeSnapshotStore` compares that field when a pin is re-taken — so the daemon's
attempt to pin its own snapshot hit `pin-id-reused`, the operation failed, and the
failure path collected the snapshot back out of the store.

That is why a snapshot laid down by hand kept disappearing at daemon start: it was
being cleaned up as the residue of a failed operation, and the failure was mine.

The working sequence needs nothing clever:

```
agh package add file:./packages/chtd     # through a pty, see below
agh package trust <id> <integrity> <capabilityHash>
agh package enable <id>                  # the daemon builds and pins the snapshot
```

`enable completed (100%)` and the state file went from four snapshots to five, with
`@chtd/track` among them — created by the harness, not by me. **Do not hand-write
entries in `.agnes-runtime-snapshots.json`; the store is a private implementation and
the daemon owns it.** The correct answer to "the snapshot is missing" is to make the
enable succeed.

### The one real limit: `--standalone` cannot load a packaged track

```
agh -p --standalone ...   ->  E_EXT_LOAD: trusted package has no immutable runtime snapshot
agh -p ...                ->  runs, preset: chtd
```

Same profile, same store, same session. The flag runs the host embedded in the CLI
process, and that path does not consult the runtime-snapshot store, so a
non-builtin package can never be loaded through it. Drive sessions through the
daemon. `CliRunner.standalone` therefore defaults to false.

## A session is identified by its working directory

`sha256(cwd)` — that is the whole derivation:

```javascript
function sessionKey4(profile, actor, cwd) {
  const scope = createHash('sha256').update(cwd).digest('hex').slice(0, 16)
  return `agnes:${actor.org || 'local'}:${profile.name}:cli:workspace:${scope}`
}
```

Two consequences, and both bit before they were understood.

**Re-running a task in the same directory reuses the session.** The harness finds the
existing key and appends to it, so the second run produces no new session row —
observed directly: one workspace accumulated four turns of the same prompt. Clearing
the directory does not help, because the key is the path and not the contents. A run
that needs its own trajectory needs its own directory, which is why `collect` names
workspaces with a timestamp.

**A run that created no session must not be given an older one.** `_newest_session`
returned the newest row in the ledger; with a reused session that is a *previous*
run, and one arm's answers get recorded under another's name with nothing to notice
it by. It now takes the moment the run started and raises if the run created nothing.
A wrong answer that announces itself is worth more than a plausible one that does
not.

## Still open

**The observation above is a single task with a single divergence.** The pipeline is
verified end to end; the *result* is not yet a finding. That needs the real task set
and enough rollouts for the comparison to say something — which is the next step, not
a caveat about this one.

**The native module build is not reproducible from the repository.** `build:native`
works, given headers, but nothing records which headers or when — and this machine is
now running a locally built artifact whose `.prebuild` sibling is the one the harness
shipped with. That is a provenance gap worth closing before results depend on it.
