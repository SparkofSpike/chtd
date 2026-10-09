# Running against Agnes

The hackathon's technical commitment is that the models are Agnes ones, so the teacher
route points at the Agnes hub. This records what that took and what it costs.

Status: **an Agnes teacher drives the frozen track end to end.** Two sessions in one
ledger, both `preset: chtd`, one shared state, one divergence, one admitted entry.

## The shape

```
teacher   agnes-2.5-pro            https://apihub.agnes-ai.com/v1   (hosted)
student   qwen3-4b-instruct-2507   http://100.113.249.5:8000/v1      (the training box)
```

The student is local because it is the side that gets LoRA trained, and a hosted API has
no weights to train. That split is a compliance question, not a technical one — see
*The open question* below.

Generate the profile with `make_agnes_profile.py`; it refuses to write a file that
contains a credential. Keys reach the daemon as `AGNES_SECRET_TEACHER_DEFAULT` and
`AGNES_SECRET_STUDENT_DEFAULT`, through the profile's `secrets: { kind: env }` adapter.

## The two keys are not interchangeable

The hackathon issued a key and the CodeWhale config already held one. Measured, same
endpoint, same request:

```
hackathon key   sk-sNw6...Tue7   HTTP 401  {"error":{"message":"Invalid token ..."}}
config key      sk-Adt...r5uM    HTTP 200  '\n\nOK'
```

Both are 51 characters in the same format. **`/models` is not evidence that a key
works** — with a browser-like User-Agent it answers 200 with the full catalogue and no
credential at all. Only a completion proves anything, which is why
`agnes_credentials.py` probes with one and caches the result.

If the issued key is still refused, that is a question for the organisers (entry U240)
rather than something to work around.

## How the harness names a preset on the command line does not work

```
agh -p --model primary=teacher/agnes-2.5-pro "..."        rc=0   preset: chtd
agh -p --preset chtd --model primary=teacher/agnes-2.5-pro "..."   rc=1
    PRESET_SWITCH_REJECTED: the requested preset or model is not available in this profile
```

The same profile, the same preset, the same model. Omitting `--preset` runs under
`presets.default`, which *is* `cthd`, and the ledger records `preset: chtd` — so the
track loads and the failure belongs to the argument path.

The real error is unrecoverable from the CLI: `mapCore` re-raises any core error whose
code is in

```javascript
REJECT_CODES = new Set(['E_PRESET_UNSUPPORTED', 'E_PRESET_UNRESOLVED',
                        'E_MODEL_UNSUPPORTED', 'E_MODEL_UNKNOWN'])
```

as `PRESET_SWITCH_REJECTED`, and the display layer renders a fixed sentence that drops
the `reason`. `--mode json` and `--meta` do not surface it either. So the working
procedure is: **do not pass `--preset`; verify the preset from the ledger**, where
`session/start` states it plainly.

`--preset standard` is refused too, and correctly so — it is not in
`presets.allowed`.

## Rate limits shape the run

A burst of requests returns 429. Paced at one request every six seconds, ten of ten
succeeded. A rollout therefore needs throttling and retry rather than parallelism: an
agent turn is several model calls, and a 429 in the middle of one is a failed run, not a
slow one.

## What the models can do

All six text models call tools and — more discriminating — refrain when no tool is
needed (`2 + 2` answers `4` without calling anything), which is the behaviour a
distillation target depends on.

| model | tool call | refrains | notes |
| --- | --- | --- | --- |
| `agnes-3.0-flash` | yes | yes | fastest of the strong ones |
| `agnes-2.5-pro` | yes | yes | reasoning field present; used here |
| `agnes-2.5-pro-alpha` | yes | yes | |
| `agnes-2.5-pro-beta` | yes | yes | |
| `agnes-2.5-flash` | yes | yes | |
| `agnes-2.0-flash` | yes | — | called `read_file` twice for one request |

`agnes-2.5-pro` replies carry a separate `reasoning` field, so its `thinkingReplay` is
`drop` — the harness would otherwise replay reasoning as if the model had said it.

## The open question

**The hackathon requires Agnes models and the student is a Qwen.** These cannot both
hold literally: a weight-level fine-tune needs weights, and the Agnes models are
reachable only through an API.

What is true today:

- the **teacher** is an Agnes model, so the supervision comes from the platform the
  competition names;
- the **student** is `Qwen3-4B-Instruct-2507` on the training box, and it is the side
  that receives the LoRA;
- the two are deliberately **not** the same family, which CHTD's design asks for in
  order to rule out tokenizer and pretraining differences.

That last point is smaller here than the design implies. The teacher is consulted over
HTTP, so it reads the student's state as text and answers as text — there is no
logit-level comparison anywhere in the pipeline, and a shared tokenizer would buy
nothing at this interface. What remains unfalsified is the ordinary claim that a
stronger model's choices teach a weaker one; what is *not* available is the design's
control for family.

The boundary is a question for the organisers, and it is worth asking before the result
is written up rather than after.
