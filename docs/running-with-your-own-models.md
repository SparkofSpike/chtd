# Running CHTD with your own models

The recorded experiments used one pair: **`agnes-3.0-flash`** as the teacher and
**`qwen3-4b-instruct-2507`** as the student, served with vLLM-compatible endpoints. Nothing in the
pipeline requires that pair — this page lists every place a model enters, what is configurable
today, and the two gates that will refuse you if the pieces disagree about which track you are on.

## Where a model enters the pipeline

| stage | what runs | how to point it somewhere else |
| --- | --- | --- |
| **Profile** | teacher and student are two *routes* in one AGH profile | edit the profile's `provider.routes`; the route names (`teacher`, `student`) are what the other stages refer to |
| **Collect** | `harness/collect_baseline.py` runs one arm at a time | `--arm teacher\|student` picks the route; `--model <id>` overrides the model id (defaults: `agnes-3.0-flash` / `qwen3-4b-instruct-2507`); the value used is recorded in the run's `collection-manifest.json` |
| **Build dataset** | `distill/reviewed_forks.py` turns reviewed forks into training rows | reads the model names from the two collection manifests and writes them into each row's provenance — no literal to edit; `--preset` declares the frozen track the data was collected on (default `minimal-rl`) |
| **Match arms** | `distill/select_matched.py` freezes equal-count samples and writes training plans | `--base-model` / `--teacher-model` are recorded into the generated plans (defaults are the recorded pair) |
| **Train** | `distill/finetune.py` | `--base-model <path-or-hub-id>` is required, so this stage has no default to fight; `--plan` carries the recipe |
| **Serve** | `serve_openai.py` / `serve_openai_bf16.py` | `--model-path`, `--model-id`, `--adapter-path` |
| **Evaluate** | `eval/holdout_sweep.py`, `eval/run_arm_h3.py` | `CHTD_STUDENT_MODEL` (default `primary=student/qwen3-4b-instruct-2507`), `CHTD_PRESET` (default `minimal-rl`), plus the endpoint env vars below |

The `primary=<route>/<model>` form is AGH's per-session model selector: the left side is the model
slot, the right side names the profile route and the model within it.

## Environment variables the evaluation flow reads

Both eval scripts follow one pattern — a default matching this project's recorded setup, overridable
by environment. Nothing needs editing to move to another machine:

```
CHTD_STUDENT_MODEL          primary=student/<your-model-id>      (the per-session selector)
CHTD_PRESET                 minimal-rl                            (the frozen track)
CHTD_EVAL_HOST              host serving the arm                  (default 127.0.0.1)
CHTD_EVAL_PORT              its port                              (default 8006)
CHTD_EVAL_ARM               label written on every record         (default base)
CHTD_PROFILE_SRC            the profile to expand                 (holdout_sweep)
CHTD_PROFILE_TEMPLATE       the profile to expand                 (run_arm_h3)
CHTD_PROFILE_STUDENT_URL    the URL inside it that gets rewritten to your endpoint
CHTD_TASKS_JSONL            the task rows to run
CHTD_TASKS_DIR              the fixture directory
CHTD_RUN_ROOT               where records and workspaces land
CHTD_SHARED_AGH_HOME        the AGH home (ledger) for this arm
CHTD_COLLECTION_MANIFEST    the frozen selection manifest to check the task list against
CHTD_AGNES_CLI / CHTD_NODE  the agnes entry point and node binary
```

## Two gates that will refuse a mismatch

These are checks, not configuration — they exist because a drifted track or a mixed-up provenance
makes a comparison meaningless, and they fail closed:

1. **The frozen track.** The preset's bytes are hash-checked at load; a drifted preset fails to
   load. The dataset builder refuses a training prefix whose preset is not the one you declared
   (`--preset`), and `CHTD_EXPECTED_TRACK` (where set) makes the sweep refuse a run whose observed
   track differs — so two arms can only be compared if they ran the identical recipe. If you change
   the preset, change it *for every arm*, declare it at each stage, and record the new hash.
2. **Provenance.** `distill/reviewed_forks.py` refuses a fork that did not come from the student
   boundary, a divergence that predates the fork, a review that does not cover every task, and a
   training prefix that is not the declared preset. The dataset records the student and teacher
   model ids it was built from — taken from the collection manifests, so they follow *your* models
   rather than this project's.

## A worked example

To evaluate your own student model on this project's 15-task screen:

```bash
# 1. serve it (any OpenAI-compatible endpoint works)
python serve_openai_bf16.py --model-path /path/to/your/model --port 8006

# 2. point the sweep at it
export CHTD_STUDENT_MODEL="primary=student/<your-model-id>"
export CHTD_EVAL_HOST=127.0.0.1
export CHTD_EVAL_PORT=8006
export CHTD_EVAL_ARM=my-arm
export CHTD_SHARED_AGH_HOME="$PWD/data/rollouts/_my-home"
export CHTD_COLLECTION_MANIFEST="$PWD/data/holdout-1-check-20261013/selection-manifest.json"

# 3. run it
python eval/holdout_sweep.py my-run-20261015
```

The profile you expand must declare a `student` route whose `baseUrl` matches the endpoint
(`CHTD_PROFILE_STUDENT_URL` is the string the sweep rewrites to `http://$CHTD_EVAL_HOST:$CHTD_EVAL_PORT/v1`).

## What is deliberately *not* configurable

- **The recorded numbers.** Every `runs/**/README.md` and every checked-in score table is a record
  of what was actually run, with the model ids and track hashes in it. Those are evidence, not
  settings.
- **The provenance fallbacks.** Where a manifest predates the `model` field,
  `distill/reviewed_forks.py` falls back to the pair this project recorded, so old datasets rebuild
  identically. A new run's manifest carries your model ids and the fallback never fires.

## If you hit something hardcoded that this page does not name

Two files are project-maintenance tools rather than pipeline stages, and they carry this project's
machine paths: `tools/pull_artefacts.py` (fetches weights from the private training box over
Tailscale; the host and the artefact list are this project's) and the `--output-root` default in
`distill/select_matched.py`. Both are safe to edit for your own setup — they are not imported by the
pipeline and no test depends on their defaults.
