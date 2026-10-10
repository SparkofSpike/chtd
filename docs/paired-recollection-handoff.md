# Paired recollection handoff

Use this freeze when collecting the CHTD experimental arm against the new
control-arm evidence in `runs/baseline-20261010-a/`. The old 19-row CHTD run
used another profile and prefix and is not part of this comparison.

| Input | Frozen value |
| --- | --- |
| Train tasks | `harness/tasks/error-recovery.jsonl`, SHA256 `037cad2617b3f53ca9c0b0903b4b046ee833917e066d2f1ad2a73a1fa421a2a8` |
| Holdout tasks | `harness/tasks/final-holdout.jsonl`, SHA256 `adcc47633d497aea32ee1c869766fed5beb5a947efbf9c2f1142967dfbe18e5e` |
| AGH source | commit `a39272bb6bcb3109584e1bd3abac595789ae365d` |
| Builtin preset | `minimal-rl`, SHA256 `b0fe54debc808a2e7552d46984f8123fa6151b8a7e24ff32aa4633c126354954` |
| Profile | `harness/profile/paired-minimal-rl.yaml`, SHA256 `1a5c5f73fa43799b657438efe6d5045bee150dd589ffd983bdd3166f4f363b2e` |
| Observed track | `d403e1e12186a4021cc9d20bbc2c8770cf8366e9fccfc67621a64fde307967ba` |
| Captured student prefix | `harness/prefix/paired-20261010.json`, digest `dbaddc7e2d8ffcfcde19c38f1fc62f6bc96a45336922357d3907b007d0b6edcd` |
| Base model | `Qwen/Qwen3-4B-Instruct-2507`; config, tokenizer and all three weight hashes in `runs/baseline-20261010-a/preflight.json` |

Collect the CHTD arm with `--expected-track d403e1e1…` so the new manifest states the
track outright. The control manifest predates that flag and carries the digest only inside
its task records; the readers accept that fallback, but a manifest that names the track is
the form the protocol asks for.

The student route in that profile is `http://100.89.65.49:8000/v1`. The server
is `serve_openai.py` on the training node: Qwen3 in NF4 4-bit, deterministic
generation capped at 512 new tokens (commit `94960db`; AGH's first-token
deadline is why the endpoint clamps any larger request), `distill.finetune.render_chat`,
and whole SSE tool-call chunks. The AGH smoke session from the local Windows deployment
resolved to the same observed track digest as the control collection. A direct
tool-call probe returned `read({"path":"config.json"})` as an OpenAI tool call.
The canonical cloud copies are under `/mnt/proj/chtd-experiments/`: the control
run is `baseline-20261010-a/`, the base weights are
`models/Qwen3-4B-Instruct-2507/`, and the frozen source is `source-1e08d66/`.
Save the new CHTD-arm ledger, task results, samples, manifest, adapter and logs
under a separate run directory within `/mnt/proj/chtd-experiments/`; do not
write the only copy to the instance's `/root` filesystem.

Collect student states from fresh copies of the 24 train fixtures. Fork the
teacher at selected student decision prefixes, including at least one state
after a student tool result or error. Save parent and child session keys,
prefix digests, task outcome evidence, and label refusals. Apply the same
workspace-relative path filter and manual diagnosis review used for the
control arm. Keep the five holdout tasks unopened until both adapters are
frozen.

The control side has 18 reviewed tasks and 76 candidate decisions. These are
**not yet the selected training rows**. Return the experimental arm's ledger,
task results, admissible sample JSONL, and manifest with per-task counts. The
pairing step will choose only common task coverage and a deterministic common
sample and optimizer budget. If no teacher fork answers a noninitial student
state successfully, the claimed recovery comparison is not runnable.

Credentials belong in local environment variables or a private file. The
profile and experiment artifacts contain no credential value.
