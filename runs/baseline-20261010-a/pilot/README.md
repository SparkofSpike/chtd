# Exploratory baseline LoRA run

This run trained on all 76 manually reviewed baseline candidate decisions from
18 tasks. It was started before the CHTD arm was recollected and paired. It is
therefore a training-path smoke result, not the strict control adapter for a
two-arm performance claim. The five holdout tasks were not evaluated.

The exact inputs are `../pilot-plan.json` and `../sft/samples.jsonl`. The cloud
run used the frozen Qwen3-4B-Instruct-2507 weights recorded in `../preflight.json`.
The run record is `run.json`; `train.log` is the captured trainer output.
For this pilot, the plan's `dataset_sha256` field names the candidate sample
JSONL digest because there is no matched two-arm dataset manifest yet.
The command on the training node was:

```sh
cd /root/chtd-control
/root/chtd-control-venv/bin/python -m distill.finetune \
  --plan runs/baseline-20261010-a/pilot-plan.json \
  --samples runs/baseline-20261010-a/sft/samples.jsonl \
  --base-model /root/models/models/Qwen--Qwen3-4B-Instruct-2507/snapshots/master \
  --output runs/baseline-20261010-a/pilot-adapter
```

Environment: Python 3.12.5; torch 2.4.0; transformers 4.57.6; peft 0.21.2;
accelerate 1.15.0; datasets 5.0.1; bitsandbytes 0.43.3; NVIDIA GeForce
RTX 4090, 49,140 MiB reported device memory, driver 595.58.03.

The 253 MiB adapter is stored on the training node at
`/root/chtd-control/runs/baseline-20261010-a/pilot-adapter/`. Its SHA256 is
`f119c29e3dcf972ac3ab47ecc533301bcdf5e33a7f2b8aa31c7e5a739a661eb2`.
The adapter was loaded into `serve_openai.py` and answered a fresh, unrelated
single-word probe with `READY`. This verifies loading and inference only.

A second copy of the complete run directory is on the account-mounted
`/mnt/user/chtd-control-backup/baseline-20261010-a/`. The copied adapter's
SHA256 was checked against the original value above. This copy is on the
account mount; its long-term retention still depends on the cloud provider.

Once the CHTD arm has been reviewed, select common tasks and a common sample
and optimizer budget; then train both adapters from the same base checkpoint.
Do not substitute this pilot adapter for that matched baseline run.
