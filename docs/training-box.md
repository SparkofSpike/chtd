# The training box

Where the LoRA step runs. Everything here was measured on the machine rather than
inferred from a spec sheet; the measurements are the point of the file.

## Host

| | |
| --- | --- |
| Name | `cgpai-node3` (Tailscale node `cgpai-node3-2`) |
| Address | `100.113.249.5` |
| Access | `ssh root@100.113.249.5`, Tailscale SSH, no key needed |
| Path to it | a Tailscale node on the CHTD tailnet |

A sibling node `cgpai-node3-1` (`100.81.47.75`) is reachable on the tailnet but
refuses SSH: `tailnet policy does not permit you to SSH to this node`. It belongs
to a different tailnet account. `cgpai-node3-2` is the node this project uses.

## The 2026-10-11 move to `cgpai-node3-5`

`cgpai-node3-2` went offline on 2026-10-08 and its Tailscale node key expired with it. The container
was reset, and because `/root` lives inside the container, everything written there went with it:
`/root/chtd_runs/` (all four adapters, including the experimental arm's `chtd-4`), the model cache
under `/root/models/`, and the pinned virtualenv.

The replacement is **`cgpai-node3-5` (`100.84.151.62`)** — same tailnet account, same `ssh root@<ip>`
Tailscale SSH path. The first connection needs a browser check at the URL Tailscale prints; after
that it is unattended. Measured on 2026-10-11:

| | |
| --- | --- |
| GPU | RTX 4090 48 GB — `mem_get_info` reports 47.0 GB free of 47.4 GB |
| CPU / memory | 96 cores, 503 GB |
| `/` | 1.8 TB, 684 GB free |
| `/mnt/proj` | the same 100 TB NFS, 69 TB free |
| Conda | `/root/miniconda3` with `myenv`; a task venv can be layered on it with `--system-site-packages` |

**`/mnt/proj` is the only thing that survives a reset.** The model cache, the adapters and a copy of
the environment itself now live there (`/mnt/proj/chtd-experiments/`), and `/root/chtd_runs/` is not
used again. A run that keeps its weights under `/root` has not finished its work; it has only
finished the training step.

Sections below that name `100.113.249.5` or `/root/...` paths are the measurements taken on the box
that was lost. The hardware and library findings carried over unchanged; the addresses did not.

`tailscale ping` reaches the box over a DERP relay (237 ms), not a direct
connection. Interactive SSH feels the latency; batch jobs do not care.

## Hardware

| | |
| --- | --- |
| GPU | RTX 4090 **48 GB** (a 48 GB board, not the stock 24 GB) |
| Usable VRAM | 47.4 GB total, 47.0 GB free when idle |
| CPU | 96 cores |
| Memory | 503 GB |
| Root disk | 1.8 TB, 751 GB free |
| Shared storage | `/mnt/proj` — 100 TB NFS, 64 TB free |

The GPU reports through **HAMi** (`/usr/local/vgpu/libvgpu.so`, injected via
`/etc/ld.so.preload`), a GPU-virtualisation layer. One consequence matters:

> `torch.cuda.get_device_properties(0).total_memory` returns **0** under HAMi.
> `torch.cuda.mem_get_info()` returns the truth (47.4 GB). Size a run against
> `mem_get_info`, or against a configured budget — not against the property.

The container is long-lived, not rebuilt per session: uptime was 36 days and
`/root/miniconda3` dates to 2024-09-25. Installed packages and downloaded weights
persist.

## Environment

Conda environment **`myenv`** (`/root/miniconda3/envs/myenv`, Python 3.12).

| Package | Version | Why this one |
| --- | --- | --- |
| torch | 2.4.0+cu118 | already present; upgrading to cu124 means a 2.5 GB download |
| transformers | 4.57.6 | last of the 4.x line; matches a torch 2.4 runtime |
| peft | 0.21.2 | |
| accelerate | 1.15.0 | |
| datasets | 5.0.1 | |
| bitsandbytes | **0.43.3** | the last release supporting CUDA 11.8; 0.44+ wants CUDA 12 |
| modelscope | 1.40.1 | the only model source that reaches this box |

Install with:

```sh
/root/miniconda3/envs/myenv/bin/pip install \
  transformers==4.57.6 peft accelerate datasets bitsandbytes==0.43.3 modelscope
```

The pinned versions are a consequence of the torch build, not a preference. If the
box is ever rebuilt with a cu12x torch, bitsandbytes can move forward and this
table should be revisited.

## Network

| Endpoint | Reachable | Notes |
| --- | --- | --- |
| `pypi.tuna.tsinghua.edu.cn` | yes | already the configured index; ~600 KB/s |
| `huggingface.co` | **no** | `curl` returns 000 |
| `hf-mirror.com` | **no** | also unreachable |
| `www.modelscope.cn` | yes | **5.46 MB/s** — 8× the pypi index |

Model weights therefore come from ModelScope. Measured download times at
5.46 MB/s: a 1 GB model ~3 min, a 7B model ~55 min, a 14B model ~100 min.

```python
from modelscope import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/root/models")
# -> /root/models/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master
```

Note the doubled path segment: `cache_dir` gains a `models/` level, and the owner
and name are joined with `--`.

## Two things this host does that break naive scripts

**`/dev/shm` is 64 MB.** A PyTorch `DataLoader` with worker processes dies on it.
Keep `dataloader_num_workers=0`.

**HAMi zeroes `total_memory`.** See above. Anything that auto-sizes from it — some
`accelerate` device-map heuristics among them — will mis-size.

## Verified: the QLoRA path works end to end

`python chtd_smoke.py` on this box, 2026-10-09. It loads a tiny model through the
full path the real student will take — 4-bit NF4, LoRA injection, a real training
step, an adapter on disk — so a failure here is a failure of the machine rather
than of a model.

```
torch 2.4.0 cuda=11.8 avail=True
gpu free=47.0GB total=47.4GB
tokenizer loaded, vocab=151643
model loaded in 4-bit
lora injected: trainable=2,162,688 / total=317,282,176 (0.682%)
dataset ready: 64 rows
starting training
loss 5.5557 -> 3.0392 over 8 steps (paged_adamw_8bit, bf16)
training done: loss=3.8274
saved adapter: adapter_config.json, adapter_model.safetensors, tokenizer.json, ...
peak gpu mem allocated: 1.09 GB
SMOKE_DONE
```

The model was `Qwen2.5-0.5B-Instruct`, chosen because the test should measure the
machine and not the model. 18.6 seconds wall clock, most of it model load.

## Serving a model over the network

`serve_openai.py` on this box exposes an OpenAI-compatible endpoint. It exists
because vLLM would need a CUDA-12 torch rebuild here, while the installed torch and
transformers already work; the script is the smaller step to a working endpoint.

| | |
| --- | --- |
| Endpoint | `http://100.113.249.5:8000/v1` (Tailscale) |
| Start | `bash /root/start_server.sh` or `/root/restart_server.sh` |
| Health | `GET /health` — model id and live VRAM |
| Logs | `/root/serve_8000.log` |
| Pid | `/root/serve_8000.pid` |

Loaded with `Qwen/Qwen3-4B-Instruct-2507`, ~8 GB resident, 39.5 GB free afterwards.

Two implementation notes that decide whether it is useful for an agent rather than
a chat client:

**`tools` are passed into the model's chat template**, and the model's
`<tool_call>` blocks are parsed back into OpenAI `tool_calls`. Verified against three
cases: a prompt that should read a file produced
`read_file({"path": "config.json"})`, one that should run a command produced
`run_shell({"command": "ls"})`, and arithmetic produced no call at all.

**Streaming holds back tool-call blocks.** A `<tool_call>` is structured data; half
of a JSON object is not a useful delta, so it is buffered and emitted whole at the
end. The first implementation leaked `StopIteration` into an executor callback,
which asyncio refuses (PEP 479) — the symptom was a stream that delivered every
token and then never terminated.

Measured throughput: about 19 tokens/s for a 64-token completion, single request,
no batching.

## Stale accelerator accounting, and how it is cleared

Measured 2026-10-09, after three fine-tunes and one script that loaded the model four times.
`nvidia-smi` reported 42.86 GB of the 47.4 GB card charged to **pid 519330**, a process that did not
exist:

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
519330, [Not Found], 42860 MiB

$ ls -d /proc/519330
ls: cannot access '/proc/519330': No such file or directory
```

**A first version of this section said the box needed a container restart or a GPU reset on the
host. That was wrong, and it was written without testing.** What follows is the test.

### The test, and what it showed

Stopping our own serving process was enough:

```
before    nvidia-smi used: 42789 MiB    compute apps: 519330, 42860 MiB
kill our server
          HAMi: Kick dead proc 6348
          nvidia-smi used:     0 MiB    compute apps: (none)
restart our server
          nvidia-smi used:  8875 MiB    compute apps: 535498, 8938 MiB
```

Three things fall out of that.

**Killing our GPU process triggers HAMi's sweep, and it clears everything.** HAMi's own string is
`Kick dead proc %d`; it fired on our process and the whole table went to zero — including the stale
entry. No host-side access is needed, and none was ever needed: the remedy was in our own container
all along.

**The PIDs in `nvidia-smi` are host PIDs, so a PID number says nothing about whose process it is.**
Our own server, which is pid 6890 inside this container, appears as **535498**. Any argument of the
form "519330 is far outside our PID range, therefore it is another tenant's" is worthless — the
ranges are different namespaces.

**It was almost certainly ours.** Everything released when our process died, and the footprint
matches our own activity rather than a stranger's: `measure_lora_for_real.py` loaded the model four
times in one process and reported 27.6 GB free at its end, and three fine-tunes had run before it.
Repeated loading in one process is what accumulates here — `del model` and `empty_cache()` do not
destroy a CUDA context.

### What to do about it

If `nvidia-smi` charges memory to a `[Not Found]` process and a run cannot allocate, restart the
container's own GPU process and re-measure. That is the whole remedy, it takes fifteen seconds, and
it is what cleared 34 GB here.

**Serving is unaffected** — the endpoint kept answering `/health` throughout, and after the restart
a 20 GB allocation succeeded where it had previously failed:

```
free 38.3 GB of 47.4 GB
ALLOCATED 20 GB — free is now 18.3 GB
after release: free 38.3 GB
```

**And the honest limit of this section:** it establishes that the accounting is cleared this way and
that the numbers match our own work. It does not establish that no other container has ever held
memory on this card. The box is one container on a platform that shares GPUs between them — HAMi
exists for that — so "the memory is not ours" is a possibility that should be *tested* rather than
argued from a PID, which is the mistake this section is here to record.
