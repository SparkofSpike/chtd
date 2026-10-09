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

## Leaked accelerator memory, and why it stops the next training run

Measured 2026-10-09, after three fine-tunes. The card is 47.4 GB and `nvidia-smi` reported 42.86 GB
of it charged to **pid 519330**, in a container where the highest live pid was 6,641.

```
$ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
519330, [Not Found], 42860 MiB

$ ls -d /proc/519330
ls: cannot access '/proc/519330': No such file or directory
```

`[Not Found]` is `nvidia-smi`'s own word for it: the process is gone and the driver still charges
its memory. A CUDA context outlived the process that owned it.

**HAMi does not reclaim it.** The library names the mechanism — `strings /usr/local/vgpu/libvgpu.so`
contains `Kick dead proc %d` and `Owner proc dead (%d), try fix` — and it fired once earlier in this
box's life, for a different pid. It did not fire here, and it did not fire when a request could not
be satisfied:

```
$ python -c "import torch; torch.empty(20*1024**3//4, device='cuda')"
HAMI-core ERROR: cuMemoryAllocate failed res=2
CUDA out of memory. Tried to allocate 20.00 GiB. GPU 0 has a total capacity of
47.37 GiB of which 5.21 GiB is free. Process 519330 has 41.86 GiB memory in use.
```

The allocation failed rather than triggering a reclaim, and the free figure did not move. `/dev/shm`
and `/tmp/vgpulock` are both empty, so the accounting lives on the host side and is not reachable
from inside the container.

**What still works and what does not.** Serving does: the endpoint holds its ~12 GB, answers
`/health`, and runs sessions. Training does not: the last run peaked at 18.3 GB and there are 5.2 GB
free. **The box needs its container restarted, or the GPU reset on the host, before another
fine-tune can run** — neither is possible from inside, and both are the operator's call.

Worth knowing before blaming the next run's failure on the model or the data: the symptom is a
CUDA out-of-memory error naming a pid that does not exist.
