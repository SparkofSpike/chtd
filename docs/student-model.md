# The student model: Qwen3.5-4B, and why it is not a drop-in

> **Status.** This file is the survey that settled the choice. As of 2026-10-09 the runs
> use **`Qwen3-4B-Instruct-2507`** on the box, because it loads under the torch/transformers
> already installed while Qwen3.5 needs a torch upgrade; the survey below is kept because
> it is what a move to Qwen3.5 would cost, and because the architecture it describes is
> the one Qwen3.6 and Qwen3.8 also use. Where this file says "the student is Qwen3.5-4B",
> read it as "the student is Qwen3.5-4B once the environment moves".

The student is `Qwen/Qwen3.5-4B`. This file records what was measured about it —
chiefly that its architecture is not a standard transformer, which changes how the
LoRA must be configured and how the results should be read.

## Why this model and not Qwen3

The obvious choice was `Qwen3-4B-Instruct-2507`: ordinary `qwen3` architecture,
loads under transformers 4.x, tool-calling template present, same tokenizer as the
rest of the Qwen3 line. It is what the box was originally set up for.

It was set aside for one reason: the run is meant to demonstrate a *current*
capability transfer, and Qwen3's post-training predates the Qwen3.5/3.6/3.8 line.
The newer line is also where the harness's own provider metadata points.

## Why not Qwen3.8

Qwen3.8 is open, but it has no small variant. Measured on 2026-10-09:

| Model | Weights (bf16) | Approx. parameters |
| --- | --- | --- |
| `Qwen3.8-Flash-Next` | 335.28 GB | ~180B |
| `Qwen3.8-27B` | 51.75 GB | ~27.8B |
| `Qwen3.6-27B` | 51.75 GB | ~27.8B |
| `Qwen3.6-35B-A3B` | 66.97 GB | ~36B (MoE) |
| `Qwen3.5-9B` | 17.98 GB | ~9.65B |
| `Qwen3.5-4B` | 8.68 GB | ~4.66B |

The smallest Qwen3.8 is 51.75 GB in bf16, against 47.4 GB of usable VRAM on this
box. It cannot be the student. The Qwen3.5 line is the newest generation with a
size that fits the student role, and it shares the 248320-token vocabulary across
the whole line — so a larger Qwen3.5 (27B, 35B-A3B) can serve as the teacher
without a tokenizer mismatch.

## The architecture is hybrid, and that is the important part

`config.json` for `Qwen3.5-4B`:

```
model_type        qwen3_5
architectures     ['Qwen3_5ForConditionalGeneration']
vocab_size        248320
num_hidden_layers 32
hidden_size       2560
layer_types       [linear_attention x3, full_attention, ...]   # 4-layer period
```

Three of every four layers are **linear attention** (`Qwen3_5GatedDeltaNet`, a
gated delta-rule recurrence); every fourth is conventional attention. The text
model is `Qwen3_5TextModel`; the top-level model also carries a vision tower,
because the checkpoint is multimodal.

### Two consequences that would otherwise be silent

**1. Both attention modules are constructed on every layer, but only one runs.**

```python
# Qwen3_5DecoderLayer.__init__
self.block_type  = config.layer_types[layer_idx]
self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)   # always built
self.self_attn   = Qwen3_5Attention(config, layer_idx)       # always built

# Qwen3_5DecoderLayer.forward
if self.block_type == "linear_attention":
    hidden_states = self.linear_attn(...)
elif self.block_type == "full_attention":
    hidden_states, _ = self.self_attn(...)
```

A LoRA whose `target_modules` is the usual
`["q_proj","k_proj","v_proj","o_proj"]` therefore attaches to `self_attn` on
**every** layer — including the 24 layers where `self_attn` is never called. The
run completes, the loss curve looks like a fine-tune, and three quarters of the
adapted parameters never execute.

The target list must cover both branches:

| Branch | Modules | Layers it applies to |
| --- | --- | --- |
| Linear attention | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj` | 3 of 4 |
| Full attention | `q_proj`, `k_proj`, `v_proj`, `o_proj` | 1 of 4 |
| MLP | `gate_proj`, `up_proj`, `down_proj` | all |

**2. A quarter of the parameters are dead weight.** Because both modules exist on
every layer, the checkpoint carries attention parameters that never run. That is
~2 GB of bf16 held for nothing, and it means parameter counts quoted for this model
overstate the effective capacity.

## Kernels

The linear-attention path imports nothing exotic. transformers guards the optional
kernels and falls back to plain PyTorch:

```python
@use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")
def causal_conv1d_fn(...):
    out = F.conv1d(...)          # pure PyTorch
```

so no `mamba_ssm` / `causal_conv1d` / `flash-linear-attention` build is required.

## What it costs to run it here

`qwen3_5` is supported from **transformers 5.19.0** onward — verified absent from
both 4.57.6 and 5.0.0 by reading the wheels' `CONFIG_MAPPING_NAMES`. And 5.19.0
requires `torch>=2.5`, while the box had 2.4.0+cu118:

```
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.0
```

So the environment moves to `torch 2.5.1+cu118` (keeping the cu118 build keeps
`bitsandbytes 0.43.3` valid, and with it the HAMi compatibility already proven) and
`transformers 5.19.0`. `peft 0.21.2` declares no upper bound on transformers, so it
survives the jump; that was checked in its installed metadata rather than assumed.

`transformers 5.19.0` also needs `huggingface_hub>=1.31.0` (it imports
`ResolvedRevision`), which the box must fetch even though it cannot reach
huggingface.co — the library is needed as an import, not as a transport. Weights
continue to come from ModelScope by local path.
