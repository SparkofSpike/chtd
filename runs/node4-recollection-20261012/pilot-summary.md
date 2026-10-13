# CHTD paired training pilot — 2026-10-12

This is an after-turn recovery pilot: AGH forks only after a completed turn. It does not test
teacher correction at every in-turn decision prefix.

| Evaluation condition | Phrase-check passes |
| --- | ---: |
| Unadapted Qwen3-4B base | 12/15 |
| Teacher-own trajectory LoRA | 0/15 |
| CHTD student-state LoRA | 12/15 |

On the 15 untouched holdout tasks, CHTD had one pass the base missed and one pass the base had,
for a net change of zero. Against the teacher-own LoRA, CHTD gained 12 tasks and lost none. The
judge is an automatic expected/forbidden phrase matcher, not independent human adjudication; these
numbers are proxy outcomes, not ground truth.

Both LoRA arms used the same Qwen3-4B base fingerprint, the same nine common training tasks, 30
epochs, 60 optimizer steps, seed 0, and the same LoRA recipe. A tokenizer dry-run showed that all
samples exceeded the originally planned 4096-token limit, so the shared limit was frozen at 8192
before either arm trained. The training track was `dc46c09dd798bc4e58edc9a833915860284a77291d2e7218a4137251f60db0c6`; the three evaluation arms shared evaluation track
`fccb56144c1ed282371f74ceadeb14e841797605f1ddbd92f6a4fe33e0c6d87f` and one profile.

This is a small pilot: one seed, nine training tasks per arm, and 15 holdout tasks. Both training
losses approached zero on the tiny datasets, so overfitting is a material limitation. Two teacher-
own collection tasks lacked valid sessions and were excluded before freezing the common split.
Full run manifests, raw evaluation ledger, logs, adapters, and checksums are archived under
`/mnt/proj/chtd-experiments/node4-recollection-20261012` and are intentionally not committed here.
