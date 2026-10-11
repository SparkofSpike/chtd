# Why a fine-tune can measure worse than its base

The observation this document exists for: in the sweep recorded in `runs/chtd-20261011-a/`, the
adapter matched its base on bf16 (2/5 against 2/5) and collapsed behind 4-bit serving (0/5, two runs
never terminating). A fine-tune that leaves the model no better — or worse — needs an explanation
rather than a shrug, and the literature has several. They are listed here in the order they apply to
this project, from "the measurement cannot support the claim" to "the fine-tune genuinely damaged
something".

None of this is a substitute for the check that matters: a run with enough tasks and enough samples
to see an effect at all. It is the reasoning for building that run.

## 1. The measurement layer: five tasks cannot compare two models

Before any mechanism, the arithmetic. A five-task score is a binomial draw with five trials. Two
correct out of five gives a 95% interval of roughly **5% to 85%**; the difference between 2/5 and
3/5 is one task, and one task is inside that interval. A ranking built on it is noise.

The literature is explicit that this is the normal failure of small evaluations rather than an
edge case:

- **Miller, "Adding Error Bars to Evals"** ([2411.00640](https://arxiv.org/abs/2411.00640)) — eval
  accuracy is a finite-sample estimate with a binomial standard error, and model comparisons should
  use paired differences and clustered errors rather than raw score gaps. The paper's whole point is
  that the gaps people quote on small evals are usually smaller than the error bar.
- **Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in NLP"**
  ([P18-1128](https://aclanthology.org/P18-1128/)) — without a significance test, a small difference
  between two systems is not interpretable, and most published comparisons do not do one.
- **Card et al., "With Little Power Comes Great Responsibility"**
  ([2020.emnlp-main.745](https://aclanthology.org/2020.emnlp-main.745/)) — a large share of NLP
  evaluations are *underpowered*: too little data to detect the effect they set out to detect.
- **Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?"**
  ([2304.15004](https://arxiv.org/abs/2304.15004)) — apparent discontinuities in capability can be
  artefacts of a discontinuous metric sampled over few items, and vanish under a different metric.
- **Maia Polo et al., "TinyBenchmarks"** ([2402.14992](https://arxiv.org/abs/2402.14992)) — even
  estimating a large benchmark from a small subset requires an item-response model; naive small
  subsets do not work.

**Applied here.** This project has now measured the *same unmodified base model* on the *same five
tasks* three times and got **0/5** (`runs/chtd-4`, where it answered "Hello! How can I help you
today?" and stopped), **5/5** (the 2026-10-10 control sweep, `eval-review-20261010-a.json`), and
**2/5** (the 2026-10-11 sweep). The base does not train, so those differences are the measurement.
A noise floor that spans the entire range of the scale is the finding that outranks every score in
this repository: **no comparison on five tasks, between any two of these arms, carries information**,
including the comparisons that look favourable.

The three runs differ in serving path and prompt delivery, which is presumably the cause — but
establishing which variable moved is exactly what a five-task evaluation cannot do.

## 2. The training layer: one to nineteen samples is not a fine-tune

The paired experiment (`matched-20261010-a`) trained each arm on **one** sample for 30 epochs. The
`chtd` arm of `chtd-4` used **nineteen**, for 20 epochs, and its loss fell to 0.077 — a plateau at
one ten-thousandth of the starting value, which is what fitting a handful of token sequences looks
like.

What the literature says about scale:

- **Hernandez et al., "Scaling Laws for Transfer"** ([2102.01293](https://arxiv.org/abs/2102.01293))
  and **Isik et al., "Scaling Laws for Downstream Task Performance"**
  ([2402.04177](https://arxiv.org/abs/2402.04177)) — downstream fine-tuning performance improves as
  a **power law** in the number of fine-tuning examples, with diminishing returns. There is no
  threshold below which the curve stops, but the curve is steep at the small end: adding examples
  early buys the most.
- **Zhou et al., "LIMA: Less Is More for Alignment"** ([2305.11206](https://arxiv.org/abs/2305.11206))
  — the well-known "you only need 1,000" paper. It is worth reading precisely because it is the
  strongest small-data result in the literature and it still used **1,000** curated examples, with a
  65B base model. The paper's claim is about *quality over quantity*, not that tens of examples
  suffice.
- **Chen et al., "AlpaGasus"** ([2307.08701](https://arxiv.org/abs/2307.08701)) — 9k high-quality
  examples beat 52k lower-quality ones, again a data-*quality* result at the thousands scale.
- **Chung et al., "Flan"** ([2210.11416](https://arxiv.org/abs/2210.11416) as summarised in the
  search results; the primary citation is the Flan collection paper) and **Longpre et al., "The
  Flan Collection"** ([2301.13688](https://arxiv.org/abs/2301.13688)) — for generalisation, **task
  diversity** is the lever, and held-out performance improves as the task *count* grows.
- **Biderman et al., "LoRA Learns Less and Forgets Less"** ([2405.09673](https://arxiv.org/abs/2405.09673))
  — LoRA adapts less than full fine-tuning on the target distribution and forgets less elsewhere.
  The relevant half here is the first: **a LoRA at rank 32 over 19 samples has little capacity to
  express anything beyond what those 19 samples show.**

**Applied here.** Nineteen samples × twenty epochs is a LoRA fitted to a single narrow token
distribution. That is a description of the input, not a hypothesis about the weights — and it means
the adapter's behaviour on any task whose prompt is not one of the nineteen is extrapolation, not
generalisation. Whether it extrapolates well is the question the next run exists to answer; whether
it *can* is bounded by how much of the behaviour the samples demonstrate, which is currently one
opening move (`ls`, then `read`).

## 3. The mechanism layer: fine-tuning genuinely can damage a model

Given a real degradation, the literature names several mechanisms. The ones plausible here:

- **Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"**
  ([2202.10054](https://arxiv.org/abs/2202.10054), ICLR 2022) — standard fine-tuning can **distort
  the pretrained representation** and lose out-of-distribution accuracy, to the point where training
  only a linear probe on frozen features beats full fine-tuning under distribution shift. Their
  remedy, **LP-FT**, first fits a head on frozen features and only then fine-tunes. This is the
  clearest published statement of "the fine-tune made it worse and here is why". Held-out tasks in
  this project are, by construction, out-of-distribution relative to the nineteen training fixtures.
- **Gekhman et al., "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?"**
  ([2405.05904](https://arxiv.org/abs/2405.05904), EMNLP 2024) — fine-tuning on examples carrying
  knowledge the base model did not have increases hallucination, and the effect scales with the
  amount of such knowledge in the data. Their recommendation is to filter fine-tuning examples to
  those the model can already produce. **Relevant to this project's phrase-check failure mode**: the
  control arm's answers invented `src/student/src` paths and JS contracts that do not exist in the
  fixtures, which is that failure shape.
- **Luo et al., "An Empirical Study of Catastrophic Forgetting in LLMs During Continual Fine-tuning"**
  ([2308.08747](https://arxiv.org/abs/2308.08747)) — sequential fine-tuning degrades earlier tasks,
  and the paper studies how. This project has one round, so classical continual forgetting does not
  directly apply; it matters for round two onward.
- **Ouyang et al., InstructGPT** ([2203.02155](https://arxiv.org/abs/2203.02155)) — the source of
  the term **"alignment tax"**: alignment training can cost performance on public NLP benchmarks, and
  their mitigation was mixing pretraining gradients into the RLHF objective.
- **Gudibande et al., "The False Promise of Imitating Proprietary LLMs"**
  ([2305.15717](https://arxiv.org/abs/2305.15717)) — imitation fine-tuning copies **style** without
  conferring the teacher's capability or factuality, and the resulting models look better to human
  raters than they measure. This is the paper most directly about this project's design: it is the
  argument for CHTD's own premise (co-harness, on-student-state data) over transcript imitation, and
  it is also the warning that a style-copying fine-tune can *look* aligned while failing at the task.
- **Shuttleworth et al., "LoRA vs Full Fine-tuning: An Illusion of Equivalence"**
  ([2410.21228](https://arxiv.org/abs/2410.21228)) — LoRA and full fine-tuning are not
  interchangeable in what they learn or how they generalise, despite similar loss.

**What the mechanism layer does *not* explain.** The 4-bit collapse in the 2026-10-11 sweep — the
same adapter at 0/5 with 89 `web_fetch` calls against a non-existent URL, versus 2/5 at bf16 — is not
covered by any of the above. Failure by *looping* rather than by *being wrong* is a different
phenomenon, and it stays unexplained and recorded as such.

## 4. The method layer: CHTD's premise is sound, its data budget is not

The design's core — collect teacher decisions at the states the **student** reached — is DAgger, and
DAgger's guarantees are conditional on how much of that data you have:

- **Ross, Gordon, Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret
  Online Learning"** ([AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)) — training on
  the learner's own state distribution gives a no-regret bound rather than the compounding-error
  behaviour of behavioural cloning.
- **Ross & Bagnell, "Efficient Reductions for Imitation Learning"** (2010) — the error of naive
  behavioural cloning can grow **quadratically** in the task horizon, which is the failure CHTD
  exists to avoid.

**Applied here.** DAgger's bound is asymptotic in the number of rounds and the number of learner
states visited. This project has one round and, per the pairing manifest, one admitted label. The
method is not what is failing; the method has not yet been run at a scale where its claim is
testable. That is the gap the next collection round has to close, and it is a data-budget problem
before it is an algorithmic one.

## What follows

Each layer above implies the same three moves, in order:

1. **Tasks: tens, not five.** Both the training fixtures and the held-out set. The held-out set binds
   the resolution of any claim; at five tasks, one task is 20 points and the base model's own run-to-run
   range spans the scale.
2. **Samples: hundreds of decision points, not one or nineteen.** Since a sample is one divergence,
   the count grows with *decision points*, not with tasks: tasks have to induce multi-step failures,
   and the collection has to keep going until the admitted set is large enough to train on without
   fitting the prompts.
3. **A scorer that measures.** The 2026-10-10 control sweep is the argument: the machine phrase check
   scored the control arm 4/5 while a human reviewer scored it 0/5, because answers that **invented
   files absent from the fixture** contained the expected substrings. A scorer that rewards
   confabulation cannot rank arms at all. Until it is fixed, no arm comparison in this project —
   favourable or unfavourable — should be quoted as a result.

The order matters. Fixing the scorer without adding tasks still leaves a measurement that cannot
resolve a difference; adding tasks without fixing the scorer produces more tasks scored the wrong way.
