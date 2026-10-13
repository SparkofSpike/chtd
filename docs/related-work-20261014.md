# What to read: the literature CHTD sits in, as of 2026-10-14

This list was assembled on **2026-10-14 by live retrieval** — arXiv API metadata and abstracts, not
from a stored reading list. The field around agent distillation moved fast enough in 2026 that a
list kept in memory goes stale within weeks: the newest entry below, HAD, was posted on 2026-10-02,
six days before this repository's first commit, and it arrives on the same ground CHTD claims.

Every entry was fetched and its abstract read on that date; version numbers and dates are what arXiv
returned. The organising question is not "what is interesting" but "what this project has to answer
for". CHTD's claim has three load-bearing parts — co-harness (§3.1), teaching at the student's
states (§3.2), and distractor stripping plus the harness-stripping ladder (§3.4, §3.5) — and the
sections are ordered by how hard each work presses on them.

## 1. The novelty boundary: work that arrives near this one

Read these first. They are not support; they are the papers a reviewer will name when asking what
CHTD adds.

- **Choi, Moon, Nam, Lee, "Harness-Aware Distillation for Small Language Model Agents"**
  ([2610.02858](https://arxiv.org/abs/2610.02858), 2026-10-02) — the closest work found, posted
  twelve days before this retrieval. Its premise is nearly CHTD's §1: *the harness stays in place
  when a larger agent is distilled into a smaller one, so the student mainly needs the teacher
  abilities the harness cannot provide — and standard distillation instead imitates the teacher's
  full output, treating the harness as part of the input.* HAD's remedy is a contrast between the
  same teacher's actions **with and without harness information**, scored after the student's own
  reasoning, plus a validity check that drops pairs whose preferred action contradicts the harness
  records. It needs no task rewards, success labels or future information. The difference from CHTD
  is the source of supervision: HAD contrasts **the teacher against itself** on harness-supplied
  context; CHTD forks into **the student's own states** and takes the teacher's decision there. That
  distinction — internal contrast versus on-policy state visit — is the shortest honest statement of
  where this project's novelty now sits, and the paper should say so explicitly.

- **Zhao, Ma, Zhang, "Training with Harnesses: On-Policy Harness Self-Distillation for Complex
  Reasoning"** ([2605.08741](https://arxiv.org/abs/2605.08741), 2026-05-09) — uses the
  harness-augmented current model as its own teacher to internalise task-specific harness capability
  into the weights. Two findings matter here. First, +10.83% over OPSD on HMMT25 in a draft–verify
  and a plan–solve setting, which establishes that the "harness as temporary training scaffold"
  direction is already published and productive. Second — and this is the one to keep — **reattaching
  the harness at inference yields no additional benefit and can even degrade performance**. That is
  the same phenomenon CHTD's harness-stripping ladder (§3.5) exists to measure, arriving from the
  opposite direction: OPHSD reports it as a win (the harness is no longer needed), CHTD would report
  it as a transfer result. The ladder should cite it.

- **Chen, Xiao, Zhu, Yuan, Zhang, Wang, "Co-Harness: Co-Evolving Harnesses and Model Weights for
  LLM Agents"** ([2607.22688](https://arxiv.org/abs/2607.22688), 2026-07-17) — the name collides
  with this project directly, and the content is nearly the inverse. A HarnessCritic analyses failed
  trajectories, proposes harness updates, and the improved harness generates the trajectories the
  model is then fine-tuned on — the harness **is optimised**, jointly with the weights. CHTD freezes
  the harness precisely so the model term remains attributable. The two positions are complementary
  and each is coherent: Co-Harness asks what a better scaffold would add, CHTD asks what the model
  adds when the scaffold cannot move. The overlap in name is a real problem for a reader arriving at
  either paper cold, and it should be addressed before submission.

**Applied here.** The co-harness premise — one frozen, byte-identical track shared by both sides so
that only `theta` varies in `f_B(theta_B, H) - f_A(theta_A, H)` — was not found in identical form in
any public work retrieved on this date. HAD keeps the harness fixed but does not require both sides
to run the same bytes; OPHSD treats the harness as a scaffold to be left behind; Co-Harness treats
it as a variable. The specific combination CHTD rests on, *same bytes and a load-time gate so that
"same track" is an attributable fact rather than a convention*, still appears unclaimed. But the
10-02 HAD posting shows the ground is being actively worked, so the delta needs to be written down
before it is discovered by someone else.

## 2. The method §3.2 rests on

The design document cites the DAgger argument for collecting supervision on the student's own state
distribution. That argument now has a 2026 LLM-agent form, and a survey that gives it a formalism.

- **Li, Qiang, Huang, Gao, Zhang, He, Dai, "Revisiting DAgger in the Era of LLM-Agents"**
  ([2605.12913](https://arxiv.org/abs/2605.12913), 2026-05-13) — the modern statement of the
  argument, and the paper CHTD's §3.2 is a harness-level instance of. Turn-level interpolation
  between student and teacher policies, then supervised training on the resulting trajectories with
  teacher labels. On SWE-bench Verified with Qwen3-4B and Qwen3-8B students it improves over the
  strongest post-training baseline by **+3.9 and +3.6 points**; the 4B agent reaches 27.3%,
  outperforming published 8B SWE-agent systems. It motivates the same dilemma CHTD does: SFT has
  dense supervision but covariate shift, RL avoids the shift but has sparse outcome feedback. The
  difference to note: they interpolate at the **turn** level, whereas CHTD forks the teacher into
  the student's exact prefix on a shared ledger. That is a finer-grained intervention, and the
  paper should argue that the granularity is what the harness enables.

- **Song & Zheng, "A Survey of On-Policy Distillation for Large Language Models"**
  ([2604.00626](https://arxiv.org/abs/2604.00626), first posted 2026-04-01, v4 2026-06-18) —
  formalises OPD as f-divergence minimisation over student-sampled trajectories and organises the
  field along three axes: what to optimise, where the signal comes from, how to stabilise training.
  Two things to take from it. Its framing of exposure bias growing roughly with the **square** of
  sequence length is the cleanest available support for CHTD's §1 claim that off-policy transcripts
  do not transfer. And its open-problems section names the failure CHTD's gate addresses:
  teacher distributions are unreliable on out-of-distribution student prefixes, with
  overconfidence and poor calibration especially at exactly those positions.

- **He, Chang, Bhardwaj, Molinari, Konuk, Kautz, Hatamizadeh, "PivotOPD: Learning to Recover from
  Pivotal Mistakes in Multi-Turn Agents"** ([2609.40285](https://arxiv.org/abs/2609.40285),
  2026-09-30, NVIDIA) — CHTD's §6 recommends **error recovery** as the target capability. This paper
  is that capability, executed. Across three Qwen3 models (8B–235B) they find that more than half of
  failed rollouts contain a *pivotal mistake* — an action that moves the agent farther from
  completion — that it typically occurs early, and that it is often recoverable. PivotOPD's teacher
  supplies a gold action at the pivotal turn and recovery actions for the next few turns; reverse KL
  on the first steers away from the mistake, forward KL on the rest transfers recovery behaviour the
  student rarely samples. Best average across 13 baselines on ALFWorld, WebShop and Search-QA, and
  the gains transfer to a Nemotron-3.5 student on SWE-Bench Verified (+3.2% resolve). If CHTD names
  error recovery, this is the paper it is measured against — read it before fixing any run design,
  because its definition of a recoverable mistake is a candidate operationalisation of CHTD's
  divergence gate.

**Applied here.** None of these three invalidates §3.2; together they raise its bar. DAgger-style
training is now a demonstrated LLM-agent method, not an analogy imported from robotics, so the
burden is to show what the shared harness adds beyond turn-level interpolation — and the honest
answer available today is *attributability*: the same bytes on both sides, gated at load, with the
track hash in the lockfile, so a divergence cannot be an artifact of scaffolding drift. That is a
claim about evidence quality rather than about score, and it is the one the deciding experiment in
§5 can actually test.

## 3. Where the teacher's signal is not trustworthy

CHTD's signal-quality gate admits a divergence only when the teacher-side choice is validated by the
outcome. This is not a novel worry; it is the most crowded subfield in the list. The 2026 work is
united on one finding — teacher supervision degrades exactly where the student needs it most — and
divided on the remedy. CHTD's gate is one point in that design space, and the design document should
place it there rather than present it as a standalone safeguard.

- **Chen, Fan, Sun, Yang, Sun, Mao, Qiao, Zhang, Wang, Sun, Hu, Pan, Liu, Zhang,
  "Look Ahead Before You Distill"** ([2608.01953](https://arxiv.org/abs/2608.01953), v2 2026-08-05)
  — the closest to CHTD's gate. High-disagreement states are promising places to teach, but whether
  the guidance helps can only be judged from what follows it; so a short teacher bridge is executed,
  the student continues, and the bridge is kept only if it increases the density of positive
  distillation signal. +16.6 over vanilla OPD and +7.6 over TCOD on average (Qwen3-32B teacher,
  Qwen3-1.7B student; ALFWorld, WebShop, ScienceWorld). This is CHTD's "outcome-validated
  divergence" made prospective instead of retrospective.

- **Liu, Li, Ling, Wang, "When Privileged Guidance Misaligns: State-Matched Routing and
  Contextualized Self-Distillation for Multi-Turn Agents"** ([2608.05219](https://arxiv.org/abs/2608.05219),
  2026-08-05) — names the failure mode CHTD's §3.3 must avoid. When the student's actions or
  subgoal order differ from the reference trajectory, privileged guidance becomes *unreliable for
  the state actually reached* — a state–reference mismatch. SMRC-SD verifies at each turn whether
  the student's execution state matches a supported state along the reference and distils only
  there. With Qwen3-1.7B, ALFWorld success 0.746 → 0.865, WebShop 0.574 → 0.693. The relevant
  question for CHTD: teacher privilege is defended in §3.3 as a leak-proof separation, but this
  paper shows the *relevance* of privileged guidance is itself conditional. CHTD's fork-at-prefix
  construction answers this structurally (the teacher answers at the student's actual state, so
  there is nothing to mismatch), and that should be stated as the advantage it is.

- **Zhong, Zheng, Song, Lin, Sun, Jiang, Wang, Fang, "SOD: Step-wise On-policy Distillation for
  Small Language Model Agents"** ([2605.07725](https://arxiv.org/abs/2605.07725), 2026-05-08) —
  documents the compounding failure: an erroneous tool call cascades, student–teacher divergence
  amplifies, and token-level teacher supervision becomes unreliable or misleading. SOD reweights
  distillation strength per step by divergence — attenuating the teacher in high-divergence regions,
  preserving it in aligned ones. Up to +20.86% over the second-best baseline; a 0.6B student reaches
  26.13% on AIME 2025. This is the *opposite* remedy to CHTD's: SOD trusts the teacher less as
  divergence grows, while CHTD argues divergence is exactly the interface to teach at — given a
  gate. Explaining why the two stances coexist is worth a paragraph.

- **Zhou, Zhang, Wu, Wang, Peng, Liu, Fan, Zhao, "SAGE-OPD: Selective Agent-Guided Intervention for
  Multi-Turn On-Policy Distillation"** ([2606.19659](https://arxiv.org/abs/2606.19659), 2026-06-17)
  — verifier-free selective intervention: observe environment feedback, use teacher judgment to
  decide skip-versus-intervene per student response, and weight token-level distillation by teacher
  confidence. Up to +13.3% relative over standard OPD on ALFWorld unseen. Note the phrase
  *verifier-free* — it achieves gating without the outcome validation CHTD's gate relies on, which
  is the cheaper option a reviewer will ask about.

- **Li, Zheng, Song, Liu, Yang, Sun, Zhong, Guo, Fang, Zhang, Wang, "On-Policy Distillation with
  Curriculum Turn-level Guidance for Multi-turn Agents" (Guided-OPD)**
  ([2606.15912](https://arxiv.org/abs/2606.15912), v2 2026-09-23) — mixes teacher- and
  student-generated turns and decays the teacher's intervention probability to zero along a
  curriculum. The framing is identical to CHTD's premise — small student errors push trajectories
  out of the teacher's familiar distribution, *so supervision is least reliable precisely where the
  student needs it most* — but the fix is scheduling rather than validation. Average relative gains
  of 21.1% Score / 25.5% Success over vanilla OPD on ALFWorld, ScienceWorld, WebShop with a
  Qwen3-30B-A3B teacher; larger gains on smaller students.

- **Tan, Zong, Li, Shi, Li, Chen, "ATOD: Annealed Turn-Aware On-Policy Distillation for Multi-Turn
  Agentic Tasks"** ([2606.27814](https://arxiv.org/abs/2606.27814), v6 2026-08-10) — the OPD→RL
  question. OPD improves fast early then saturates as the student approaches the teacher; RL
  optimises the reward ceiling but is sample-inefficient early. ATOD anneals between them and gates
  the distillation signal by turn-level disagreement and uncertainty. +4.16 points success over OPD
  and +23.62 over GRPO across three student sizes, and — worth noting as a ceiling — it *surpasses
  the corresponding teacher models by 2.16 points* on ALFWorld/WebShop/Search-QA. Relevant to §7's
  non-goal: CHTD runs one round and stops, and ATOD is the evidence that single-round OPD has a
  saturation ceiling, which is what round two would address.

- **Yeo, Choi, Ki, Hwang, "HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents"**
  ([2605.17873](https://arxiv.org/abs/2605.17873), v3 2026-10-02, EMNLP Findings 2026) — uses
  full-trajectory hindsight to select *which* actions were failure-relevant and distils only on
  those spans, instead of feeding back every turn. Up to +13.60 points over dense per-turn feedback
  on BFCL v3 and AppWorld, with a 2.26× reduction in time per training step. Its conclusion —
  *selecting where to distill is key* — is the same thesis as CHTD's divergence-only dataset, which
  is encouraging for §3.4's training layer.

**Applied here.** The training layer of §3.4 (keep only divergence points, discard the rest) is now
the consensus position rather than a CHTD-specific choice; HINT-SD, SAGE-OPD and FutureBridge all
arrive at some version of it. What remains distinctive is the **gate's evidence standard**: CHTD
admits a divergence only when the task outcome or a verifier validates the teacher side, and the
refusal to admit a kept co-mover without a reason is one of the pipeline's four enforced refusals.
The cheaper alternatives now in print — teacher confidence weighting, curriculum decay,
verifier-free selection — will be proposed as ablations, and the paper should say why outcome
validation was chosen over them, or run one of them as a baseline.

## 4. Evaluation and confounds

§1 of the design says the evaluation rewards the wrong thing; §3.4 and §3.5 are the response. The
2026 literature both supports the diagnosis and supplies a warning about the remedy.

- **Zhu, Sun, Yu, Su, "LLM Agent Performance Is Not a Single Evaluation Target"**
  ([2602.03238](https://arxiv.org/abs/2602.03238), v3 2026-08-06) — the theoretical backing for
  §3.4. A benchmark score is shaped by the model *and* the harness, environment, evaluator and
  inference budget; the claim supported by a score depends on the declared candidate boundary and
  condition policy. Their position: "LLM agent performance" does not denote one target —
  model-comparison-under-a-reference-stack, whole-system comparison, and robustness across
  predeclared conditions are three different questions. CHTD's four-group decomposition (target /
  distractor / transfer / regression, §3.4) is an instance of this discipline, and citing this paper
  turns the stripping rationale from a local preference into a methodological position.

- **Wang, Zhu, Hu, Yuan, Chen, Senthil, Hajishirzi, Tsvetkov, Smith, Dasigi, Xiao, "Rethinking the
  Evaluation of Harness Evolution for Agents"** ([2607.12227](https://arxiv.org/abs/2607.12227),
  v4 2026-10-05) — the strongest available caution. Automatic harness evolution fails to outperform
  simple test-time scaling under matched feedback and inference budgets on Terminal-Bench 2.1, and
  generalises poorly to held-out tasks; much prior work also searched using verification signals
  drawn from the same benchmarks it reported on. Their conclusion — *matched-budget baselines and
  held-out evaluation are needed to distinguish genuine harness improvements from benchmark-specific
  search* — is CHTD's §3.5 argument, independently arrived at, and the paper should cite it as
  convergent evidence that the stripping ladder is not optional.

- **Qian, Zhao, Yang, Wang, Qiu, Ji, Savarese, Wang, Heinecke, "AI4AI at Test-Time: Strong-to-Weak
  Capability Transfer via Harnesses"** ([2608.12307](https://arxiv.org/abs/2608.12307), 2026-08-12)
  — strong-to-weak transfer **without any parameter updates**: a builder model iteratively refines
  an inference-time harness for a weaker target, nearly doubling average performance across four
  Theory-of-Mind benchmarks (0.49 → 0.91). The gains come from offloading unstable reasoning into
  deterministic code, benchmark-specific routing and strict answer-format enforcement. This cuts
  both ways for CHTD. It is a threat to any claim that weight-level distillation is *necessary* for
  capability transfer; and it is the strongest argument that harness contributions must be
  separated from model contributions, because here the harness alone accounts for the entire gain.
  A reviewer will ask why CHTD trains weights at all, and §4's L2 justification should engage with
  this paper rather than dismiss scaffold-level transfer.

- **Cloud, Le, et al., "Language models transmit behavioural traits through hidden signals in data"**
  ([Nature 652, 2026-04-16](https://doi.org/10.1038/s41586-026-10319-8); earlier as *Subliminal
  Learning*, [2507.14805](https://arxiv.org/abs/2507.14805)) — a teacher with a behavioural trait
  generates a dataset of *number sequences only*, semantically unrelated to the trait, and the
  student acquires the trait anyway. This is the sharpest constraint on §3.4. CHTD's distractor
  stripping rests on the premise that a signal can be identified and removed from the data; this
  paper shows transmission can occur through channels that carry no semantic relation to what is
  transmitted, so absence-of-content is not absence-of-signal. It does not refute the stripping
  approach, but it means a stripped dataset cannot be assumed clean, and clean-ness has to be
  *measured* on the reported metrics — which, fortunately, is what the four-group decomposition does.

**Applied here.** The distractor analysis in §3.4 is where the paper is most exposed and most
defensible at the same time. Most defensible, because matched-budget and held-out protocols are now
the published standard for exactly this kind of claim. Most exposed, because the Nature result means
"we removed the suspicious fields" is not a sufficient answer — the report has to show the
distractors did not move, and the correlation check described in §3.4 has to be run for every
candidate rather than asserted for the obvious ones.

## 5. Two adjacent routes worth one paragraph each

Not central, but the paper will look incomplete without acknowledging them.

- **Tan, Zhang, Li, Cui, Geng, Zhang, Zhang, Chen, Wang, Wang, Yin, Hu, Zhang, Bai, "SKT: Skill-Use
  Training at Scale via Verified Synthetic Data Generation"**
  ([2608.02287](https://arxiv.org/abs/2608.02287), v1 2026-08-03, v2 updated 2026-10-07) — verified data synthesis for
  skill use: rule-based and agent-based verification with feedback-guided repair, retaining only
  trajectories that substantially use every required skill; 27,164 verified trajectories from 2,000
  public skills. Two of its findings are directly on CHTD's axes: gains **extend beyond a single
  agent interface** in cross-harness evaluation, and they **depend on high-quality supervision**
  (verification ablations). That is a positive transfer result under harness change — exactly what
  §3.5's ladder tests for — obtained in a different pipeline.

- **Kim, Kim, Hwang, "Agent Memory Distillation: Empowering Small LLM Agents with Hierarchical
  Teacher Memory"** ([2608.07169](https://arxiv.org/abs/2608.07169), 2026-08-07) — training-free
  transfer of structured knowledge from teacher trajectories into three memory types (workflow,
  subtask, function), injected proactively or retrieved reactively on errors. Average gains of
  27.2 / 11.2 / 3.4 points on AppWorld, BFCL V3, ToolSandbox with GPT-5-mini as teacher and 4B–8B
  students; 4B students benefit most. The relevant detail: its hierarchy is a manual decomposition
  of what a teacher knows, and it competes with weight-level distillation on the cheap. It belongs
  next to AI4AI in the paper's discussion of why L2 was chosen.

## 6. If only three are read

1. **[HAD, 2610.02858](https://arxiv.org/abs/2610.02858)** — the novelty boundary. Nothing else on
   this list changes what the paper must argue as much, and it is twelve days old.
2. **[Revisiting DAgger, 2605.12913](https://arxiv.org/abs/2605.12913)** — the method's modern
   form and the baseline CHTD's on-policy arm is a refinement of.
3. **[PivotOPD, 2609.40285](https://arxiv.org/abs/2609.40285)** — the recommended target capability,
   already executed at scale, with an operational definition of the mistakes CHTD wants to teach at.

A fourth, if the reader's question is about evaluation rather than method:
**[Rethinking the Evaluation of Harness Evolution, 2607.12227](https://arxiv.org/abs/2607.12227)**.

## Method note

Retrieval was by arXiv API (`export.arxiv.org/api/query`) for metadata and abstracts, plus keyword
search for discovery. Every abstract quoted above was read from the API response on 2026-10-14;
version numbers are those returned. Where a claim comes from a paper's own abstract it is stated as
the paper's claim, not endorsed. No third-party evaluation of the newest entries (HAD, PivotOPD)
was located, so their numbers should be read as their authors' claims without independent
corroboration yet. Judge-by-title risks — in particular assuming two papers with the same name do the same
thing — were the reason each entry was fetched rather than inferred; the Co-Harness collision in §1
is exactly that trap.
