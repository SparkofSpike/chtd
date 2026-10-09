# Strict baseline-versus-CHTD protocol

## Question and units

The intervention is the source of teacher-labelled states: the teacher's own
trajectory (baseline) or states reached by the base student (CHTD). The outcome
is held-out task success by an adapted student. A task is the paired evaluation
unit; multiple decisions from one task are training examples, not independent
evaluation replicates. Train both adapters from the same unadapted checkpoint.

## Gates before either training run

1. **Fix the task split first.** Record immutable training and holdout task IDs,
   fixture digests, prompts, checks, and any manual adjudication rubric. Do not
   use the holdout to select labels, recipes, checkpoints, or thresholds.
2. **Pin the common runtime.** Record AGH commit, package and preset bytes/hash,
   resolved profile, tool disclosure, system prompt, step/turn budget, model
   routes, tokenizer, base weight-file digests, and generation settings. Both
   collection arms must use the same frozen track and initial fixture bytes.
3. **Collect two real state distributions.** Run the teacher independently for
   the baseline. Run the base student first for CHTD, then fork the teacher at
   *each* selected student decision prefix. Record the parent/child session
   keys and verify that each fork answered that prefix. An independent teacher
   run from the initial workspace only covers shared early states; it does not
   substitute for later forks.
4. **Gate labels symmetrically.** Keep only decisions whose teacher branch has
   an observed successful outcome or a predeclared verifier verdict. Apply the
   same safety/path filters to both arms. Store rejected examples and reasons.
   Never put outcomes, reference answers, or teacher-only context into the
   student's input. Record coverage at non-initial states and after tool errors.
5. **Match the training budget.** Predeclare deterministic sampling when arm
   yields differ. Match task coverage, admitted example count, supervised token
   budget or optimizer updates, LoRA hyperparameters, maximum length, seed,
   package versions, and checkpoint-selection rule. Archive exact sample
   digests and the selection manifest; do not choose the better seed afterward.
6. **Validate data before GPU use.** Re-render both arms with the serving chat
   template, verify prefix tokens and tool schemas, reject truncation, and
   compare train/holdout task and fixture hashes for leakage. Refuse empty arms
   or an arm with no teacher-answered student recovery state.

## Training and evaluation

7. Train baseline and CHTD independently from the same base weights. Record
   commands, code commit, sample/plan digests, seed, GPU and package versions,
   loss, step count, adapter hashes, and failures. Repeat with predeclared seeds
   if compute permits; seed runs are training replicates, not extra held-out
   tasks.
8. Evaluate both adapters, plus the unadapted base, on identical untouched
   holdout fixtures and the same AGH track. Reset the workspace for every run;
   interleave or randomize arm order to limit time/endpoint drift. Save ledgers,
   responses, checks, latency and token usage. A missing or blocked run is
   undecided, not an automatic failure.
9. Review each diagnosis blind to arm. The current phrase scorer has observed
   false positives and false negatives; report both the automatic check and a
   rubric-based adjudication, with disagreements and full response references.
   Report per-task paired outcomes, gains/losses, and uncertainty; five holdout
   tasks are only a pilot, not a precise success-rate estimate.
10. Repeat the paired evaluation under a different preset for harness-transfer
    evidence. If this is not run, restrict the claim to the frozen track. Keep
    competition compliance separate: the current event rule requires Agnes
    model calls, while the existing Qwen student runs do not meet that rule.

## Current gate result (2026-10-10)

- GPU and existing QLoRA path: available on the training box.
- Existing CHTD samples: 19 latest rows, all at the same initial decision
  prefix, with no tool result before the teacher label. They do not test
  recovery from a student's mistake.
- Baseline samples/adapter and raw AGH ledgers/outcomes: absent from the
  inspected local checkout and training-box paths.
- AGH source: freshly cloned at `a39272bb6bcb3109584e1bd3abac595789ae365d`;
  lockfile dependencies and the Windows CLI build pass with Node 24.19.0 and
  pnpm 10.34.5. A local platform diagnostic runs, but no model route or
  collection profile has been configured.
- Agnes credential and eligible trainable student weights: not configured in
  the inspected runtime. Never put a key in a repository or experiment log.

The two-arm training gate is therefore closed. Reusing the 19 first-decision
rows as if they were recovery states, or fabricating baseline rows from them,
would answer a different question. The next executable step is to recover the
original ledgers/configuration or recollect both arms with the fork protocol.
