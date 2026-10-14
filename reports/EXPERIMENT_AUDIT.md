# Experiment audit, 2026-10-14

**Verdict: FAIL for a confirmatory/pre-registered comparison; WARN for exploratory recorded phrase-gate counts.** The arithmetic is reproducible, but the run deviated from the existing preregistration, the checker is lexically gameable, and CHTD seed 1 includes an unrecorded first attempt followed by a retry.

- **Ground truth — WARN.** Task targets are frozen synthetic phrase lists in `data/holdout-4-20261014-v3/tasks.jsonl`, not derived from predictions. The intended human primary endpoint in `preregistration.json` was not run. Classification: synthetic lexical proxy.
- **Metric and leakage — FAIL for semantic claims.** `harness/tasks/check_diagnosis.py` checks phrases and identifiers, not meaning. The clean control seed 0 `h4-csv-bom-1` answer is factually wrong yet passes. Earlier runs exposed `check.py`; those are excluded and now labelled in private audit metadata. Current staging excludes that file and has a regression test.
- **Result arithmetic — PASS.** Recorded rows reproduce Base 34/48, controls 34/48 each, CHTD seed 0 7/48, CHTD seed 1 38/48, and paired results 0/27/21 and 6/2/40.
- **Attempt integrity — FAIL for one-shot completeness.** CHTD seed 1 has 33 + 15 disjoint recorded rows for all 48 task IDs, but an additional unrecorded near-timeout session for `h4-form-decode-2` was retried. The score excluding that task is 37/47; paired result remains 6 wins, 2 losses, 39 ties.
- **Provenance — WARN.** Evaluation run manifests do not pin adapter artifact IDs or hashes; the seed 1 shards have no run manifests. Do not treat the intended adapter labels as independently verified weights.
- **Preregistration and scope — FAIL/WARN.** The existing preregistration specifies a teacher-demo 9/30/90-row dose ladder and human correctness, not these seeded CHTD/control phrase-gate arms. Treat this run as post hoc exploratory. Only two seeds were evaluated; CHTD scores vary 7/48 vs 38/48.

**Claim impact:** The report may state the recorded phrase-gate counts with all caveats. It cannot support a semantic correctness claim or a general CHTD superiority claim. A confirmatory run needs a prospective matching protocol, pinned adapter hashes, attempt logging, and human or validated semantic grading.

**Review source:** fresh same-family audit agents, provisional; deterministic counts cross-checked against the recorded JSONL files.
