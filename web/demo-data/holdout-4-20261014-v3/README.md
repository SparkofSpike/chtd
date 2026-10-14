# Holdout 4 task set (private)

This fresh diagnostic holdout is for the CHTD teacher-demo dose experiment. Keep this directory, rubrics, and gold repairs out of Git and out of model training prompts.

- 48 task instances across 24 mechanisms, two variants per mechanism.
- Every task has a broken implementation and a check. Gold repairs are used only for fixture self-checks.
- All 48 broken checks fail; all 48 gold repairs pass on Windows and cloud Linux Python 3.8.10.
- Holdout IDs have zero overlap with the 44 task IDs represented in teacher-demo-v5 training rows.
- Paired variants are clustered observations; uncertainty is evaluated across 24 mechanism clusters, not as iid n=48.
- The 9/30/90 row sets are frozen, nested, and stratified by task in dose-row-selection.json.

Mechanical fixture validity is verified. The DeepSeek grader calibration and blinded human-read gate have not run. No model has been trained or evaluated on this holdout.
