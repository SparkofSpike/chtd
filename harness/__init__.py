"""CHTD — Co-Harness Targeted Distillation.

Teacher and student LLMs share one harness and run the same task loop, so the teacher's agentic
capability is targetedly distilled into the student while distractors unrelated to that capability
are automatically stripped.

The modules here are the harness-side half of the pipeline; `distill/` builds the dataset and
`eval/` measures the result. See `docs/design.md` for the method and `docs/plan.md` for the plan.
"""

__all__: list[str] = []
