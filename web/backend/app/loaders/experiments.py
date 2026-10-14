"""Parse the run ledgers into the exported experiment shapes.

MVP scope (per narrowed brief): dose-ladder + pooled md numbers, dose-cells.json,
tool-call-census.json, and scores-*.json. Other runs (handread/llm-judge/early
eval/teacher-state/h1h2) are deferred and represented as null with status
"暂无数据" rather than fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import PATHS


def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# --- Structured numbers transcribed from the checked-in markdown ledgers ----
# (confirmed against runs/holdout-4-20261014/dose-ladder-20261014.md and
#  pooled-analysis-20261014.md; provenance recorded per block).

DOSE_LADDER = {
    "task_set_size": 15,
    "rows": [
        {"arm": "base", "training_rows": 0, "ordinal_mean": "1.333-1.400",
         "cpf": "10/0/5 ...", "failures": "4-5/15", "gate": "13/15",
         "note": "six readings, byte-identical answers"},
        {"arm": "chtd-student-state", "training_rows": 9, "ordinal_mean": 1.600,
         "cpf": "11/2/2", "failures": "2/15", "gate": "14/15"},
        {"arm": "v1", "training_rows": 33, "ordinal_mean": 1.667,
         "cpf": "12/1/2", "failures": "2/15", "gate": "13/15"},
        {"arm": "v2", "training_rows": 63, "ordinal_mean": 1.867,
         "cpf": "13/2/0", "failures": "0/15", "gate": "14/15"},
        {"arm": "v2b", "training_rows": 63, "ordinal_mean": 1.533,
         "cpf": "8/7/0", "failures": "0/15", "gate": "15/15",
         "note": "v2 re-evaluated; 9/15 answers differ"},
        {"arm": "v4", "training_rows": 63, "ordinal_mean": 1.733,
         "cpf": "12/2/1", "failures": "1/15", "gate": "14/15", "note": "v2 data, seed1"},
        {"arm": "v3", "training_rows": 83, "ordinal_mean": 1.667,
         "cpf": "11/3/1", "failures": "1/15", "gate": "12/15"},
        {"arm": "v5", "training_rows": 93, "ordinal_mean": 1.600,
         "cpf": "10/4/1", "failures": "1/15", "gate": "14/15"},
        {"arm": "v5b", "training_rows": 93, "ordinal_mean": 1.467,
         "cpf": "9/4/2", "failures": "2/15", "gate": "13/15"},
    ],
    "joint_probability": [
        {"assumed_base_rate": "40%", "p": "4.8e-14"},
        {"assumed_base_rate": "33%", "p": "1.1e-10"},
        {"assumed_base_rate": "30%", "p": "3.4e-9"},
        {"assumed_base_rate": "27%", "p": "8.1e-8"},
        {"assumed_base_rate": "20%", "p": "2.6e-5"},
    ],
}

POOLED = {
    "tasks": 57,
    "ordinal_diff": {
        "chtd_minus_base": {"h1": 0.267, "h2": 0.083, "h3": 0.156, "pooled": 0.170,
                             "test": "t=1.29, p=0.20; sign p=0.60; bootstrap CI [-0.08,+0.43]"},
        "chtd_minus_ctrl": {"pooled": -0.050, "test": "t=-0.49, p=0.62"},
    },
    "failure_rate": {
        "base": {"h1": 33.3, "h2": 8.3, "h3": 40.0, "pooled": 31.6, "of": "18/57"},
        "chtd": {"h1": 13.3, "h2": 8.3, "h3": 23.3, "pooled": 17.5, "of": "10/57"},
        "ctrl": {"h1": 13.3, "h2": 25.0, "h3": 30.0, "pooled": 24.6, "of": "14/57"},
        "v5": {"h1": 13.3, "h2": 0.0, "h3": 3.3, "pooled": 5.3, "of": "3/57"},
    },
    "paired_flips": {"h1": "4:1", "h2": "1:1", "h3": "7:2", "pooled": "12:4",
                     "mcnemar_exact_p": 0.077},
}

DETERMINISM = {
    "base": {"evaluations": 6, "result": "byte-identical answers on all 15 tasks",
             "ordinal_range": "1.333-1.400"},
    "adapter": {"comparison": "v2 vs v2b (same weights)", "differing_answers": "9/15",
                "ordinal_gap": 0.33,
                "note": "single 15-task comparison of trained arms is uninformative"},
}

SCORER_POWER = {
    "gate_vs_author_pct": 75.7,
    "gate_vs_double_reader_pct": 75.0,
    "human_human_pct": 84.0,
    "judge_three_way_pct": 85.0,
    "judge_binary_pct": 89.0,
    "mde_30tasks_pp": 26.6,
    "tasks_for_10pp": "~227/arm",
    "confirmatory_14pp": "~110/arm",
    "ordinal_power_need": "~268",
}


def load_dose_cells(paths=PATHS) -> dict[str, Any]:
    cells = _load_json(paths.holdout4("dose-cells.json"))
    if cells is None:
        return {"status": "暂无数据", "arms": [], "n": 0, "_src": "dose-cells.json"}
    arms = {}
    for c in cells:
        arms.setdefault(c.get("arm"), []).append({
            "task": c.get("task"), "set": c.get("set"),
            "passed_gate": c.get("passed_gate"), "dose_rows": c.get("dose_rows"),
            "dose_note": c.get("dose_note"),
        })
    return {"n": len(cells), "arms": arms, "_src": "runs/holdout-4-20261014/dose-cells.json"}


def load_tool_census(paths=PATHS) -> dict[str, Any]:
    data = _load_json(paths.holdout4("tool-call-census.json"))
    if data is None:
        return {"status": "暂无数据", "arms": {}, "_src": "tool-call-census.json"}
    arms = {}
    for name, a in (data.get("arms") or {}).items():
        arms[name] = {
            "sessions": a.get("sessions"),
            "passed": a.get("passed"),
            "first_tool_counts": a.get("first_tool_counts"),
            "read_results": a.get("read_results"),
            "sessions_ever_read": a.get("sessions_ever_read"),
            "answers_directory_does_not_exist": a.get("answers_directory_does_not_exist"),
        }
    return {"instrument": data.get("instrument"), "arms": arms,
            "_src": "runs/holdout-4-20261014/tool-call-census.json"}


def load_machine_scores(paths=PATHS) -> list[dict[str, Any]]:
    out = []
    d = paths.root / "runs" / "holdout-3-20261014" / "machine-scores"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("scores-*.json")):
        j = _load_json(f)
        if j is None:
            continue
        out.append({"label": j.get("label"), "passed": j.get("passed"),
                    "total": j.get("total"), "n_results": len(j.get("results", [])),
                    "_src": f.name})
    return out


# --- Aggregators used by export_data.py ----------------------------------

def summary_block() -> dict[str, Any]:
    return {
        "diagnosis_tasks": {
            "n": 15, "base_failures": "4-5", "trained_failures": "0-2",
            "joint_p_max": "3.4e-9", "joint_p_conservative": "2.6e-5",
        },
        "pooled": {"n": 57, "base_failure_rate": "31.6%", "chtd_failure_rate": "17.5%",
                   "mcnemar_p": 0.077},
        "data_efficiency": {"student_state_rows": 9, "teacher_demo_rows": 33,
                            "note": "quarter of data, one-cell margin, suggestive"},
        "determinism": {"base_evals": 6, "adapter_differing": "9/15", "ordinal_gap": 0.33},
        "scorer": {"gate_human_pct": 75.0, "human_human_pct": 84.0, "mde_30_pp": 26.6},
        "supports": [
            "15-task endpoint: fine-tuning cuts failures 4-5 -> 0-2, joint p <= 3.4e-9",
            "57 pooled failure rate 31.6% -> 17.5% (McNemar p=0.077, strong hint not proof)",
            "Data efficiency: 9 student-state rows ~= 33 teacher-demo rows (one-cell margin)",
            "Measurement floor quantified: base deterministic, adapter +-0.33 ordinal points",
        ],
        "does_not_support": [
            "No claim that CHTD beats alternatives; p=0.077 not <0.05, ordinal not beyond noise",
            "9-rows-equals-33 is a single-cell margin on single 15-task evals, suggestive",
            "33-93 dose curve is flat at this n; do not claim 'more data is better'",
            "v5 is full teacher demos (10x rows), not CHTD shape evidence",
        ],
    }
