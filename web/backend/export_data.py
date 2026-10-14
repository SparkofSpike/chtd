"""Export read-only experiment JSONs into backend/data/*.json.

Run once (or when ledgers change). Every exported file carries a top-level
`_meta` with provenance. Numbers are taken verbatim from checked-in files;
nothing is fabricated. Missing sources degrade to {"status":"暂无数据"}.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app import config
from app.loaders import experiments as E


def _meta(sources: list[str], note: str = "") -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chtd_root": str(config.CHTD_ROOT),
        "sources": sources,
        "loader_version": "0.1.0",
        "status": "ok",
        "note": note,
    }


def main() -> None:
    out_dir = config.DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== export_data.py: each number's source file ===")
    exports: dict[str, dict] = {}

    # summary
    summary = E.summary_block()
    summary["_meta"] = _meta([
        "runs/holdout-4-20261014/dose-ladder-20261014.md (15-task failures, joint p)",
        "runs/holdout-4-20261014/pooled-analysis-20261014.md (57 pooled 31.6->17.5, McNemar p=0.077)",
    ], "Overview headline cards; see supports / does_not_support for evidence bounds.")
    exports["summary"] = summary
    print("summary.joint_p  <- dose-ladder-20261014.md (joint test table)")
    print("summary.pooled   <- pooled-analysis-20261014.md (failure-rate table)")

    # dose_ladder
    dose = {"rows": E.DOSE_LADDER["rows"], "joint_probability": E.DOSE_LADDER["joint_probability"],
            "task_set_size": E.DOSE_LADDER["task_set_size"]}
    # Recharts shapes (numeric; ranges take the upper bound, noted where applicable)
    dose["fifteen_bar"] = [
        {"arm": "base", "failures": 5, "note": "4-5/15 (upper bound; 6 byte-identical readings)"},
        {"arm": "chtd-student-state", "failures": 2},
        {"arm": "v1", "failures": 2},
        {"arm": "v2", "failures": 0},
        {"arm": "v2b", "failures": 0},
        {"arm": "v4", "failures": 1},
        {"arm": "v3", "failures": 1},
        {"arm": "v5", "failures": 1},
        {"arm": "v5b", "failures": 2},
    ]
    dose["efficiency"] = [
        {"arm": "chtd-student-state", "rows": 9, "failures": 2, "note": "1/4 data, one-cell margin"},
        {"arm": "v1", "rows": 33, "failures": 2},
    ]
    dose["dose_curve"] = [
        {"rows": 0, "ordinal_mean": 1.37, "note": "base; range 1.333-1.400 over 6 readings"},
        {"rows": 9, "ordinal_mean": 1.60},
        {"rows": 33, "ordinal_mean": 1.667},
        {"rows": 63, "ordinal_mean": 1.867},
        {"rows": 83, "ordinal_mean": 1.667},
        {"rows": 93, "ordinal_mean": 1.60},
    ]
    dose["_meta"] = _meta(["runs/holdout-4-20261014/dose-ladder-20261014.md",
                           "runs/holdout-4-20261014/dose-cells.json"],
                          "ladder table transcribed verbatim from the md; chart arrays derived.")
    exports["dose_ladder"] = dose
    print("dose_ladder      <- dose-ladder-20261014.md (ladder + joint probability tables)")

    # pooled
    pooled = E.POOLED
    pooled["pooled_bar"] = [
        {"family": "base", "fail_rate": 31.6, "of": "18/57"},
        {"family": "chtd", "fail_rate": 17.5, "of": "10/57"},
        {"family": "ctrl", "fail_rate": 24.6, "of": "14/57"},
        {"family": "v5", "fail_rate": 5.3, "of": "3/57"},
    ]
    pooled["_meta"] = _meta(["runs/holdout-4-20261014/pooled-analysis-20261014.md",
                             "eval/failure_rate.py", "eval/pooled_effect.py"])
    exports["pooled"] = pooled
    print("pooled           <- pooled-analysis-20261014.md (ordinal diff, failure rate, McNemar 12:4 p=0.077)")

    # failure_rate = combined 15-task + 57-pooled
    failure_rate = {
        "fifteen_task": [{"arm": r["arm"], "failures": r["failures"],
                          "training_rows": r["training_rows"]}
                         for r in E.DOSE_LADDER["rows"]],
        "pooled_failure_rate": E.POOLED["failure_rate"],
    }
    failure_rate["_meta"] = _meta(["dose-ladder-20261014.md", "pooled-analysis-20261014.md"])
    exports["failure_rate"] = failure_rate
    print("failure_rate     <- dose-ladder-20261014.md (15-task) + pooled-analysis (57)")

    # determinism
    det = E.DETERMINISM
    det["determinism"] = [
        {"arm": "base", "differing": 0, "note": "6 evaluations, byte-identical answers on 15 tasks"},
        {"arm": "adapter (v2 vs v2b)", "differing": 9, "note": "same weights, 9/15 answers differ, 0.33 ordinal points"},
    ]
    det["_meta"] = _meta(["runs/holdout-4-20261014/dose-ladder-20261014.md",
                          "eval/replication_floor.py", "eval/base_spread.py"])
    exports["determinism"] = det
    print("determinism      <- dose-ladder-20261014.md (base 6 byte-identical; adapter 9/15 differ 0.33)")

    # tool_health = parsed census
    census = E.load_tool_census()
    census["_meta"] = _meta([census.pop("_src", "tool-call-census.json")],
                            "first-tool / read ok_rate / ever_read / passed per arm.")
    exports["tool_health"] = census
    print("tool_health      <- runs/holdout-4-20261014/tool-call-census.json (whole-file json.load)")

    # scorer_power = constants + machine scores list
    sp = dict(E.SCORER_POWER)
    sp["machine_scores"] = E.load_machine_scores()
    sp["_meta"] = _meta(["runs/holdout-3-20261014/machine-scores/scores-*.json",
                         "eval/scorer_benchmark_holdout3.py", "eval/power_holdout3.py"],
                        "MVP: machine-scores parsed; handread/llm-judge deferred.")
    exports["scorer_power"] = sp
    print("scorer_power     <- machine-scores/scores-*.json + power numbers from holdout-3 records")

    # early_eval deferred -> 暂无数据
    exports["early_eval"] = {"status": "暂无数据",
                             "_meta": _meta(["runs/chtd-20261011-a/eval-4arms.json (deferred)"],
                                            "deferred for MVP.")}
    print("early_eval       <- 暂无数据 (deferred)")

    # Write
    for name, block in exports.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(block, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {path}")

    print(f"\nDone. {len(exports)} files in {out_dir}")


if __name__ == "__main__":
    main()
