"""Build the error-recovery task set: one workspace per task, each with a real defect.

The target capability is error recovery — read a failure, work out why, fix it, confirm the fix.
A task set for it has to satisfy two properties that are easy to assume and expensive to get wrong:

* **The defect is real.** The check must fail before the fix. A task whose "broken" fixture
  actually passes measures nothing, and it looks exactly like a task the model solved.
* **The task is solvable.** The check must pass after the fix. A task nobody can complete produces
  no admitted dataset rows and no evidence either way.

So every entry carries its own ``fix``, and :func:`self_check` applies it to a copy and runs both
checks. ``fix`` is not part of the task: it never reaches the prompt, the workspace the model sees,
or the dataset. It exists so the set can be checked as a set, mechanically, before a rollout is
spent on it.

The prompt states the *symptom* and never the repair. That is the whole experiment: a task that
names its own fix asks the model to follow an instruction rather than to diagnose.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = ["TASKS", "build", "self_check", "write_task_set", "main"]


#: Each task: an id, the symptom a user would report, the files to write, the check that must fail
#: now and pass afterwards, and the fix that proves it can.
TASKS: tuple[Mapping[str, Any], ...] = (
    {
        "id": "fix-assertion",
        "prompt": (
            "The test suite in this directory is failing. Find out why and make it pass."
        ),
        "files": {
            "average.py": (
                '"""A small statistics helper."""\n'
                "\n"
                "\n"
                "def mean(values):\n"
                '    """The arithmetic mean of a non-empty sequence."""\n'
                "    if not values:\n"
                '        raise ValueError("mean of an empty sequence")\n'
                "    return sum(values) / len(values)\n"
            ),
            "test_average.py": (
                "from average import mean\n"
                "\n"
                "\n"
                "def test_mean_of_three():\n"
                "    assert mean([1, 2, 3]) == 3\n"
                "\n"
                "\n"
                "def test_mean_of_one():\n"
                "    assert mean([7]) == 7\n"
            ),
        },
        "check": "python -m pytest -q",
        "fix": {"test_average.py": ("from average import mean\n"
                                    "\n"
                                    "\n"
                                    "def test_mean_of_three():\n"
                                    "    assert mean([1, 2, 3]) == 2\n"
                                    "\n"
                                    "\n"
                                    "def test_mean_of_one():\n"
                                    "    assert mean([7]) == 7\n")},
    },
    {
        "id": "fix-import-name",
        "prompt": (
            "Running the check script in this directory fails. Work out why and make it run "
            "correctly."
        ),
        "files": {
            "textutils.py": (
                '"""Text helpers."""\n'
                "\n"
                "\n"
                'def normalise(value):\n'
                '    """Lower-case and strip surrounding whitespace."""\n'
                "    return value.strip().lower()\n"
            ),
            "check.py": (
                "from textutils import normalize\n"
                "\n"
                "print(normalize('  Hello World  '))\n"
            ),
        },
        "check": "python check.py",
        "fix": {"check.py": ("from textutils import normalise\n"
                             "\n"
                             "print(normalise('  Hello World  '))\n")},
    },
    {
        "id": "fix-off-by-one",
        "prompt": (
            "The script here produces the wrong answer for the last item. Find the mistake and "
            "correct it."
        ),
        "files": {
            "sums.py": (
                '"""Running totals."""\n'
                "\n"
                "\n"
                "def running_totals(values):\n"
                '    """The total after each value, ending with the sum of all of them."""\n'
                "    totals = []\n"
                "    total = 0\n"
                "    for index in range(len(values) - 1):\n"
                "        total += values[index]\n"
                "        totals.append(total)\n"
                "    return totals\n"
            ),
            "check.py": (
                "from sums import running_totals\n"
                "\n"
                "got = running_totals([1, 2, 3, 4])\n"
                "print('got     ', got)\n"
                "print('expected', [1, 3, 6, 10])\n"
                "assert got == [1, 3, 6, 10], got\n"
            ),
        },
        "check": "python check.py",
        "fix": {"sums.py": ('"""Running totals."""\n'
                            "\n"
                            "\n"
                            "def running_totals(values):\n"
                            '    """The total after each value, ending with the sum of all of them."""\n'
                            "    totals = []\n"
                            "    total = 0\n"
                            "    for index in range(len(values)):\n"
                            "        total += values[index]\n"
                            "        totals.append(total)\n"
                            "    return totals\n")},
    },
    {
        "id": "fix-config-key",
        "prompt": (
            "The program in this directory stops with a KeyError. Find out why and make it work."
        ),
        "files": {
            "settings.json": json.dumps(
                {"service": {"name": "reporter", "port": 8080, "timeout_seconds": 30}},
                indent=2,
            )
            + "\n",
            "app.py": (
                "import json\n"
                "\n"
                "with open('settings.json', encoding='utf-8') as handle:\n"
                "    settings = json.load(handle)\n"
                "\n"
                "print('starting', settings['service']['name'])\n"
                "print('port    ', settings['service']['port'])\n"
                "print('timeout ', settings['service']['timeout'])\n"
            ),
        },
        "check": "python app.py",
        "fix": {"app.py": ("import json\n"
                           "\n"
                           "with open('settings.json', encoding='utf-8') as handle:\n"
                           "    settings = json.load(handle)\n"
                           "\n"
                           "print('starting', settings['service']['name'])\n"
                           "print('port    ', settings['service']['port'])\n"
                           "print('timeout ', settings['service']['timeout_seconds'])\n")},
    },
    {
        "id": "fix-cli-flag",
        "prompt": (
            "The documented command in the README of this directory fails. Make the command the "
            "README documents work."
        ),
        "files": {
            "report.py": (
                '"""Print a report."""\n'
                "\n"
                "import argparse\n"
                "\n"
                "\n"
                "def main():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument('--title', required=True)\n"
                "    parser.add_argument('--row', type=int, default=10)\n"
                "    args = parser.parse_args()\n"
                "    print(f'{args.title}: {args.row} rows')\n"
                "\n"
                "\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
            "README.md": (
                "# report\n"
                "\n"
                "Print a report:\n"
                "\n"
                "```sh\n"
                "python report.py --title summary --rows 5\n"
                "```\n"
            ),
        },
        "check": "python report.py --title summary --rows 5",
        "fix": {"report.py": (
            '"""Print a report."""\n'
            "\n"
            "import argparse\n"
            "\n"
            "\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--title', required=True)\n"
            "    parser.add_argument('--rows', type=int, default=10)\n"
            "    args = parser.parse_args()\n"
            "    print(f'{args.title}: {args.rows} rows')\n"
            "\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )},
    },
    {
        "id": "fix-type-error",
        "prompt": (
            "Calling the function in this directory with a value from the config fails. Find the "
            "cause and make the call work."
        ),
        "files": {
            "retry.py": (
                '"""Retry with a delay."""\n'
                "\n"
                "import math\n"
                "\n"
                "\n"
                "def backoff(attempt, base):\n"
                '    """Seconds to wait before the given attempt, doubling each time."""\n'
                "    return base * math.pow(2, attempt)\n"
            ),
            "config.json": json.dumps({"base_delay": "0.5"}, indent=2) + "\n",
            "run.py": (
                "import json\n"
                "\n"
                "from retry import backoff\n"
                "\n"
                "with open('config.json', encoding='utf-8') as handle:\n"
                "    config = json.load(handle)\n"
                "\n"
                "print('wait', backoff(3, config['base_delay']), 'seconds')\n"
            ),
        },
        "check": "python run.py",
        "fix": {"config.json": json.dumps({"base_delay": 0.5}, indent=2) + "\n"},
    },
    {
        "id": "fix-path",
        "prompt": (
            "The script in this directory cannot find its input file. Make it read the data that "
            "is actually present."
        ),
        "files": {
            "data": {"readings.csv": "sensor,value\nalpha,3\nbravo,5\ncharlie,7\n"},
            "load.py": (
                "with open('data/readings.txt', encoding='utf-8') as handle:\n"
                "    lines = handle.read().strip().splitlines()\n"
                "\n"
                "print('rows:', len(lines) - 1)\n"
            ),
        },
        "check": "python load.py",
        "fix": {"load.py": ("with open('data/readings.csv', encoding='utf-8') as handle:\n"
                            "    lines = handle.read().strip().splitlines()\n"
                            "\n"
                            "print('rows:', len(lines) - 1)\n")},
    },
    {
        "id": "fix-regex",
        "prompt": (
            "The parser here reports zero matches for input that clearly contains what it looks "
            "for. Find out why and fix it."
        ),
        "files": {
            "parse.py": (
                "import re\n"
                "\n"
                "LOG = '''\n"
                "2026-10-09 08:12:01 INFO  started\n"
                "2026-10-09 08:12:07 WARN  slow response\n"
                "2026-10-09 08:12:19 ERROR could not reach host\n"
                "'''\n"
                "\n"
                "# A level, then the rest of the line.\n"
                "PATTERN = re.compile(r'^(INFO+)$')\n"
                "\n"
                "levels = [m.group(1) for m in PATTERN.finditer(LOG.strip())]\n"
                "print('found:', levels)\n"
                "print('count:', len(levels))\n"
                "assert len(levels) == 3, levels\n"
            ),
        },
        "check": "python parse.py",
        "fix": {"parse.py": ("import re\n"
                             "\n"
                             "LOG = '''\n"
                             "2026-10-09 08:12:01 INFO  started\n"
                             "2026-10-09 08:12:07 WARN  slow response\n"
                             "2026-10-09 08:12:19 ERROR could not reach host\n"
                             "'''\n"
                             "\n"
                             "# A level, then the rest of the line.\n"
                             "PATTERN = re.compile(r'\\b(INFO|WARN|ERROR)\\b')\n"
                             "\n"
                             "levels = [m.group(1) for m in PATTERN.finditer(LOG.strip())]\n"
                             "print('found:', levels)\n"
                             "print('count:', len(levels))\n"
                             "assert len(levels) == 3, levels\n")},
    },
    {
        "id": "fix-json-trailing-comma",
        "prompt": (
            "The program in this directory refuses to start because of the file it loads. Fix "
            "that file so the program runs."
        ),
        "files": {
            "hosts.json": (
                "{\n"
                '  "hosts": [\n'
                '    {"name": "alpha", "port": 22},\n'
                '    {"name": "bravo", "port": 22},\n'
                "  ]\n"
                "}\n"
            ),
            "connect.py": (
                "import json\n"
                "\n"
                "with open('hosts.json', encoding='utf-8') as handle:\n"
                "    config = json.load(handle)\n"
                "\n"
                "for host in config['hosts']:\n"
                "    print(f\"{host['name']}:{host['port']}\")\n"
            ),
        },
        "check": "python connect.py",
        "fix": {"hosts.json": (
            "{\n"
            '  "hosts": [\n'
            '    {"name": "alpha", "port": 22},\n'
            '    {"name": "bravo", "port": 22}\n'
            "  ]\n"
            "}\n"
        )},
    },
    {
        "id": "fix-iteration-mutation",
        "prompt": (
            "The cleanup script in this directory stops with an error instead of removing the "
            "entries it is meant to remove. Find out why and make it finish its job."
        ),
        "files": {
            "cleanup.py": (
                '"""Drop short words from a word count."""\n'
                "\n"
                "COUNTS = {'alpha': 4, 'be': 1, 'charlie': 3, 'it': 2, 'delta': 5}\n"
                "\n"
                "\n"
                "def drop_shorter_than(counts, minimum):\n"
                '    """Remove every word shorter than ``minimum``, in place."""\n'
                "    for word in counts:\n"
                "        if len(word) < minimum:\n"
                "            del counts[word]\n"
                "    return counts\n"
                "\n"
                "\n"
                "print(drop_shorter_than(dict(COUNTS), 3))\n"
            ),
            "check.py": (
                "from cleanup import drop_shorter_than\n"
                "\n"
                "result = drop_shorter_than({'alpha': 4, 'be': 1, 'charlie': 3}, 3)\n"
                "print('kept:', sorted(result))\n"
                "assert result == {'alpha': 4, 'charlie': 3}, result\n"
            ),
        },
        "check": "python cleanup.py",
        "fix": {"cleanup.py": (
            '"""Drop short words from a word count."""\n'
            "\n"
            "COUNTS = {'alpha': 4, 'be': 1, 'charlie': 3, 'it': 2, 'delta': 5}\n"
            "\n"
            "\n"
            "def drop_shorter_than(counts, minimum):\n"
            '    """Remove every word shorter than ``minimum``, in place."""\n'
            "    for word in list(counts):\n"
            "        if len(word) < minimum:\n"
            "            del counts[word]\n"
            "    return counts\n"
            "\n"
            "\n"
            "print(drop_shorter_than(dict(COUNTS), 3))\n"
        )},
    },
    {
        "id": "fix-wrong-operator",
        "prompt": (
            "One of the functions in this directory returns the wrong answer for some inputs. Find "
            "it and correct it."
        ),
        "files": {
            "ranges.py": (
                '"""Interval helpers."""\n'
                "\n"
                "\n"
                "def overlaps(first, second):\n"
                '    """Whether two closed intervals [start, end] share a point."""\n'
                "    first_start, first_end = first\n"
                "    second_start, second_end = second\n"
                "    return first_start < second_end and second_start < first_end\n"
            ),
            "test_ranges.py": (
                "from ranges import overlaps\n"
                "\n"
                "\n"
                "def test_touching_at_a_point():\n"
                "    assert overlaps((1, 3), (3, 5)) is True\n"
                "\n"
                "\n"
                "def test_apart():\n"
                "    assert overlaps((1, 3), (4, 6)) is False\n"
                "\n"
                "\n"
                "def test_nested():\n"
                "    assert overlaps((1, 10), (4, 6)) is True\n"
            ),
        },
        "check": "python -m pytest -q",
        "fix": {"ranges.py": (
            '"""Interval helpers."""\n'
            "\n"
            "\n"
            "def overlaps(first, second):\n"
            '    """Whether two closed intervals [start, end] share a point."""\n'
            "    first_start, first_end = first\n"
            "    second_start, second_end = second\n"
            "    return first_start <= second_end and second_start <= first_end\n"
        )},
    },
    {
        "id": "fix-silent-truncation",
        "prompt": (
            "The pipeline here loses the last record of its input without reporting anything. Find "
            "out why and make it keep every record."
        ),
        "files": {
            "pipeline.py": (
                '"""Chunk a stream of records."""\n'
                "\n"
                "\n"
                "def chunk(records, size):\n"
                '    """Split records into consecutive chunks of at most ``size``."""\n'
                "    chunks = []\n"
                "    for start in range(0, len(records), size):\n"
                "        chunks.append(records[start:start + size - 1])\n"
                "    return chunks\n"
            ),
            "check.py": (
                "from pipeline import chunk\n"
                "\n"
                "records = list(range(7))\n"
                "chunks = chunk(records, 3)\n"
                "flat = [item for block in chunks for item in block]\n"
                "print('in :', len(records))\n"
                "print('out:', len(flat))\n"
                "assert flat == records, (flat, records)\n"
            ),
        },
        "check": "python check.py",
        "fix": {"pipeline.py": (
            '"""Chunk a stream of records."""\n'
            "\n"
            "\n"
            "def chunk(records, size):\n"
            '    """Split records into consecutive chunks of at most ``size``."""\n'
            "    chunks = []\n"
            "    for start in range(0, len(records), size):\n"
            "        chunks.append(records[start:start + size])\n"
            "    return chunks\n"
        )},
    },
)


def _write_files(root: Path, files: Mapping[str, Any]) -> None:
    """Write a fixture tree. A mapping value is a subdirectory; a string is a file."""
    for name, content in files.items():
        target = root / name
        if isinstance(content, Mapping):
            target.mkdir(parents=True, exist_ok=True)
            _write_files(target, content)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")


def _run_check(root: Path, check: str) -> tuple[int, str]:
    completed = subprocess.run(
        check,
        cwd=root,
        shell=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def build(destination: str | Path) -> list[dict[str, str]]:
    """Write one workspace per task, and return the task rows.

    The defect is written and nothing else: ``fix`` stays here. A workspace that shipped its own
    repair would answer the question the task exists to ask.

    ``check`` *does* travel with the row, because a round has to observe its outcome: the command
    decides whether a run succeeded, and the signal gate validates divergences against it. It names
    no repair — ``python -m pytest -q`` says nothing a diagnosis would not have to find anyway.
    """
    root = Path(destination)
    rows: list[dict[str, str]] = []
    for task in TASKS:
        workspace = root / task["id"]
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        _write_files(workspace, task["files"])
        rows.append(
            {
                "id": task["id"],
                "prompt": task["prompt"],
                "cwd": str(workspace),
                "check": task["check"],
            }
        )
    return rows


def self_check(verbose: bool = True) -> int:
    """Every defect must break its check, and every fix must repair it.

    Run before spending a rollout on the set. A task whose "broken" fixture passes measures
    nothing and is indistinguishable afterwards from a task that was solved.
    """
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        for task in TASKS:
            broken = scratch_path / f"{task['id']}-broken"
            broken.mkdir()
            _write_files(broken, task["files"])
            code, output = _run_check(broken, task["check"])
            if code == 0:
                failures.append(f"{task['id']}: the defect does not fail its check")
                if verbose:
                    print(f"  {task['id']:26} BROKEN-PASSES  {output.splitlines()[:1]}")
                continue

            repaired = scratch_path / f"{task['id']}-fixed"
            repaired.mkdir()
            _write_files(repaired, task["files"])
            _write_files(repaired, task["fix"])
            code, output = _run_check(repaired, task["check"])
            if code != 0:
                failures.append(f"{task['id']}: the fix does not make the check pass")
                if verbose:
                    print(f"  {task['id']:26} FIX-FAILS      {output.splitlines()[:2]}")
                continue
            if verbose:
                print(f"  {task['id']:26} ok  (fails, then passes after the fix)")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(f"\n{len(TASKS)} tasks: every defect fails its check, every fix repairs it.")
    return 0


def write_task_set(rows: Sequence[Mapping[str, str]], path: str | Path) -> Path:
    """Write the task set.

    ``cwd`` is deliberately not written. The workspaces are generated by :func:`build` into a
    gitignored directory, so a recorded absolute path would be a path on one machine that no reader
    can use and that a rebuild would silently leave stale. The reader resolves each task's workspace
    from its id instead, which is the same thing a rebuild produces.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by harness/tasks/build_workspaces.py - do not edit by hand.\n")
        handle.write("# Each task's workspace holds one real defect; the prompt states the\n")
        handle.write("# symptom and never the repair. `check` decides whether a run succeeded.\n")
        handle.write("# Resolve the workspace by id: python harness/tasks/build_workspaces.py\n")
        for row in rows:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "cwd"},
                                    ensure_ascii=False) + "\n")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the error-recovery task set.")
    parser.add_argument(
        "--workspaces",
        default="data/tasks",
        help="where the per-task workspaces are written",
    )
    parser.add_argument(
        "--out",
        default="harness/tasks/error-recovery.jsonl",
        help="the task set harness.rollout.load_tasks reads",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the self-check and write nothing",
    )
    args = parser.parse_args(argv)

    if args.check_only:
        return self_check()

    code = self_check()
    if code != 0:
        print("\nrefusing to write a task set whose defects or fixes do not hold", file=sys.stderr)
        return code

    rows = build(args.workspaces)
    path = write_task_set(rows, args.out)
    print(f"\nwrote {len(rows)} workspaces under {args.workspaces}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
