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

__all__ = [
    "EXPECT",
    "FINAL_HOLDOUT",
    "FORBID",
    "TASKS",
    "build",
    "main",
    "self_check",
    "split",
    "write_task_set",
]


#: Each task: an id, the symptom a user would report, the files to write, the check that must fail
#: now and pass afterwards, and the fix that proves it can.
def _bug(task_id: str, module_name: str, broken: str, fixed: str, check: str) -> dict[str, Any]:
    """A one-module defect and its repair, with a check that fails now and passes after.

    Most of the set has this shape: one file holds the defect, a second exercises it and asserts.
    Spelling the pair out as a dict each time was boilerplate that invited transcription mistakes —
    putting the broken and fixed versions next to each other in one call keeps them from drifting
    apart. `--check-only` still proves both halves on every build.
    """
    return {
        "id": task_id,
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {module_name: broken, "check.py": check},
        "check": "python check.py",
        "fix": {module_name: fixed},
    }


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
    # --- held out from the round-f fine-tune ------------------------------------------------
    # These four defects are the same kind as the twelve above and were written after the dataset
    # was built, so a model that answers them has not seen them. A behaviour change measured on the
    # training tasks cannot tell learning from memorising; one measured here can.
    {
        "id": "fix-argument-order",
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {
            "greet.py": (
                '"""Greeting helpers."""\n'
                "\n"
                "\n"
                "def greet(name, greeting):\n"
                '    """Return a greeting addressed to a name."""\n'
                '    return f"{greeting}, {name}!"\n'
            ),
            "check.py": (
                "from greet import greet\n"
                "\n"
                "print(greet('Hello', 'Ada'))\n"
                "assert greet('Hello', 'Ada') == 'Hello, Ada!', greet('Hello', 'Ada')\n"
            ),
        },
        "check": "python check.py",
        "fix": {"check.py": (
            "from greet import greet\n"
            "\n"
            "print(greet('Ada', 'Hello'))\n"
            "assert greet('Ada', 'Hello') == 'Hello, Ada!', greet('Ada', 'Hello')\n"
        )},
    },
    {
        "id": "fix-empty-input",
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {
            "words.py": (
                '"""Word helpers."""\n'
                "\n"
                "\n"
                "def first_word(text):\n"
                '    """The first word of a sentence, or an empty string when there is none."""\n'
                "    return text.split()[0]\n"
            ),
            "check.py": (
                "from words import first_word\n"
                "\n"
                "print(repr(first_word('hello there')))\n"
                "print(repr(first_word('')))\n"
            ),
        },
        "check": "python check.py",
        "fix": {"words.py": (
            '"""Word helpers."""\n'
            "\n"
            "\n"
            "def first_word(text):\n"
            '    """The first word of a sentence, or an empty string when there is none."""\n'
            "    parts = text.split()\n"
            "    return parts[0] if parts else ''\n"
        )},
    },
    {
        "id": "fix-missing-prefix",
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {
            "report.py": (
                "TOTAL = 5\n"
                "NAME = 'widgets'\n"
                "\n"
                'print("Item: {NAME} ({TOTAL} in stock)")\n'
            ),
            "check.py": (
                "import subprocess, sys\n"
                "\n"
                "out = subprocess.run([sys.executable, 'report.py'], capture_output=True, text=True)\n"
                "print('got     ', out.stdout.strip())\n"
                "print('expected', 'Item: widgets (5 in stock)')\n"
                "assert out.stdout.strip() == 'Item: widgets (5 in stock)', out.stdout\n"
            ),
        },
        "check": "python check.py",
        "fix": {"report.py": (
            "TOTAL = 5\n"
            "NAME = 'widgets'\n"
            "\n"
            'print(f"Item: {NAME} ({TOTAL} in stock)")\n'
        )},
    },
    {
        "id": "fix-slice-direction",
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {
            "tail.py": (
                '"""List helpers."""\n'
                "\n"
                "\n"
                "def last_three(items):\n"
                '    """The final three items, or all of them when there are fewer."""\n'
                "    return items[:-3]\n"
            ),
            "check.py": (
                "from tail import last_three\n"
                "\n"
                "got = last_three([1, 2, 3, 4, 5])\n"
                "print('got     ', got)\n"
                "print('expected', [3, 4, 5])\n"
                "assert got == [3, 4, 5], got\n"
            ),
        },
        "check": "python check.py",
        "fix": {"tail.py": (
            '"""List helpers."""\n'
            "\n"
            "\n"
            "def last_three(items):\n"
            '    """The final three items, or all of them when there are fewer."""\n'
            "    return items[-3:]\n"
        )},
    },
    # --- the wider set -------------------------------------------------------------------------
    # Twenty-four more defects, written to the same rule: a real break, a real repair, and a check
    # that proves both. `FINAL_HOLDOUT` names the ones no run will train on.
    _bug(
        "fix-default-mutable", "collect.py",
        'def collect(item, into=[]):\n'
        '    """Append an item and return the collection."""\n'
        "    into.append(item)\n"
        "    return into\n",
        'def collect(item, into=None):\n'
        '    """Append an item and return the collection."""\n'
        "    if into is None:\n"
        "        into = []\n"
        "    into.append(item)\n"
        "    return into\n",
        "from collect import collect\n"
        "\n"
        "first = collect(1)\n"
        "second = collect(2)\n"
        "print('first ', first)\n"
        "print('second', second)\n"
        "assert first == [1], first\n"
        "assert second == [2], second\n",
    ),
    _bug(
        "fix-integer-division", "average.py",
        'def average(values):\n'
        '    """The arithmetic mean."""\n'
        "    return sum(values) // len(values)\n",
        'def average(values):\n'
        '    """The arithmetic mean."""\n'
        "    return sum(values) / len(values)\n",
        "from average import average\n"
        "\n"
        "got = average([1, 2])\n"
        "print('got     ', got)\n"
        "print('expected', 1.5)\n"
        "assert got == 1.5, got\n",
    ),
    _bug(
        "fix-shadowed-total", "prices.py",
        'def total(prices):\n'
        '    """The sum of the prices."""\n'
        "    total = 0\n"
        "    for price in prices:\n"
        "        total = price\n"
        "    return total\n",
        'def total(prices):\n'
        '    """The sum of the prices."""\n'
        "    total = 0\n"
        "    for price in prices:\n"
        "        total += price\n"
        "    return total\n",
        "from prices import total\n"
        "\n"
        "got = total([3, 4, 5])\n"
        "print('got     ', got)\n"
        "print('expected', 12)\n"
        "assert got == 12, got\n",
    ),
    _bug(
        "fix-string-case", "answers.py",
        'def is_yes(answer):\n'
        '    """Whether the answer means yes, however it was typed."""\n'
        "    return answer == 'yes'\n",
        'def is_yes(answer):\n'
        '    """Whether the answer means yes, however it was typed."""\n'
        "    return answer.strip().lower() == 'yes'\n",
        "from answers import is_yes\n"
        "\n"
        "for value in ('yes', 'Yes', 'YES', ' yes '):\n"
        "    print(repr(value), is_yes(value))\n"
        "    assert is_yes(value) is True, value\n"
        "print('no ->', is_yes('no'))\n"
        "assert is_yes('no') is False\n",
    ),
    _bug(
        "fix-missing-dict-default", "wordcount.py",
        'def count_words(words):\n'
        '    """How often each word appears."""\n'
        "    counts = {}\n"
        "    for word in words:\n"
        "        counts[word] = counts[word] + 1\n"
        "    return counts\n",
        'def count_words(words):\n'
        '    """How often each word appears."""\n'
        "    counts = {}\n"
        "    for word in words:\n"
        "        counts[word] = counts.get(word, 0) + 1\n"
        "    return counts\n",
        "from wordcount import count_words\n"
        "\n"
        "got = count_words(['a', 'b', 'a'])\n"
        "print('got     ', got)\n"
        "print('expected', {'a': 2, 'b': 1})\n"
        "assert got == {'a': 2, 'b': 1}, got\n",
    ),
    _bug(
        "fix-sort-key", "sorting.py",
        'def by_length(words):\n'
        '    """The words shortest first."""\n'
        "    return sorted(words, key=len(words))\n",
        'def by_length(words):\n'
        '    """The words shortest first."""\n'
        "    return sorted(words, key=len)\n",
        "from sorting import by_length\n"
        "\n"
        "got = by_length(['ccc', 'a', 'bb'])\n"
        "print('got     ', got)\n"
        "print('expected', ['a', 'bb', 'ccc'])\n"
        "assert got == ['a', 'bb', 'ccc'], got\n",
    ),
    _bug(
        "fix-parity-test", "parity.py",
        'def is_even(number):\n'
        '    """Whether a whole number is even."""\n'
        "    return number % 2 == 1\n",
        'def is_even(number):\n'
        '    """Whether a whole number is even."""\n'
        "    return number % 2 == 0\n",
        "from parity import is_even\n"
        "\n"
        "for value, expected in ((4, True), (7, False)):\n"
        "    print(value, is_even(value), 'expected', expected)\n"
        "    assert is_even(value) is expected, value\n",
    ),
    _bug(
        "fix-loop-continue", "search.py",
        'def first_negative(values):\n'
        '    """The first negative value, or None when there is none."""\n'
        "    for value in values:\n"
        "        if value < 0:\n"
        "            continue\n"
        "    return None\n",
        'def first_negative(values):\n'
        '    """The first negative value, or None when there is none."""\n'
        "    for value in values:\n"
        "        if value < 0:\n"
        "            return value\n"
        "    return None\n",
        "from search import first_negative\n"
        "\n"
        "got = first_negative([3, -4, 5])\n"
        "print('got     ', got)\n"
        "print('expected', -4)\n"
        "assert got == -4, got\n"
        "print('none    ', first_negative([1, 2]))\n"
        "assert first_negative([1, 2]) is None\n",
    ),
    _bug(
        "fix-missing-return", "doubling.py",
        'def double(number):\n'
        '    """Twice the number."""\n'
        "    number * 2\n",
        'def double(number):\n'
        '    """Twice the number."""\n'
        "    return number * 2\n",
        "from doubling import double\n"
        "\n"
        "got = double(21)\n"
        "print('got     ', got)\n"
        "print('expected', 42)\n"
        "assert got == 42, got\n",
    ),
    _bug(
        "fix-min-max-clamp", "clamp.py",
        'def clamp(value, low, high):\n'
        '    """Keep a value inside a range."""\n'
        "    return min(low, max(high, value))\n",
        'def clamp(value, low, high):\n'
        '    """Keep a value inside a range."""\n'
        "    return max(low, min(high, value))\n",
        "from clamp import clamp\n"
        "\n"
        "for value, expected in ((-5, 0), (5, 5), (50, 10)):\n"
        "    got = clamp(value, 0, 10)\n"
        "    print(value, '->', got, 'expected', expected)\n"
        "    assert got == expected, value\n",
    ),
    _bug(
        "fix-shallow-copy", "table.py",
        'def blanked(table, column):\n'
        '    """A copy of the table with a column filled with zeroes."""\n'
        "    for row in table:\n"
        "        row[column] = 0\n"
        "    return table\n",
        'def blanked(table, column):\n'
        '    """A copy of the table with a column filled with zeroes."""\n'
        "    result = [dict(row) for row in table]\n"
        "    for row in result:\n"
        "        row[column] = 0\n"
        "    return result\n",
        "from table import blanked\n"
        "\n"
        "original = [{'n': 1}, {'n': 2}]\n"
        "result = blanked(original, 'n')\n"
        "print('result  ', result)\n"
        "print('original', original)\n"
        "assert original == [{'n': 1}, {'n': 2}], original\n"
        "assert result == [{'n': 0}, {'n': 0}], result\n",
    ),
    _bug(
        "fix-ordinal-index", "ordinals.py",
        'def nth(items, position):\n'
        '    """The item at a 1-based position."""\n'
        "    return items[position]\n",
        'def nth(items, position):\n'
        '    """The item at a 1-based position."""\n'
        "    return items[position - 1]\n",
        "from ordinals import nth\n"
        "\n"
        "got = nth(['a', 'b', 'c'], 1)\n"
        "print('got     ', got)\n"
        "print('expected', 'a')\n"
        "assert got == 'a', got\n"
        "assert nth(['a', 'b', 'c'], 3) == 'c'\n",
    ),
    {
        "id": "fix-single-name",
        "prompt": "unused - every task shares BUG_PROMPT",
        "files": {
            "names.py": (
                'def split_name(full):\n'
                '    """The first and last name, from a full name."""\n'
                "    parts = full.split()\n"
                "    return parts[0], parts[1]\n"
            ),
            "check.py": (
                "from names import split_name\n"
                "\n"
                "print('two names:', split_name('Ada Lovelace'))\n"
                "print('one name :', split_name('Ada'))\n"
            ),
        },
        "check": "python check.py",
        "fix": {"names.py": (
            'def split_name(full):\n'
            '    """The first and last name, from a full name."""\n'
            "    parts = full.split()\n"
            "    if len(parts) < 2:\n"
            "        return parts[0] if parts else '', ''\n"
            "    return parts[0], parts[1]\n"
        )},
    },
)

#: The second tranche, written 2026-10-11 because the first twenty-nine could not resolve a
#: difference: with five held-out tasks, one task is twenty points of the score, and the base model's
#: own run-to-run range measured larger than any arm gap. A set that cannot see an effect is a set
#: that cannot support a claim, so the defect kinds are broadened here beyond indexing and arithmetic
#: — scope, iteration, error handling, and text handling each get their own fixtures.
#:
#: Every one still obeys the same rule as the rest: the defect must fail its own check, and the fix
#: must make that check pass. `self_check` refuses to write a task set where that is not true, so a
#: task below that looks right and is not cannot reach a round.
TASKS = TASKS + (
    _bug(
        # Calls `.strip` instead of `.strip()`, so the function returns a bound method. The class of
        # defect where the code names the right operation and never invokes it.
        "fix-strip-whitespace", "clean.py",
        'def tidy(text):\n'
        '    """The text without surrounding whitespace."""\n'
        '    return text.strip\n',
        'def tidy(text):\n'
        '    """The text without surrounding whitespace."""\n'
        '    return text.strip()\n',
        'from clean import tidy\n'
        '\n'
        'assert tidy("  hi  ") == "hi", repr(tidy("  hi  "))\n'
        'print("ok")\n',
    ),
    _bug(
        # Two computed totals compared with `==`. Correct for integers, wrong for the floats a
        # running total actually is.
        "fix-float-equal", "totals.py",
        'def same(a, b):\n'
        '    """Whether two computed totals agree."""\n'
        '    return a == b\n',
        'def same(a, b):\n'
        '    """Whether two computed totals agree."""\n'
        '    return abs(a - b) < 1e-9\n',
        'from totals import same\n'
        '\n'
        'assert same(0.1 + 0.2, 0.3), (0.1 + 0.2, 0.3)\n'
        'print("ok")\n',
    ),
    _bug(
        # A list of closures over a loop variable: every one of them reads the value the loop
        # variable holds *after* the loop, so all of them behave alike.
        "fix-closure-late-binding", "handlers.py",
        'def build(prefixes):\n'
        '    """One function per prefix, each prepending that prefix."""\n'
        '    return [lambda text: prefix + text for prefix in prefixes]\n',
        'def build(prefixes):\n'
        '    """One function per prefix, each prepending that prefix."""\n'
        '    return [lambda text, prefix=prefix: prefix + text for prefix in prefixes]\n',
        'from handlers import build\n'
        '\n'
        'first, second = build(["[a]", "[b]"])\n'
        'assert first("x") == "[a]x", first("x")\n'
        'assert second("x") == "[b]x", second("x")\n'
        'print("ok")\n',
    ),
    _bug(
        # The same generator consumed twice: counting it exhausts it, so the total that follows is
        # zero. A defect that makes a function quietly return a plausible wrong number.
        "fix-generator-exhaust", "streams.py",
        'def doubled(values):\n'
        '    """A generator of each value, doubled."""\n'
        '    return (value * 2 for value in values)\n'
        '\n'
        '\n'
        'def summarise(values):\n'
        '    """How many values there are, and their total."""\n'
        '    stream = doubled(values)\n'
        '    return sum(1 for _ in stream), sum(stream)\n',
        'def doubled(values):\n'
        '    """A generator of each value, doubled."""\n'
        '    return (value * 2 for value in values)\n'
        '\n'
        '\n'
        'def summarise(values):\n'
        '    """How many values there are, and their total."""\n'
        '    doubled_values = list(doubled(values))\n'
        '    return len(doubled_values), sum(doubled_values)\n',
        'from streams import summarise\n'
        '\n'
        'count, total = summarise([1, 2, 3])\n'
        'assert (count, total) == (3, 12), (count, total)\n'
        'print("ok")\n',
    ),
    _bug(
        # A bare `except` that turns every failure into a missing value, including the ones that
        # are bugs in the caller rather than an absent item.
        "fix-broad-except", "reader.py",
        'def load(values, index):\n'
        '    """The value at an index, or None when it is not there."""\n'
        '    try:\n'
        '        return values[index]\n'
        '    except:\n'
        '        return None\n',
        'def load(values, index):\n'
        '    """The value at an index, or None when it is not there."""\n'
        '    try:\n'
        '        return values[index]\n'
        '    except IndexError:\n'
        '        return None\n',
        'from reader import load\n'
        '\n'
        'assert load(["a", "b"], 1) == "b"\n'
        'assert load(["a", "b"], 9) is None\n'
        'try:\n'
        '    load(["a", "b"], "not an index")\n'
        'except TypeError:\n'
        '    print("ok: a real error still surfaces")\n'
        'else:\n'
        '    raise AssertionError("a bare except swallowed a TypeError")\n',
    ),
    _bug(
        # Text joined to a number without converting it, so the function raises rather than
        # returning the label it was written to return.
        "fix-string-concat-int", "report.py",
        'def label(count):\n'
        '    """A line saying how many items were seen."""\n'
        '    return "seen " + count + " items"\n',
        'def label(count):\n'
        '    """A line saying how many items were seen."""\n'
        '    return "seen " + str(count) + " items"\n',
        'from report import label\n'
        '\n'
        'assert label(3) == "seen 3 items", label(3)\n'
        'print("ok")\n',
    ),
    _bug(
        # Two lookups that assume both levels exist, where the docstring promises a missing one is
        # reported as an absent setting.
        "fix-dict-get-chain", "configs.py",
        'def lookup(settings, section, key):\n'
        '    """A setting, or None when either level is absent."""\n'
        '    return settings[section][key]\n',
        'def lookup(settings, section, key):\n'
        '    """A setting, or None when either level is absent."""\n'
        '    return settings.get(section, {}).get(key)\n',
        'from configs import lookup\n'
        '\n'
        'assert lookup({"a": {"b": 1}}, "a", "b") == 1\n'
        'assert lookup({"a": {}}, "a", "b") is None\n'
        'assert lookup({}, "a", "b") is None\n'
        'print("ok")\n',
    ),
    _bug(
        # `max` over a mapping compares its keys. The function reads as if it returns the highest
        # scorer and returns the last name in alphabetical order instead — wrong, and plausible.
        "fix-max-returns-key", "scores.py",
        'def best(scores):\n'
        '    """The name of the highest scorer."""\n'
        '    return max(scores)\n',
        'def best(scores):\n'
        '    """The name of the highest scorer."""\n'
        '    return max(scores, key=scores.get)\n',
        'from scores import best\n'
        '\n'
        'table = {"ada": 3, "bob": 9, "cy": 5}\n'
        'assert best(table) == "bob", best(table)\n'
        'print("ok")\n',
    ),
    _bug(
        # A guard that cannot fail: `len(items) >= 0` holds for every list including the empty one,
        # so the branch meant to protect the index never runs.
        "fix-always-true-guard", "peek.py",
        'def last(items):\n'
        '    """The final item, or None when there is none."""\n'
        '    return items[-1] if len(items) >= 0 else None\n',
        'def last(items):\n'
        '    """The final item, or None when there is none."""\n'
        '    return items[-1] if items else None\n',
        'from peek import last\n'
        '\n'
        'assert last([1, 2, 3]) == 3\n'
        'assert last([]) is None, last([])\n'
        'print("ok")\n',
    ),
    _bug(
        # The accumulator is created inside the loop, so every element carries only itself and the
        # function returns a copy of its input while looking like it accumulates.
        "fix-accumulator-reset", "running.py",
        'def running(values):\n'
        '    """Each value plus everything before it."""\n'
        '    out = []\n'
        '    for value in values:\n'
        '        total = 0\n'
        '        total += value\n'
        '        out.append(total)\n'
        '    return out\n',
        'def running(values):\n'
        '    """Each value plus everything before it."""\n'
        '    out = []\n'
        '    total = 0\n'
        '    for value in values:\n'
        '        total += value\n'
        '        out.append(total)\n'
        '    return out\n',
        'from running import running\n'
        '\n'
        'assert running([1, 2, 3]) == [1, 3, 6], running([1, 2, 3])\n'
        'print("ok")\n',
    ),
    _bug(
        # A list of lists built one element at a time with `append`, so the result is a list of
        # lists where the callers of this function expect a flat one.
        "fix-append-not-extend", "flat.py",
        'def flatten(groups):\n'
        '    """Every value from a list of lists, as one list."""\n'
        '    out = []\n'
        '    for group in groups:\n'
        '        out.append(group)\n'
        '    return out\n',
        'def flatten(groups):\n'
        '    """Every value from a list of lists, as one list."""\n'
        '    out = []\n'
        '    for group in groups:\n'
        '        out.extend(group)\n'
        '    return out\n',
        'from flat import flatten\n'
        '\n'
        'assert flatten([[1, 2], [3]]) == [1, 2, 3], flatten([[1, 2], [3]])\n'
        'print("ok")\n',
    ),
    _bug(
        # `list.sort` sorts in place and returns None, so a function written as `return words.sort()`
        # returns nothing at all.
        "fix-sort-returns-none", "order.py",
        'def ordered(words):\n'
        '    """The words in alphabetical order."""\n'
        '    return words.sort()\n',
        'def ordered(words):\n'
        '    """The words in alphabetical order."""\n'
        '    return sorted(words)\n',
        'from order import ordered\n'
        '\n'
        'assert ordered(["b", "a"]) == ["a", "b"], ordered(["b", "a"])\n'
        'print("ok")\n',
    ),
    _bug(
        # An optional cap that is compared even when it was not given, so the documented default
        # path raises instead of returning the value unchanged.
        "fix-none-cap", "limits.py",
        'def limit(value, cap=None):\n'
        '    """The value, never above the cap when one is given."""\n'
        '    return min(value, cap)\n',
        'def limit(value, cap=None):\n'
        '    """The value, never above the cap when one is given."""\n'
        '    return value if cap is None else min(value, cap)\n',
        'from limits import limit\n'
        '\n'
        'assert limit(5) == 5, limit(5)\n'
        'assert limit(5, 3) == 3, limit(5, 3)\n'
        'print("ok")\n',
    ),
    _bug(
        # Numbering that starts where `enumerate` starts rather than where the docstring says.
        "fix-enumerate-start", "listing.py",
        'def numbered(items):\n'
        '    """One line per item, numbered from one."""\n'
        '    return [f"{i}: {item}" for i, item in enumerate(items)]\n',
        'def numbered(items):\n'
        '    """One line per item, numbered from one."""\n'
        '    return [f"{i}: {item}" for i, item in enumerate(items, 1)]\n',
        'from listing import numbered\n'
        '\n'
        'assert numbered(["a", "b"]) == ["1: a", "2: b"], numbered(["a", "b"])\n'
        'print("ok")\n',
    ),
    _bug(
        # `zip` stops at the shorter input, so mismatched lists are silently paired short instead of
        # reported — the failure mode that produces a plausible result from bad data.
        "fix-zip-truncates", "pairs.py",
        'def pairs(names, values):\n'
        '    """Name and value for every name; the lists must be the same length."""\n'
        '    return list(zip(names, values))\n',
        'def pairs(names, values):\n'
        '    """Name and value for every name; the lists must be the same length."""\n'
        '    if len(names) != len(values):\n'
        '        raise ValueError(f"{len(names)} names against {len(values)} values")\n'
        '    return list(zip(names, values))\n',
        'from pairs import pairs\n'
        '\n'
        'assert pairs(["a"], [1]) == [("a", 1)]\n'
        'try:\n'
        '    pairs(["a", "b"], [1])\n'
        'except ValueError:\n'
        '    print("ok: a length mismatch is reported")\n'
        'else:\n'
        '    raise AssertionError("zip silently truncated to the shorter list")\n',
    ),
    _bug(
        # Removing entries while iterating the mapping they live in.
        "fix-dict-mutation-while-iterating", "prune.py",
        'def prune(counts):\n'
        '    """The counts with every zero removed."""\n'
        '    for word in counts:\n'
        '        if counts[word] == 0:\n'
        '            del counts[word]\n'
        '    return counts\n',
        'def prune(counts):\n'
        '    """The counts with every zero removed."""\n'
        '    return {word: count for word, count in counts.items() if count != 0}\n',
        'from prune import prune\n'
        '\n'
        'assert prune({"a": 1, "b": 0}) == {"a": 1}, prune({"a": 1, "b": 0})\n'
        'print("ok")\n',
    ),
    _bug(
        # `or` used as a default: it cannot tell "absent" from "falsy", so a legitimate zero is
        # replaced by the fallback.
        "fix-or-default-falsy", "ports.py",
        'def port(settings):\n'
        '    """The configured port, or 8080 when none is set."""\n'
        '    return settings.get("port") or 8080\n',
        'def port(settings):\n'
        '    """The configured port, or 8080 when none is set."""\n'
        '    value = settings.get("port")\n'
        '    return 8080 if value is None else value\n',
        'from ports import port\n'
        '\n'
        'assert port({}) == 8080\n'
        'assert port({"port": 0}) == 0, port({"port": 0})\n'
        'assert port({"port": 9000}) == 9000\n'
        'print("ok")\n',
    ),
    _bug(
        # Parentheses instead of a tuple: `(value)` is the value, so the function returns a number
        # where every caller unpacks or indexes.
        "fix-one-element-tuple", "wrap.py",
        'def wrap(value):\n'
        '    """A one-element tuple holding the value."""\n'
        '    return (value)\n',
        'def wrap(value):\n'
        '    """A one-element tuple holding the value."""\n'
        '    return (value,)\n',
        'from wrap import wrap\n'
        '\n'
        'assert wrap(3) == (3,), wrap(3)\n'
        'assert isinstance(wrap(3), tuple), type(wrap(3))\n'
        'print("ok")\n',
    ),
    _bug(
        # A ratio that does not handle the denominator the docstring says it handles.
        "fix-division-by-zero", "ratios.py",
        'def ratio(part, whole):\n'
        '    """Part as a fraction of whole, or None when whole is zero."""\n'
        '    return part / whole\n',
        'def ratio(part, whole):\n'
        '    """Part as a fraction of whole, or None when whole is zero."""\n'
        '    return None if whole == 0 else part / whole\n',
        'from ratios import ratio\n'
        '\n'
        'assert ratio(1, 4) == 0.25\n'
        'assert ratio(1, 0) is None, ratio(1, 0)\n'
        'print("ok")\n',
    ),
    _bug(
        # A grid whose rows are one list repeated, so writing a cell writes it in every row. The
        # classic aliasing defect, and invisible until something mutates the result.
        "fix-row-alias", "grid.py",
        'def grid(rows, cols):\n'
        '    """A rows-by-cols grid of zeroes."""\n'
        '    return [[0] * cols] * rows\n',
        'def grid(rows, cols):\n'
        '    """A rows-by-cols grid of zeroes."""\n'
        '    return [[0] * cols for _ in range(rows)]\n',
        'from grid import grid\n'
        '\n'
        'cells = grid(2, 3)\n'
        'cells[0][0] = 1\n'
        'assert cells[1][0] == 0, cells\n'
        'print("ok")\n',
    ),
    _bug(
        # `+=` on a list extends the caller\'s list rather than producing a new one, so a function
        # that reads as pure changes its argument.
        "fix-augmented-assign-mutates", "tags.py",
        'def add_tag(tags, tag):\n'
        '    """A new list of tags with this one added."""\n'
        '    tags += [tag]\n'
        '    return tags\n',
        'def add_tag(tags, tag):\n'
        '    """A new list of tags with this one added."""\n'
        '    return tags + [tag]\n',
        'from tags import add_tag\n'
        '\n'
        'original = ["a"]\n'
        'result = add_tag(original, "b")\n'
        'assert result == ["a", "b"], result\n'
        'assert original == ["a"], original\n'
        'print("ok")\n',
    ),
)


#: What the prompt says. One sentence for every task, and deliberately so: the defect is the same
#: kind of thing in all of them, and a prompt written per task would be a prompt that tells the model
#: where to look. "Do not run commands and do not change files" is not politeness — `minimal-rl`,
#: the only builtin track two arms can share, gates everything but `read` behind an approval nobody
#: can answer, and a task that asks for a command the track will refuse measures the track.
BUG_PROMPT = (
    "Something in this directory is broken. Read the files and tell me what the defect is. "
    "Do not run any commands and do not change any files — I want to know what is wrong first."
)

#: For each task, phrases a run that found the defect will use and a run that did not will not. The
#: prompt never names them. They are matched against the run's own words by
#: `harness/tasks/check_diagnosis.py`, because a read-only task's result is an answer rather than a
#: changed workspace.
EXPECT: dict[str, tuple[str, ...]] = {
    "fix-assertion": ("assert", "mean"),
    "fix-import-name": ("normalise", "normalize"),
    "fix-off-by-one": ("range", "len"),
    "fix-config-key": ("timeout_seconds", "timeout"),
    "fix-cli-flag": ("--row", "--rows"),
    "fix-type-error": ("base_delay", "string"),
    "fix-path": (".txt", ".csv"),
    "fix-regex": ("INFO", "pattern"),
    "fix-json-trailing-comma": ("json", "trailing"),
    "fix-iteration-mutation": ("iterat", "del"),
    "fix-wrong-operator": ("<=", "overlap"),
    "fix-silent-truncation": ("chunk", "size"),
    # The held-out four use a single word that only a run which *found* the defect would write, and
    # they are strict on purpose. Measured against the first tuned run on these tasks: three answers
    # described the code and concluded it was fine, and a looser pair of phrases matched them anyway
    # — "splits a given text" contains "split" while the answer says there is no bug. Each phrase here
    # names the defect itself: an IndexError, an argument order, a missing format prefix, a slice.
    #
    # `slic` rather than `slice` because the word an answer uses is "slicing", and `slice` is not a
    # substring of it — the e becomes i. Caught by running the check against the answer that was
    # plainly correct and watching it fail.
    "fix-argument-order": ("argument",),
    "fix-empty-input": ("index", "empty"),
    "fix-missing-prefix": ("prefix|f-string|formatted",),
    "fix-slice-direction": ("slic",),
    # The wider set. One word each, chosen to name the defect rather than its subject — the lesson
    # from the held-out four above, where a phrase matching the subject passed an answer saying
    # there was no bug at all.
    "fix-default-mutable": ("mutable|shared|default argument",),
    "fix-integer-division": ("floor", "int"),
    "fix-shadowed-total": ("assign", "+="),
    "fix-string-case": ("case", "lower|insensitive|capital|uppercase|sensitive"),
    "fix-missing-dict-default": ("keyerror", "get"),
    # `sort` alongside `key` because `key` alone is satisfied by the word "KeyError", which an answer
    # about a sorting defect may well use for a different reason — the same one-sidedness that let
    # the control arm's confabulations pass. An answer that found this defect says either `sorted` or
    # `sorting`, so the second phrase costs nothing and closes the collision.
    "fix-sort-key": ("key", "sort"),
    "fix-parity-test": ("odd", "even"),
    "fix-loop-continue": ("continue",),
    "fix-missing-return": ("return",),
    "fix-min-max-clamp": ("min", "max"),
    "fix-shallow-copy": ("table|blanked", "copy|in.?place|mutat|changes the original|modifies the original"),
    # These three shared the single phrase `index`, which made them interchangeable: an answer
    # saying only "there is an IndexError" passed all three. Each now carries a second phrase that
    # only its own defect produces — an empty input, a 1-based position, a two-part assumption.
    "fix-ordinal-index": ("index", "based|off.by.one|zero"),
    "fix-single-name": ("index", "two part|one part|one-word|single"),
    # The second tranche, 2026-10-11. Each pair names the mechanism the defect actually involves,
    # not the area of code it sits in: an answer that says "there is a bug in clean.py" names
    # neither phrase, and an answer that found the missing call says `strip` and `call`.
    "fix-strip-whitespace": ("strip", "call"),
    "fix-float-equal": ("float", "compar"),
    "fix-closure-late-binding": ("closure|capture|lambda", "bind|by reference|by value|same variable|outer scope"),
    "fix-generator-exhaust": ("generator", "exhaust|consum|twice|already|reuse"),
    "fix-broad-except": ("bare|catches|swallow|blanket|indiscriminat", "except"),
    "fix-string-concat-int": ("str", "concat"),
    "fix-dict-get-chain": ("both levels|nested|index|lookup", "keyerror|missing|absent|get"),
    "fix-max-returns-key": ("max", "dict"),
    "fix-always-true-guard": ("empty", "guard"),
    "fix-accumulator-reset": ("accumulat|total", "reset|inside the loop|each iteration|reinitiali"),
    # The second half of the tranche: scope, iteration and the falsy/None distinction.
    "fix-append-not-extend": ("extend", "append"),
    "fix-sort-returns-none": ("sorted", "none"),
    "fix-none-cap": ("none", "cap"),
    "fix-enumerate-start": ("enumerate", "start"),
    "fix-zip-truncates": ("zip|pairs", "length|truncat|mismatch|silently"),
    "fix-dict-mutation-while-iterating": ("dict|mapping", "iterat|delet|mutat|modif"),
    "fix-or-default-falsy": ("falsy|fals|zero|0", "none|8080|default"),
    "fix-one-element-tuple": ("tuple", "parenthes"),
    "fix-division-by-zero": ("zero", "division"),
    "fix-row-alias": ("alias|referenc|shared|same (inner )?list|same object|shallow copy",),
    "fix-augmented-assign-mutates": ("caller|original|same list|mutat|in.?place", "list|tags"),
}

#: Phrases that must **not** appear, per task. A phrase list can only ask whether the right words are
#: present; it cannot ask whether the answer has wandered into a neighbouring defect. The three
#: `index` tasks are the case in point — they are mutually exclusive fixtures, and an answer about
#: one of them naming another's mechanism is describing a file it did not read.
#:
#: Kept deliberately small. A forbidden phrase is a claim about what a *correct* answer would never
#: say, and that claim is easy to get wrong; the invented-file check covers the general case without
#: anyone having to predict the wording.
FORBID: dict[str, tuple[str, ...]] = {
    "fix-empty-input": ("1-based", "off-by-one"),
    "fix-ordinal-index": ("empty string", "empty input"),
    "fix-single-name": ("1-based", "off-by-one"),
    # A phrase list is loosened so that a correct diagnosis in unfamiliar words still passes; the
    # loosening has to be paid for, or it starts passing answers that concluded the opposite. These
    # two tasks are where that happened. `fix-or-default-falsy` names the right file, the right
    # expression and the word `falsy`, and then says the function is *correct* — while `0 or 8080`
    # returns 8080, not 0. Measured on 2026-10-11, that answer scored as a pass until this list
    # existed. An answer that says the code is fine has not diagnosed the defect the task is about,
    # whatever words it used on the way.
    "fix-or-default-falsy": ("otherwise correct", "correctly returns", "no defect",
                             "not a defect", "code is correct", "appears to be correct"),
    "fix-none-cap": ("no defect", "not a defect", "appears to be correct",
                     "code is correct", "there is no"),
}

#: Tasks no run will train on and no result reported so far has used. The held-out four in the tuple
#: above were written before run 2 and were used to measure it; reusing them would measure the same
#: model on the same questions and call it evidence. `build` leaves these out of the training set
#: rather than trusting a caller to remember.
#:
#: **Fifteen as of 2026-10-11, from five.** Five tasks gave a score where one task was twenty points
#: and the base model's own run-to-run range measured wider than any gap between arms — a set that
#: cannot see an effect cannot support a claim, in either direction. The ten added here were chosen
#: to span the defect kinds the second tranche introduced (scope, iteration, error handling, the
#: falsy/None distinction, aliasing) rather than to be the ten easiest, so a difference that shows
#: up on them is a difference in the capability and not in one fixture's phrasing.
FINAL_HOLDOUT: frozenset[str] = frozenset({
    "fix-string-case",
    "fix-sort-key",
    "fix-shallow-copy",
    "fix-ordinal-index",
    "fix-single-name",
    # Added with the second tranche, 2026-10-11.
    "fix-strip-whitespace",
    "fix-closure-late-binding",
    "fix-broad-except",
    "fix-generator-exhaust",
    "fix-none-cap",
    "fix-enumerate-start",
    "fix-row-alias",
    "fix-augmented-assign-mutates",
    "fix-zip-truncates",
    "fix-or-default-falsy",
})

CHECK_SCRIPT = Path(__file__).resolve().parent / "check_diagnosis.py"


def check_command(task_id: str) -> str:
    """The command a round runs to judge this task.

    Absolute, because a check runs with the run's workspace as its working directory and the depth
    from there to this repository is a property of where the round wrote its workspaces. The
    alternative — a relative walk — is a path that silently resolves somewhere else.

    Both phrase lists travel on the command line, separated by ``--forbid``, because a stored `check`
    string is the only channel a task has: it runs in the run's workspace with no environment of its
    own. The forbidden list is omitted entirely when a task has none, so an existing command is
    unchanged in meaning.
    """
    phrases = " ".join(f'"{phrase}"' for phrase in EXPECT[task_id])
    forbid = FORBID.get(task_id, ())
    if forbid:
        banned = " ".join(f'"{phrase}"' for phrase in forbid)
        return f'python "{CHECK_SCRIPT}" {phrases} --forbid {banned}'
    return f'python "{CHECK_SCRIPT}" {phrases}'


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

    Each row carries two commands, for two different questions. ``verify`` is the original check —
    it decides whether the fixture is broken, and whether the fix repairs it, which is what
    :func:`self_check` uses before a rollout is spent. ``check`` is what a round runs: a diagnosis
    reader, because the task asks for a diagnosis.
    """
    root = Path(destination)
    rows: list[dict[str, str]] = []
    for task in TASKS:
        workspace = root / task["id"]
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        _write_files(workspace, task["files"])
        if task["id"] not in EXPECT:
            raise KeyError(
                f"{task['id']} has no entry in EXPECT, so a round could not judge it. Every task "
                "needs phrases that only a run which found the defect would say."
            )
        rows.append(
            {
                "id": task["id"],
                "prompt": BUG_PROMPT,
                "verify": task["check"],
                "check": check_command(task["id"]),
                "expect": list(EXPECT[task["id"]]),
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
        handle.write("# Each task's workspace holds one real defect and the prompt names none of\n")
        handle.write("# them. `verify` decides whether the fixture is broken; `check` judges a\n")
        handle.write("# run's diagnosis, which is the outcome a gate validates against.\n")
        handle.write("# Resolve the workspace by id: python harness/tasks/build_workspaces.py\n")
        for row in rows:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "cwd"},
                                    ensure_ascii=False) + "\n")
    return target


def split(rows: Sequence[Mapping[str, str]]) -> tuple[list[Mapping[str, str]], list[Mapping[str, str]]]:
    """The rows a round may train on, and the rows it may not.

    A single call rather than two filters at two call sites: the holdout is only a holdout if every
    training path leaves it out, and a caller who forgets is a caller who trains on the test set
    without noticing.
    """
    training = [row for row in rows if row["id"] not in FINAL_HOLDOUT]
    held_out = [row for row in rows if row["id"] in FINAL_HOLDOUT]
    if not held_out:
        raise ValueError(
            "FINAL_HOLDOUT names nothing in the task set; a training set with no held-out tasks "
            "cannot be measured on anything it has not seen"
        )
    return training, held_out


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
        help="the training task set harness.rollout.load_tasks reads",
    )
    parser.add_argument(
        "--holdout-out",
        default="harness/tasks/final-holdout.jsonl",
        help="the tasks no round trains on",
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
    training, held_out = split(rows)
    path = write_task_set(training, args.out)
    held_path = write_task_set(held_out, args.holdout_out)
    print(f"\nwrote {len(rows)} workspaces under {args.workspaces}")
    print(f"wrote {path}  ({len(training)} training tasks)")
    print(f"wrote {held_path}  ({len(held_out)} held out)")
    for row in held_out:
        print(f"    held out: {row['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
