"""Prove the local copy can rebuild a round without the training box.

A backup is only as good as a restore that has been tried. This runs the actual rebuild — the same
code on the same local inputs — and compares the result to the digests that were trained on. When it
matches, the box can be lost and the round is still reproducible here; when it does not, the local copy
has a hole and this is the moment to find out.

**The trap this tool exists to catch, found on its own first run.** Task order is part of the digest.
`reviewed_forks` walks `manifest["task_ids"]` in the order it finds them, and `samples_digest` hashes
rows in order, so two runs over the same thirteen rows in different orders produce different digests.
The first version of this check fed the raw collection manifest and got 13 rows that were *identical
field for field* and a digest that did not match — which reads exactly like a corrupted backup until
you look. The round was built from a manifest whose ids had been sorted, so a reproduction has to use
that one; `--student-run` selects it and defaults to the filtered directory the round used.

Usage:

    python tools/verify_local_rebuild.py --round data/chtd-round-20261011

Exit 0 when every digest matches, 1 when one does not, 2 when the inputs are not all present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from distill.sft import samples_digest


def discover_round(root: Path) -> tuple[Path, dict[str, object]]:
    """The inputs a rebuild needs, and the digests it should reproduce.

    The expected digests come from the round's own record rather than from a constant in this file:
    a tool that hard-codes what it expects can only confirm things that have already happened.
    """
    built = root / "built"
    manifest_file = built / "reviewed-manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"no reviewed-manifest.json under {built}; nothing to check against")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    inputs = {
        "student run": root / "student-filtered",
        "forks": root / "forks",
        "review": root / "review.json",
        "ledger": root / "agh_home" / "data" / "sessions.db",
        "prefix": root / "prefix.json",
    }
    expected = {
        "samples_sha256": manifest.get("samples_sha256"),
        "dataset_sha256": manifest.get("dataset_sha256"),
    }
    return inputs, expected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=Path, required=True,
                        help="the round directory, e.g. data/chtd-round-20261011")
    parser.add_argument("--scratch", type=Path, default=None,
                        help="where to build; default is a temp directory beside the round")
    args = parser.parse_args(argv)

    root = args.round.resolve()
    if not root.is_dir():
        print(f"no such round: {root}")
        return 2

    inputs, expected = discover_round(root)
    print("inputs")
    missing = []
    for label, path in inputs.items():
        exists = path.exists()
        detail = ""
        if path.is_file():
            detail = f"{path.stat().st_size:,} bytes"
        elif path.is_dir():
            detail = f"{len(list(path.rglob('*')))} entries"
        print(f"  {'OK  ' if exists else 'MISS'}  {label:14s} {detail}")
        if not exists:
            missing.append(label)
    if missing:
        print(f"\ncannot rebuild: missing {missing}")
        return 2

    scratch = args.scratch or Path(tempfile_root(root))
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    print("\nrunning the rebuild")
    command = [
        sys.executable, "-m", "distill.reviewed_forks",
        "--student-run", str(inputs["student run"]),
        "--fork-run", str(inputs["forks"]),
        "--review", str(inputs["review"]),
        "--ledger", str(inputs["ledger"]),
        "--prefix", str(inputs["prefix"]),
        "--out", str(scratch),
    ]
    result = subprocess.run(command, cwd=str(REPO), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=1800, check=False)
    print((result.stdout or "").strip()[-300:])
    if result.returncode:
        print((result.stderr or "").strip()[-1200:])
        return 1

    rebuilt = scratch / "samples.jsonl"
    if not rebuilt.exists():
        print("the rebuild produced no samples.jsonl")
        return 1

    lines = [line for line in rebuilt.read_text(encoding="utf-8").splitlines() if line.strip()]
    got_samples = samples_digest(lines)
    manifest = json.loads((scratch / "reviewed-manifest.json").read_text(encoding="utf-8"))
    got_dataset = manifest.get("dataset_sha256")

    print("\ndigests")
    ok = True
    for label, got, want in (("samples", got_samples, expected["samples_sha256"]),
                             ("dataset", got_dataset, expected["dataset_sha256"])):
        match = got == want
        ok = ok and match
        print(f"  {label:8s} rebuilt  {got}")
        print(f"  {'':8s} expected {want}")
        print(f"  {'':8s} {'MATCH' if match else 'MISMATCH'}")

    print(f"\nrows rebuilt: {len(lines)}")
    print()
    if ok:
        print("VERDICT: the local copy rebuilds this round byte for byte.")
        print("         The box can be lost and the experiment is still reproducible here.")
    else:
        print("VERDICT: MISMATCH.")
        print("         Check the task order first — see this module's docstring. Then check that")
        print("         every input above came from the same round.")
    return 0 if ok else 1


def tempfile_root(root: Path) -> Path:
    return root.parent / f".rebuild-check-{root.name}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
