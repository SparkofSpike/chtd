"""Bring the irreplaceable artefacts from the training box to this machine.

Why this exists rather than a `scp -r`. The path to the box is a Tailscale DERP relay -- the box sits
behind a symmetric NAT, so no direct connection can be established -- and it manages 100-200 KB/s with
a failure mode that is worse than slowness: **a transfer can stop making progress and sit there
indefinitely**. Measured: one 264 MB `scp` froze at 12,353,536 bytes for over a minute and never moved
again; eight parallel 33 MB transfers moved zero bytes in 140 seconds. Chunking with a stall detector
fixed it -- the same link then sustained 114-171 KB/s, because the capacity was never the problem, the
long-lived connection was.

So the design is: 4 MB chunks, a fresh connection each, a watchdog that kills a chunk after
`--stall-seconds` without progress, and a retry. A stall then costs forty seconds instead of half an
hour. Every assembled file is checked against the sha256 **the box reports for the original**, not
against a hash of the bytes this script just wrote — an integrity check that compares a copy to itself
checks nothing.

See `docs/artefact-safety.md` for what each of the three copies protects against, and for the
box-side mirror that runs at local disk speed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: The artefacts, most valuable first. A name is a label; the path is on the box.
#:
#: Order matters when the link is slow and the session is finite: the arm the current experiment rests
#: on comes before the ones kept as contrast.
DEFAULT_ARTEFACTS: list[tuple[str, str]] = [
    ("round20261011-allturns",
     "/mnt/proj/chtd-experiments/round-20261011/adapter-allturns/adapter_model.safetensors"),
    ("matched-chtd",
     "/mnt/proj/chtd-experiments/matched-20261010-a/chtd-adapter/adapter_model.safetensors"),
    ("matched-baseline",
     "/mnt/proj/chtd-experiments/matched-20261010-a/baseline-adapter/adapter_model.safetensors"),
    ("round20261011-lastonly",
     "/mnt/proj/chtd-experiments/round-20261011/adapter/adapter_model.safetensors"),
    ("chtd-20261011-a",
     "/mnt/proj/chtd-experiments/chtd-20261011-a/adapter/adapter_model.safetensors"),
    ("baseline-pilot",
     "/mnt/proj/chtd-experiments/baseline-20261010-a/pilot-adapter/adapter_model.safetensors"),
]


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def say(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Box:
    """The remote side, through one ssh/scp pair of option sets."""

    def __init__(self, host: str, workdir: str = "/tmp/chtd-artefact-parts") -> None:
        common = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
                  "-o", "UserKnownHostsFile=NUL"]
        self.host = host
        self.workdir = workdir
        self.ssh = ["ssh", *common, host]
        self.scp = ["scp", *common]

    def run(self, command: str, timeout: int = 300) -> subprocess.CompletedProcess:
        return subprocess.run([*self.ssh, command], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout, check=False)

    def digest(self, path: str) -> str | None:
        r = self.run(f"sha256sum {path}")
        if r.returncode:
            return None
        return r.stdout.split()[0]

    def split(self, path: str, label: str, chunk_mb: int) -> list[tuple[str, int]]:
        """Split on the box and return (remote path, size) for each chunk."""
        directory = f"{self.workdir}/{label}"
        r = self.run(f"rm -rf {directory} && mkdir -p {directory} && "
                     f"split -b {chunk_mb}m -d -a 3 {path} {directory}/part-")
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout).strip()[:300])
        listing = self.run(f"stat -c '%n %s' {directory}/part-*").stdout.strip().splitlines()
        parts = [(line.split()[0], int(line.split()[1])) for line in listing if line.strip()]
        if not parts:
            raise RuntimeError(f"no chunks produced for {label}")
        return parts

    def clean(self, label: str) -> None:
        self.run(f"rm -rf {self.workdir}/{label}", timeout=120)


def _kill(process: subprocess.Popen) -> None:
    """Stop a transfer that is not going to finish.

    Not finishing is the expected case on this link, so this is ordinary control flow rather than an
    error path: a process that has already exited is not a problem, which is why the lookup and timeout
    failures are swallowed here and reported by the caller instead.
    """
    try:
        process.kill()
        process.wait(timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass


def fetch(box: Box, source: str, target: Path, size: int, stall_seconds: int,
          ceiling: int) -> bool:
    """One chunk under a watchdog. True when the file ended exactly `size` bytes long.

    The watchdog is the whole point. A transfer here does not merely run slowly: it can stop moving and
    stay stopped, which a plain `scp` call turns into an unbounded wait. Watching the destination file
    converts that into a bounded retry.
    """
    if target.exists() and target.stat().st_size == size:
        return True
    if target.exists():
        target.unlink()

    process = subprocess.Popen([*box.scp, source, str(target)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last_size, last_change, started = -1, time.time(), time.time()
    try:
        while process.poll() is None:
            time.sleep(3)
            current = target.stat().st_size if target.exists() else 0
            if current != last_size:
                last_size, last_change = current, time.time()
            elif time.time() - last_change > stall_seconds:
                say(f"        stalled at {current:,}/{size:,} after "
                    f"{time.time() - started:.0f}s")
                return False
            if time.time() - started > ceiling:
                say(f"        over the {ceiling}s ceiling at {current:,}/{size:,}")
                return False
    finally:
        # Every exit from the loop above — settled, stalled, timed out — leaves the process to be
        # reaped here, so no branch can leak one.
        if process.poll() is None:
            _kill(process)
        else:
            process.wait()

    got = target.stat().st_size if target.exists() else 0
    elapsed = time.time() - started
    if got == size:
        say(f"        ok {got:>9,}  {elapsed:4.0f}s  {got / elapsed / 1024:5.0f} KB/s")
        return True
    say(f"        short {got:>9,}/{size:,} after {elapsed:.0f}s")
    return False


def pull(box: Box, label: str, remote: str, landing: Path, scratch: Path,
         chunk_mb: int, stall_seconds: int, attempts: int, ceiling: int) -> bool:
    final = landing / f"{label}.safetensors"
    expected = box.digest(remote)
    if expected is None:
        say(f"  {label}: cannot hash {remote} on the box")
        return False
    if final.exists() and sha256(final) == expected:
        say(f"  {label}: already local and correct  {expected[:16]}")
        return True

    parts = box.split(remote, label, chunk_mb)
    total = sum(size for _, size in parts)
    say(f"  {label}: {len(parts)} chunks, {total:,} bytes, sha {expected[:16]}")

    local_dir = scratch / label
    local_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, size) in enumerate(parts, 1):
        target = local_dir / Path(name).name
        for attempt in range(1, attempts + 1):
            if fetch(box, f"{box.host}:{name}", target, size, stall_seconds, ceiling):
                break
            if attempt == attempts:
                say(f"    gave up on chunk {index} after {attempts} attempts")
                return False
            time.sleep(2 + attempt)
        if index % 10 == 0 or index == len(parts):
            say(f"    {index}/{len(parts)} chunks")

    say("    assembling")
    with final.open("wb") as out:
        for name, _ in parts:
            out.write((local_dir / Path(name).name).read_bytes())

    got = sha256(final)
    if got != expected:
        say(f"    HASH MISMATCH  got {got[:16]}  expected {expected[:16]}")
        final.unlink(missing_ok=True)
        return False
    say(f"    VERIFIED  {got[:16]}  {final.stat().st_size:,} bytes")
    box.clean(label)
    shutil.rmtree(local_dir, ignore_errors=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@100.84.151.62",
                        help="the training box, as ssh sees it")
    parser.add_argument("--landing", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data" / "adapters",
                        help="where the artefacts are written locally")
    parser.add_argument("--chunk-mb", type=int, default=4)
    parser.add_argument("--stall-seconds", type=int, default=40)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--ceiling", type=int, default=900,
                        help="give up on a single chunk after this many seconds regardless")
    parser.add_argument("--only", action="append", default=None,
                        help="pull just these labels; repeatable")
    args = parser.parse_args(argv)

    wanted = set(args.only) if args.only else None
    artefacts = [item for item in DEFAULT_ARTEFACTS if not wanted or item[0] in wanted]
    if not artefacts:
        print(f"no artefact matches {sorted(wanted)}")
        return 2

    landing = args.landing
    scratch = landing / "_parts"
    landing.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    box = Box(args.host)

    say(f"{len(artefacts)} artefacts -> {landing}")
    say(f"{args.chunk_mb} MB chunks, {args.stall_seconds}s stall detector, {args.attempts} attempts")
    results = {}
    for label, remote in artefacts:
        say(f"--- {label}")
        try:
            results[label] = pull(box, label, remote, landing, scratch,
                                  args.chunk_mb, args.stall_seconds, args.attempts, args.ceiling)
        except Exception as exc:  # noqa: BLE001 - deliberate: see below
            # One artefact failing must not abandon the rest. On a link that stalls, a single bad
            # transfer is an expected outcome rather than a bug, and the alternative — narrowing to
            # the exception types anticipated — aborts the run on the one that was not. The failure is
            # recorded against its label and the summary reports it.
            say(f"    {type(exc).__name__}: {exc}")
            results[label] = False

    say("")
    say("SUMMARY")
    for label, ok in results.items():
        say(f"  {'OK  ' if ok else 'FAIL'}  {label}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
