"""Shared helpers for the reproduce/ scripts.

Every experiment run is one subprocess of the experiment code. Runs with an output and a matching successful-completion record
are skipped, so an interrupted script can simply be restarted. `--dry-run`
prints one JSON line per run (name, working directory, environment, command) and exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class Run:
    name: str
    cmd: list[str]
    cwd: Path
    done: Path  # expected final output, checked with the successful-completion record
    log: Path
    env: dict[str, str] = field(default_factory=dict)


def parser(doc: str, default_out: str | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    if default_out is not None:
        p.set_defaults(out_root=str(REPO / "outputs" / default_out))
    p.add_argument("--workers", type=int, default=1, help="number of processes in parallel")
    p.add_argument("--dry-run", action="store_true", help="print the planned runs and exit")
    return p


def python() -> str:
    return sys.executable


def execute(runs: list[Run], workers: int, dry_run: bool) -> int:
    if dry_run:
        for r in runs:
            print(json.dumps({"name": r.name, "cwd": os.path.relpath(r.cwd, REPO), "env": r.env,
                              "cmd": [os.path.basename(r.cmd[0])] + r.cmd[1:]}))
        return 0
    todo = []
    for r in runs:
        marker = r.log.with_suffix(".success.json")
        signature = {"cmd": r.cmd, "env": r.env, "cwd": str(r.cwd), "done": str(r.done)}
        try:
            saved = json.loads(marker.read_text())
            complete = (r.done.is_file() and saved["signature"] == signature
                        and saved["sha256"] == hashlib.sha256(r.done.read_bytes()).hexdigest())
        except (OSError, ValueError, KeyError):
            complete = False
        if complete:
            print(f"[reproduce] skip {r.name} (completed successfully)", flush=True)
        else:
            todo.append(r)

    def one(r: Run) -> int:
        r.log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        for var in THREAD_VARS:
            env[var] = "1"  # one thread per process; parallelism comes from --workers
        env.update(r.env)
        marker = r.log.with_suffix(".success.json")
        marker.unlink(missing_ok=True)
        with open(r.log, "w") as log:
            rc = subprocess.run(r.cmd, cwd=r.cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc == 0 and not r.done.is_file():
            print(f"[reproduce] missing expected output: {r.done}", flush=True)
            rc = 1
        if rc == 0:
            saved = {"signature": {"cmd": r.cmd, "env": r.env, "cwd": str(r.cwd), "done": str(r.done)},
                     "sha256": hashlib.sha256(r.done.read_bytes()).hexdigest()}
            temporary = marker.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved, sort_keys=True) + "\n")
            temporary.replace(marker)
        print(f"[reproduce] {r.name}: exit {rc} (log {r.log})", flush=True)
        return rc

    print(f"[reproduce] {len(todo)} runs, {workers} in parallel", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        failed = [r.name for r, rc in zip(todo, pool.map(one, todo)) if rc != 0]
    if failed:
        print(f"[reproduce] FAILED: {failed}", flush=True)
        return 1
    return 0
