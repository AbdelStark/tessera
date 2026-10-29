#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR MIT
"""Run a pinned fuzz campaign; count wait4 CPU usage, never requested wall time."""

import argparse
import datetime
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BUDGETS = {
    name: 7200
    for name in (
        "decode",
        "envelope",
        "protocol",
        "multiproof",
        "inclusion",
        "sumcheck",
        "audit",
    )
}
BUDGETS.update(claims=14400, verify=21600)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.chunk_seconds <= 3600:
        parser.error("workers must be 1..8; chunk seconds must be 1..3600")
    args.out.mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.call(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "verifier",
            "spec/vectors",
            ".github/workflows/fuzz.yml",
        ],
        cwd=ROOT,
    ):
        parser.error("commit the campaign source before running")
    binaries = {n: args.binaries.resolve() / n for n in BUDGETS}
    seed_paths = {n: args.seeds.resolve() / n for n in BUDGETS}
    manifest = {}
    for name in BUDGETS:
        files = sorted(seed_paths[name].iterdir())
        if not files:
            parser.error(f"no seeds for {name}")
        manifest[name] = {
            "binary_sha256": digest(binaries[name]),
            "seed_files": {p.name: digest(p) for p in files},
        }
        for sub in ("corpus", "artifacts", "logs"):
            (args.out / sub / name).mkdir(parents=True)
    fixture_paths = [
        ROOT / "verifier/crates/tv-core/tests/fixtures/driver" / n
        for n in ("ni.tnp", "claims.tcl", "weights.tce")
    ] + [
        ROOT / "spec/vectors/tamper" / n
        for n in ("ni.tnp", "claims.tcl", "weights.twr", "protocol-1.json")
    ]
    manifest["runtime_fixtures"] = {
        str(p.relative_to(ROOT)): digest(p) for p in fixture_paths
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    status = {
        "schema_version": 1,
        "source_revision": revision,
        "state": "running",
        "started_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "cpu_budget_seconds": BUDGETS,
        "manifest_file": "manifest.json",
        "manifest_sha256": digest(manifest_path),
        "runs": [],
        "live": [],
        "scope": "AddressSanitizer fuzz binaries; test-small certifies nothing; known-false cases have exact oracles.",
    }
    totals = dict.fromkeys(BUDGETS, 0.0)
    active = {}
    stop = False
    last_save = 0.0

    def interrupt(_sig, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)

    def save():
        status["clean_cpu_seconds"] = dict(totals)
        status["total_clean_cpu_hours"] = sum(totals.values()) / 3600
        status["updated_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        status["live"] = [
            {"pid": pid, "target": job[0], "wall_seconds": time.monotonic() - job[2]}
            for pid, job in active.items()
        ]
        temp = args.out / "status.tmp"
        temp.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temp.replace(args.out / "status.json")

    try:
        while not stop:
            if (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip()
                != revision
            ):
                status["state"] = "source-changed"
                break
            if subprocess.call(
                [
                    "git",
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    "verifier",
                    "spec/vectors",
                    ".github/workflows/fuzz.yml",
                ],
                cwd=ROOT,
            ):
                status["state"] = "source-changed"
                break
            busy = {j[0] for j in active.values()}
            pending = [n for n in BUDGETS if totals[n] < BUDGETS[n] and n not in busy]
            pending.sort(key=lambda n: totals[n] / BUDGETS[n])
            for name in pending[: args.workers - len(active)]:
                number = sum(r["target"] == name for r in status["runs"])
                seconds = min(
                    args.chunk_seconds, max(1, math.ceil(BUDGETS[name] - totals[name]))
                )
                max_len = (
                    4096
                    if name in ("decode", "envelope")
                    else 65536
                    if name == "claims"
                    else 2 << 20
                )
                command = [
                    str(binaries[name]),
                    str((args.out / "corpus" / name).resolve()),
                    str(seed_paths[name]),
                    f"-max_total_time={seconds}",
                    "-timeout=10",
                    "-rss_limit_mb=1024",
                    f"-max_len={max_len}",
                    "-print_final_stats=1",
                    f"-seed={0x1850000 + len(status['runs']) + len(active)}",
                    f"-artifact_prefix={(args.out / 'artifacts' / name).resolve()}/",
                ]
                log = args.out / "logs" / name / f"{number:04}.log"
                with log.open("xb") as stream:
                    process = subprocess.Popen(
                        command,
                        cwd=HERE,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                    )
                active[process.pid] = (
                    name,
                    process,
                    time.monotonic(),
                    command,
                    str(log),
                    seconds,
                )
            if not active:
                status["state"] = "complete"
                break
            for pid, job in list(active.items()):
                waited, result, usage = os.wait4(pid, os.WNOHANG)
                if not waited:
                    if time.monotonic() - job[2] > job[5] + 180:
                        job[1].kill()
                        status["state"] = "worker-overdue"
                        stop = True
                    continue
                name, process, started, command, log, _seconds = active.pop(pid)
                code = os.waitstatus_to_exitcode(result)
                process.returncode = code
                cpu = usage.ru_utime + usage.ru_stime
                status["runs"].append(
                    {
                        "target": name,
                        "pid": pid,
                        "command": command,
                        "exit_code": code,
                        "user_cpu_seconds": usage.ru_utime,
                        "system_cpu_seconds": usage.ru_stime,
                        "cpu_seconds": cpu,
                        "wall_seconds": time.monotonic() - started,
                        "log": log,
                        "max_rss_platform_units": usage.ru_maxrss,
                    }
                )
                if code == 0:
                    totals[name] += cpu
                else:
                    status["state"] = "finding-or-worker-failure"
                    stop = True
            if time.monotonic() - last_save >= 10:
                save()
                last_save = time.monotonic()
            time.sleep(1)
    except BaseException as error:
        status["state"] = "controller-error"
        status["error_type"] = type(error).__name__
        raise
    finally:
        for pid, job in list(active.items()):
            job[1].terminate()
        for pid, job in list(active.items()):
            limit = time.monotonic() + 5
            while True:
                waited, result, usage = os.wait4(pid, os.WNOHANG)
                if waited:
                    break
                if time.monotonic() >= limit:
                    job[1].kill()
                    _, result, usage = os.wait4(pid, 0)
                    break
                time.sleep(0.05)
            job[1].returncode = os.waitstatus_to_exitcode(result)
            status["runs"].append(
                {
                    "target": job[0],
                    "pid": pid,
                    "exit_code": job[1].returncode,
                    "cpu_seconds": usage.ru_utime + usage.ru_stime,
                    "interrupted": True,
                    "log": job[4],
                }
            )
        active.clear()
        if status["state"] == "running":
            status["state"] = "interrupted"
        if status["state"] == "complete":
            assert all(totals[n] >= BUDGETS[n] for n in BUDGETS)
            assert sum(totals.values()) >= 24 * 3600
            assert not any(p.is_file() for p in (args.out / "artifacts").rglob("*")), (
                "artifact despite successful workers"
            )
        save()
    print(
        json.dumps(
            {
                k: status[k]
                for k in ("state", "total_clean_cpu_hours", "source_revision")
            }
        )
    )
    return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
