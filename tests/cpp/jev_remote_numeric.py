#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run one exact scheduler GTest filter in record and local-API modes.

CTest must invoke this under the governed resource wrapper. The parity server
is an in-process thread and is closed through its own shutdown API. This
program never sends a signal or times out a child process.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from threading import Thread
from typing import Any


GTEST_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\Z")
CASE_KEY = re.compile(r"jev_remote_numeric_[a-z_]+\Z")
SITE_PREFIX = re.compile(r"(?:PW|RD|TR|IP|OP|IOP)\Z")
PASSED = re.compile(r"^\[\s+PASSED\s+\]\s+(\d+) tests?\.$", re.M)
RUNNING = re.compile(r"^\[==========\] Running (\d+) tests? from", re.M)
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    # Preserve the last complete stage even when a numerical phase fails.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checked_environment() -> tuple[Path, Path, list[str]]:
    module_dir = Path(os.environ.get("JEV_PARITY_MODULE_DIR", ""))
    loader = Path(os.environ.get("JEV_NUMERIC_LOADER", ""))
    libraries_text = os.environ.get("JEV_NUMERIC_LIBRARY_PATH", "")
    if not module_dir.is_absolute() or not module_dir.is_dir():
        raise ValueError("JEV_PARITY_MODULE_DIR must be an existing absolute directory")
    for name in ("jev_shim.py", "compare_decisions.py"):
        if not (module_dir / name).is_file():
            raise ValueError(f"JEV_PARITY_MODULE_DIR lacks {name}")
    if not loader.is_absolute() or not loader.is_file():
        raise ValueError("JEV_NUMERIC_LOADER must name an existing absolute loader")
    libraries = libraries_text.split(":")
    if not libraries or any(not part or not Path(part).is_absolute() or
                            not Path(part).is_dir() for part in libraries):
        raise ValueError("JEV_NUMERIC_LIBRARY_PATH must contain only existing absolute directories")
    return module_dir, loader, libraries


def check_numeric_output(output: str, expected: int) -> dict[str, int]:
    running = [int(value) for value in RUNNING.findall(output)]
    passed = [int(value) for value in PASSED.findall(output)]
    if running != [expected] or passed != [expected] or "[  SKIPPED ]" in output:
        raise ValueError(
            f"GTest ran/passed {running}/{passed}; expected exactly {expected} "
            "non-skipped numerical tests"
        )
    return {"running": running[0], "passed": passed[0]}


def run_numeric(
    *, mode: str, binary: Path, names: list[str], case_key: str,
    loader: Path, libraries: list[str], directory: Path, port: int | None,
) -> dict[str, Any]:
    decision_log = directory / f"{mode}.jsonl"
    output = directory / f"{mode}.stdout.txt"
    errors = directory / f"{mode}.stderr.txt"
    decision_log.touch(exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({"NVFUSER_JEV_MODE": mode,
                "NVFUSER_JEV_LOG": str(decision_log),
                "NVFUSER_JEV_CASE_KEY": case_key})
    if port is not None:
        env["NVFUSER_JEV_PORT"] = str(port)
    argv = [str(loader), "--library-path", ":".join(libraries),
            str(binary), "--gtest_color=no",
            "--gtest_filter=" + ":".join(names)]
    # The enclosing CTest process is the same foreground governed workload.
    # Do not use subprocess timeout: Python would signal the child on expiry.
    with output.open("x", encoding="utf-8") as stdout, errors.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, check=False)
    summary: dict[str, Any] = {
        "mode": mode, "argv": argv, "returncode": completed.returncode,
        "stdout": str(output), "stderr": str(errors), "decision_log": str(decision_log),
    }
    for path in (output, errors, decision_log):
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError(f"{path} exceeds the {MAX_ARTIFACT_BYTES}-byte artifact bound")
    summary["numeric"] = check_numeric_output(output.read_text(errors="replace"), len(names))
    summary["artifact_sha256"] = {path.name: file_hash(path) for path in
                                  (output, errors, decision_log)}
    if completed.returncode != 0:
        raise ValueError(f"{mode} GTest exited {completed.returncode}; see {errors}")
    return summary


def check_parity(report: dict[str, Any], case_key: str, prefix: str) -> None:
    if not report["ok"]:
        kinds = sorted({issue["kind"] for issue in report["issues"]})
        raise ValueError(f"record/remote/server decision parity failed: {kinds}")
    summary = report["summary"]
    if min(summary["record_events"], summary["remote_events"],
           summary["server_successes"]) == 0:
        raise ValueError("the numeric test made no Jev Choice request")
    if summary["remote_events"] != summary["server_successes"]:
        raise ValueError("remote client and API request counts differ")
    # These exact GTests run once per mode in fresh processes. Unlike the
    # timed benchmark, their scheduler-call multiplicity must match too.
    unequal = [entry for entry in report["multiplicity"]
               if entry["record"] != entry["remote"]]
    if unequal:
        sites = sorted({entry["site"] for entry in unequal})
        raise ValueError(f"record/remote decision multiplicity differs at {sites}")
    cases = report["case_coverage"]
    if set(cases) != {case_key}:
        raise ValueError(f"decision logs contain unexpected case keys: {sorted(cases)}")
    if min(cases[case_key].values()) == 0:
        raise ValueError("the case key has no record or remote events")
    sites = report["site_coverage"]
    if not any(site.startswith(prefix) and value["status"] == "covered"
               and value["server_events"] > 0 for site, value in sites.items()):
        raise ValueError(f"no matched {prefix} Choice was exercised")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--gtest-filter", required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--site-prefix", required=True)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    args = parser.parse_args(argv)
    names = args.gtest_filter.split(":")
    if not 1 <= len(names) <= 4 or len(set(names)) != len(names) or any(
        GTEST_NAME.fullmatch(name) is None for name in names
    ):
        parser.error("--gtest-filter must contain 1..4 distinct exact GTest names")
    if CASE_KEY.fullmatch(args.case_key) is None or SITE_PREFIX.fullmatch(args.site_prefix) is None:
        parser.error("invalid test case key or scheduler site prefix")
    if not args.binary.is_absolute() or not args.binary.is_file():
        parser.error("--binary must name an existing absolute test executable")
    if not args.artifacts_root.is_absolute():
        parser.error("--artifacts-root must be an absolute directory")
    module_dir, loader, libraries = checked_environment()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.case_key + "-", dir=args.artifacts_root))
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": "nvfuser-jev-remote-numeric-v1",
        "case_key": args.case_key, "gtests": names, "required_site_prefix": args.site_prefix,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "binary": str(args.binary), "binary_sha256": file_hash(args.binary),
        "loader": str(loader), "loader_sha256": file_hash(loader),
        "library_path": libraries,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "parity_modules": {name: file_hash(module_dir / name) for name in
                           ("jev_shim.py", "compare_decisions.py")},
        "runs": [], "status": "started",
    }
    write_manifest(manifest_path, manifest)
    print(f"Jev numeric artifacts: {directory}", flush=True)
    try:
        sys.path.insert(0, str(module_dir))
        DecisionServer = importlib.import_module("jev_shim").DecisionServer
        compare_decisions = importlib.import_module("compare_decisions").compare_decisions

        manifest["active_mode"] = "record"
        write_manifest(manifest_path, manifest)
        record = run_numeric(mode="record", binary=args.binary, names=names,
                             case_key=args.case_key, loader=loader, libraries=libraries,
                             directory=directory, port=None)
        manifest["runs"].append(record)
        manifest.pop("active_mode")
        write_manifest(manifest_path, manifest)

        server_log = directory / "server.jsonl"
        server_log.touch(exist_ok=False)
        server = DecisionServer(0, server_log)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05},
                        name="nvfuser-jev-numeric-api", daemon=True)
        thread.start()
        try:
            manifest["active_mode"] = "remote"
            write_manifest(manifest_path, manifest)
            remote = run_numeric(mode="remote", binary=args.binary, names=names,
                                 case_key=args.case_key, loader=loader, libraries=libraries,
                                 directory=directory, port=server.server_port)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        manifest["runs"].append(remote)
        manifest.pop("active_mode")
        write_manifest(manifest_path, manifest)
        if server_log.stat().st_size > MAX_ARTIFACT_BYTES:
            raise ValueError("API log exceeds the artifact bound")
        manifest["server_log"] = str(server_log)
        manifest["server_log_sha256"] = file_hash(server_log)

        report = compare_decisions(Path(record["decision_log"]),
                                   Path(remote["decision_log"]), server_log,
                                   expected_cases=(args.case_key,))
        report_path = directory / "decision_parity.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest["decision_parity"] = str(report_path)
        manifest["decision_parity_sha256"] = file_hash(report_path)
        check_parity(report, args.case_key, args.site_prefix)
        manifest["status"] = "numeric_and_decision_parity_pass"
        write_manifest(manifest_path, manifest)
        print(json.dumps({"numeric_tests_per_mode": len(names),
                          "record_decisions": report["summary"]["record_events"],
                          "remote_decisions": report["summary"]["remote_events"],
                          "server_successes": report["summary"]["server_successes"],
                          "covered_sites": sorted(site for site, item in report["site_coverage"].items()
                                                  if item["status"] == "covered")}, sort_keys=True))
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        print(f"Jev numeric failure: {manifest['failure']}; see {manifest_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
