#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Numerically check one non-native scheduler Choice through the local API.

CTest invokes this under the governed resource wrapper. The in-process server
first runs the unmodified parity evaluator, including its full request and
candidate validation. The selected test then changes one named answer to a
legal option.
The production analysis server and scheduler are never patched on disk.
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
import sys
import tempfile
from threading import Lock, Thread
from typing import Any, Callable

from jev_remote_numeric import (checked_environment, file_hash, run_numeric,
                                write_manifest)


SITE = "RD3.grid"
ALTERNATE = "g1"
GTEST = "Gpu1Test.JevAlternateInnerGrid_CUDA"
CASE_KEY = "jev_alternate_numeric_inner_grid"
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


def request_hash(request: dict[str, Any]) -> str:
    wire = json.dumps(request, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)
    return hashlib.sha256(wire.encode("utf-8")).hexdigest()


class AlternateChoice:
    """Wrap a strict evaluator and record every target request exactly."""

    def __init__(self, evaluator: Callable[[Any], tuple[dict[str, Any], Any]],
                 site: str, alternate: str):
        self.evaluator = evaluator
        self.site = site
        self.alternate = alternate
        self.lock = Lock()
        self.events: list[dict[str, Any]] = []

    def __call__(self, request: Any) -> tuple[dict[str, Any], Any]:
        # Keep validation before override; malformed requests cannot acquire
        # an alternate answer, and every other site remains deterministic.
        response, analysis = self.evaluator(request)
        if self.site not in request["questions"]:
            return response, analysis

        candidates = list(request["questions"][self.site]["criteria"])
        native = response["answers"][self.site]["choice"]
        eligible = self.alternate in candidates and native != self.alternate
        event = {
            "site": self.site, "request_sha256": request_hash(request),
            "state": request["state"], "candidates": candidates,
            "native": native, "requested_alternate": self.alternate,
            "forced": eligible,
        }
        with self.lock:
            self.events.append(event)
        if not eligible:
            return response, analysis

        changed = dict(response)
        changed["answers"] = dict(response["answers"])
        answer = dict(changed["answers"][self.site])
        answer["choice"] = self.alternate
        answer["probabilities"] = {
            candidate: float(candidate == self.alternate)
            for candidate in candidates
        }
        answer["confidence"] = 1.0
        changed["answers"][self.site] = answer
        return changed, analysis


def check_alternate_report(report: dict[str, Any],
                           events: list[dict[str, Any]], *, site: str,
                           alternate: str, case_key: str,
                           require_grid: bool = False,
                           require_inner_targets: bool = False) -> dict[str, Any]:
    if len(events) != 1 or not events[0]["forced"]:
        raise ValueError(f"expected exactly one eligible {site} override; got {events}")
    event = events[0]
    target = [entry for entry in report["multiplicity"]
              if entry["site"] == site]
    if len(target) != 1:
        raise ValueError(f"expected exactly one shared {site} request, got {len(target)}")
    entry = target[0]
    if (entry["request_sha256"] != event["request_sha256"] or
            entry["record"] != {event["native"]: 1} or
            entry["remote"] != {alternate: 1}):
        raise ValueError(f"{site} record/remote requests or winners differ: {entry}")
    issues = [{"kind": problem["kind"], "site": problem.get("site"),
               "request_sha256": problem.get("request_sha256")}
              for problem in report["issues"]]
    intended = {"kind": "winner_mismatch", "site": site,
                "request_sha256": event["request_sha256"]}
    if intended not in issues:
        raise ValueError(f"missing intended winner mismatch: {issues}")
    if site == "OP3.unroll":
        # The selected iteration factor changes the valid CTA menu, so the
        # downstream OP3.cta request must be replaced rather than compared
        # under the old state hash.
        ctas = [entry for entry in report["multiplicity"]
                if entry["site"] == "OP3.cta"]
        record_cta = [entry for entry in ctas
                      if sum(entry["record"].values()) == 1 and not entry["remote"]]
        remote_cta = [entry for entry in ctas
                      if sum(entry["remote"].values()) == 1 and not entry["record"]]
        if len(ctas) != 2 or len(record_cta) != 1 or len(remote_cta) != 1:
            raise ValueError(f"expected exactly one replaced OP3.cta request: {ctas}")
        if (not record_cta[0]["request"]["state"].startswith(
                "outer_persistent.block_cta.v1\n") or
                not remote_cta[0]["request"]["state"].startswith(
                    "outer_persistent.block_cta.v2\n")):
            raise ValueError("OP3.cta state versions did not follow factor choice")
        selected = dict(line.split("=", 1) for line in
                        remote_cta[0]["request"]["state"].splitlines()[1:])
        if (selected.get("iter_unroll") != alternate[1:] or
                int(selected.get("vector_width", "0")) < int(alternate[1:])):
            raise ValueError(f"OP3.cta did not receive selected {alternate}: {selected}")
        expected = [intended,
            {"kind": "record_only_state", "site": "OP3.cta",
             "request_sha256": record_cta[0]["request_sha256"]},
            {"kind": "remote_only_state", "site": "OP3.cta",
             "request_sha256": remote_cta[0]["request_sha256"]},
        ]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected dependent CTA comparison issues: {issues}")
    elif site == "OP2.base":
        # The alternate base must produce one different dependent tail state
        # on each run. The comparator has already checked each remote tail
        # answer against the API server; require the expected one-for-one
        # replacement and exact base coordinates rather than hiding it.
        tails = [entry for entry in report["multiplicity"]
                 if entry["site"] == "OP2.tail"]
        record_tail = [entry for entry in tails
                       if sum(entry["record"].values()) == 1 and not entry["remote"]]
        remote_tail = [entry for entry in tails
                       if sum(entry["remote"].values()) == 1 and not entry["record"]]
        if len(tails) != 2 or len(record_tail) != 1 or len(remote_tail) != 1:
            raise ValueError(f"expected exactly one replaced OP2.tail request: {tails}")
        for entry, base_id in ((record_tail[0], event["native"]),
                               (remote_tail[0], alternate)):
            match = re.fullmatch(r"x([0-9]+)_y([0-9]+)_b([0-9]+)", base_id)
            if match is None:
                raise ValueError(f"invalid OP2.base Choice ID: {base_id}")
            x, y, bx = map(int, match.groups())
            state = dict(line.split("=", 1) for line in
                         entry["request"]["state"].splitlines()[1:])
            actual = tuple(int(state[key]) for key in
                           ("base_gdimx", "base_gdimy", "base_bdimx",
                            "base_bdimy"))
            if actual != (x, y, bx, 256 // bx):
                raise ValueError(f"OP2.tail does not follow {base_id}: {actual}")
        expected = [intended,
            {"kind": "record_only_state", "site": "OP2.tail",
             "request_sha256": record_tail[0]["request_sha256"]},
            {"kind": "remote_only_state", "site": "OP2.tail",
             "request_sha256": remote_tail[0]["request_sha256"]},
        ]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected decision comparison issues: {issues}")
    elif require_grid:
        if set(event["candidates"]) != {"warp16", "warp32"}:
            raise ValueError("outer warp test did not reach both hard-valid plans")
        allowed = {site, "RD9.stride", "RD8"}
        if any(issue["site"] not in allowed for issue in issues):
            raise ValueError(f"unrelated decision comparison issues: {issues}")
        rd8 = [row for row in report["multiplicity"] if row["site"] == "RD8"]
        if sum(row["remote"].get("grid", 0) for row in rd8) != 1:
            raise ValueError(f"alternate warp did not reach the grid plan: {rd8}")
        if not any(issue["site"] in {"RD9.stride", "RD8"} for issue in issues):
            raise ValueError("alternate warp did not alter downstream grid analysis")
    elif require_inner_targets:
        if (site != "IP5.targets" or event["native"] != "o16_w28" or
                alternate != "o32_w16" or
                set(event["candidates"]) != {
                    "o16_w16", "o16_w28", "o32_w16", "o32_w28"}):
            raise ValueError("inner target fixture lacks its expected four-policy menu")
        expected_issues = sorted((
            ("winner_mismatch", "IP5.targets"),
            ("record_only_state", "IP5.batch"),
            ("remote_only_state", "IP5.batch"),
        ))
        if sorted((issue["kind"], issue["site"]) for issue in issues) != expected_issues:
            raise ValueError(f"unexpected inner target comparison issues: {issues}")
        nested = [row for row in report["multiplicity"]
                  if row["site"] == "IP5.batch"]
        if len(nested) != 2:
            raise ValueError(f"expected one record and one remote IP5.batch state: {nested}")
        for row in nested:
            record_count = sum(row["record"].values())
            remote_count = sum(row["remote"].values())
            if (record_count, remote_count) not in {(1, 0), (0, 1)}:
                raise ValueError(f"duplicate or shared nested batch request: {row}")
            state = row["request"]["state"]
            expected_overhead, expected_warps = ((16, 28) if record_count
                                                   else (32, 16))
            if (not state.startswith("inner_persistent.reg_batch.v2\n") or
                    f"\npolicy_register_overhead={expected_overhead}\n" not in state or
                    f"\npolicy_target_warps_per_sm={expected_warps}\n" not in state):
                raise ValueError(f"nested batch did not use selected target policy: {row}")
    elif issues != [intended]:
        raise ValueError(f"unexpected decision comparison issues: {issues}")
    if report["summary"]["remote_events"] != report["summary"]["server_successes"]:
        raise ValueError("remote and server request counts differ")
    if set(report["case_coverage"]) != {case_key}:
        raise ValueError("decision logs contain unexpected case keys")
    return {"site": site, "native": event["native"], "alternate": alternate,
            "candidates": event["candidates"],
            "request_sha256": event["request_sha256"],
            "record_events": report["summary"]["record_events"],
            "remote_events": report["summary"]["remote_events"],
            "server_successes": report["summary"]["server_successes"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    parser.add_argument("--site", default=SITE)
    parser.add_argument("--alternate", default=ALTERNATE)
    parser.add_argument("--gtest", default=GTEST)
    parser.add_argument("--case-key", default=CASE_KEY)
    parser.add_argument("--require-grid", action="store_true")
    parser.add_argument("--require-inner-targets", action="store_true")
    args = parser.parse_args(argv)
    if not args.binary.is_absolute() or not args.binary.is_file():
        parser.error("--binary must name an existing absolute test executable")
    if not args.artifacts_root.is_absolute():
        parser.error("--artifacts-root must be an absolute directory")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]*", args.site):
        parser.error("--site must be a single decision name")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.alternate):
        parser.error("--alternate must be a single Choice identifier")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", args.gtest):
        parser.error("--gtest must be one exact GTest name")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.case_key):
        parser.error("--case-key must use lowercase letters, digits, underscores")
    module_dir, loader, libraries = checked_environment()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.case_key + "-", dir=args.artifacts_root))
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": "nvfuser-jev-alternate-numeric-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_key": args.case_key, "gtest": args.gtest,
        "target_site": args.site, "requested_alternate": args.alternate,
        "binary": str(args.binary), "binary_sha256": file_hash(args.binary),
        "loader": str(loader), "loader_sha256": file_hash(loader),
        "library_path": libraries,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "modules": {name: file_hash(module_dir / name) for name in
                    ("jev_shim.py", "compare_decisions.py")},
        "harness_sha256": file_hash(Path(__file__)),
        "runs": [], "status": "started",
    }
    write_manifest(manifest_path, manifest)
    print(f"Jev alternate numeric artifacts: {directory}", flush=True)
    try:
        sys.path.insert(0, str(module_dir))
        shim = importlib.import_module("jev_shim")
        compare = importlib.import_module("compare_decisions").compare_decisions

        manifest["active_mode"] = "record"
        write_manifest(manifest_path, manifest)
        record = run_numeric(
            mode="record", binary=args.binary, names=[args.gtest],
            case_key=args.case_key, loader=loader, libraries=libraries,
            directory=directory, port=None)
        manifest["runs"].append(record)
        manifest["active_mode"] = "remote"
        write_manifest(manifest_path, manifest)

        server_log = directory / "server.jsonl"
        server_log.touch(exist_ok=False)
        original = shim.evaluate
        override = AlternateChoice(original, args.site, args.alternate)
        shim.evaluate = override
        try:
            server = shim.DecisionServer(0, server_log)
            thread = Thread(target=server.serve_forever,
                            kwargs={"poll_interval": 0.05},
                            name="nvfuser-jev-alternate-api", daemon=True)
            thread.start()
            try:
                remote = run_numeric(
                    mode="remote", binary=args.binary, names=[args.gtest],
                    case_key=args.case_key, loader=loader, libraries=libraries,
                    directory=directory, port=server.server_port)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()
        finally:
            shim.evaluate = original
            # Persist the override trace even if the alternate kernel fails
            # compilation, execution, or its numerical reference check.
            forced_path = directory / "forced_choice.json"
            forced_path.write_text(
                json.dumps(override.events, indent=2, sort_keys=True) + "\n")
            manifest["forced_choice_log"] = str(forced_path)
            manifest["forced_choice_log_sha256"] = file_hash(forced_path)
            manifest["server_log"] = str(server_log)
            manifest["server_log_sha256"] = file_hash(server_log)
            write_manifest(manifest_path, manifest)
        manifest["runs"].append(remote)
        manifest.pop("active_mode")
        for path in (server_log, forced_path):
            if path.stat().st_size > MAX_ARTIFACT_BYTES:
                raise ValueError(f"{path} exceeds the artifact bound")
        write_manifest(manifest_path, manifest)

        report = compare(Path(record["decision_log"]),
                         Path(remote["decision_log"]), server_log,
                         expected_cases=(args.case_key,), required_sites=(args.site,))
        report_path = directory / "decision_comparison.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        manifest["decision_comparison"] = str(report_path)
        manifest["decision_comparison_sha256"] = file_hash(report_path)
        summary = check_alternate_report(
            report, override.events, site=args.site,
            alternate=args.alternate, case_key=args.case_key,
            require_grid=args.require_grid,
            require_inner_targets=args.require_inner_targets)
        manifest["alternate"] = summary
        manifest["status"] = "alternate_numeric_pass"
        write_manifest(manifest_path, manifest)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        print(f"Jev alternate numeric failure: {manifest['failure']}; see {manifest_path}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
