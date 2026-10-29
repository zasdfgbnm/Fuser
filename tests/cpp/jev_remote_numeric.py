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
SITE_PREFIX = re.compile(r"(?:PW|RD|RS|TR|IP|OP|IOP)\Z")
REQUIRED_SITE = re.compile(r"[A-Z]{2,3}[0-9]+(?:\.[a-z][a-z0-9_]*)?\Z")
NO_CHOICE_EXPECTATION = re.compile(
    r"(?P<site>[A-Z]{2,3}[0-9]+(?:\.[a-z][a-z0-9_]*)?)="
    r"(?P<selected>[a-z][a-z0-9_]*)\Z")
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
            "non-skipped tests"
        )
    return {"running": running[0], "passed": passed[0]}


def run_numeric(
    *, mode: str, binary: Path, names: list[str], case_key: str,
    loader: Path, libraries: list[str], directory: Path, port: int | None,
    analysis_only: bool = False, protocol: str | None = None,
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
    if protocol is not None:
        if protocol not in {"text-v1", "semantic-v2", "semantic-v3"}:
            raise ValueError("unknown Jev numeric protocol")
        env["NVFUSER_JEV_PROTOCOL"] = protocol
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
    summary["analysis" if analysis_only else "numeric"] = check_numeric_output(
        output.read_text(errors="replace"), len(names))
    summary["artifact_sha256"] = {path.name: file_hash(path) for path in
                                  (output, errors, decision_log)}
    if completed.returncode != 0:
        raise ValueError(f"{mode} GTest exited {completed.returncode}; see {errors}")
    return summary


def check_parity(report: dict[str, Any], case_key: str, prefix: str,
                 required_site: str | None = None) -> None:
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
    # The comparison report also counts coverage skips. Zero skips is the
    # ordinary successful case, so inspect the two required event counts.
    if min(cases[case_key][key] for key in ("record_events", "remote_events")) == 0:
        raise ValueError("the case key has no record or remote events")
    sites = report["site_coverage"]
    if not any(site.startswith(prefix) and value["status"] == "covered"
               and value["server_events"] > 0 for site, value in sites.items()):
        raise ValueError(f"no matched {prefix} Choice was exercised")
    if required_site is not None and (
            required_site not in sites or
            sites[required_site]["status"] != "covered" or
            sites[required_site]["server_events"] == 0):
        raise ValueError(f"required Choice {required_site} was not covered")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"missing or oversized JSONL artifact: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def server_site(row: dict[str, Any]) -> str | None:
    request = row.get("request")
    questions = request.get("questions") if isinstance(request, dict) else None
    if isinstance(questions, dict) and len(questions) == 1:
        return next(iter(questions))
    return row.get("site") if isinstance(row.get("site"), str) else None


def check_no_choice(
    record: dict[str, Any], remote: dict[str, Any], server_log: Path,
    report: dict[str, Any], case_key: str, site: str, selected: str,
    reason: str | None = None, source: str | None = None,
    expected_count: int = 1,
) -> dict[str, Any]:
    # A structural singleton is a completed computeHeuristics outcome. It
    # must appear in both native outcome streams and nowhere on the Choice
    # wire, while the other sites still exercise the API through check_parity.
    site_coverage = report["site_coverage"].get(site, {})
    if any(site_coverage.get(key, 0) for key in
           ("record_events", "remote_events", "server_events", "record_skipped")):
        raise ValueError(f"{site} unexpectedly appeared on the Choice wire: {site_coverage}")
    evidence = {}
    for mode, run in (("record", record), ("remote", remote)):
        decision_log = Path(run["decision_log"])
        if any(row.get("site") == site for row in read_jsonl(decision_log)):
            raise ValueError(f"{mode} wrote a Choice or coverage skip at {site}")
        outcome_log = Path(str(decision_log) + ".outcome.jsonl")
        matching = [row for row in read_jsonl(outcome_log)
                    if row.get("site") == site]
        invocation_ids = [row.get("compute_invocation_id") for row in matching]
        invalid = any(
            row.get("schema") != "nvfuser.jev.compute_invocation.v1" or
            row.get("kind") != "no_choice" or
            row.get("mode") != mode or
            row.get("case_key") != case_key or
            row.get("selected") != selected or
            not row.get("reason") or not row.get("selection_source") or
            (reason is not None and row.get("reason") != reason) or
            (source is not None and row.get("selection_source") != source)
            for row in matching)
        if (len(matching) != expected_count or
            len(set(invocation_ids)) != expected_count or
            any(not isinstance(value, str) or not value
                for value in invocation_ids) or invalid):
            raise ValueError(
                f"expected {expected_count} distinct {mode} no_choice "
                f"invocations at {site}={selected}: {matching}")
        evidence[mode] = {"outcome_log": str(outcome_log),
                          "outcome_sha256": file_hash(outcome_log),
                          "no_choice": matching[0] if expected_count == 1
                          else matching}
    if any(server_site(row) == site for row in read_jsonl(server_log)):
        raise ValueError(f"API server received a singleton Choice at {site}")
    return evidence


def check_prior_choice(
    record: dict[str, Any], remote: dict[str, Any], server_log: Path,
    request_site: str, prior_site: str, selected: str,
    expected_count: int = 1,
) -> dict[str, Any]:
    expected = {"site": prior_site, "choice": selected}
    evidence = {}
    for mode, run in (("record", record), ("remote", remote)):
        rows = [row for row in read_jsonl(Path(run["decision_log"]))
                if row.get("site") == request_site]
        outcome_log = Path(str(run["decision_log"]) + ".outcome.jsonl")
        fixed = [row for row in read_jsonl(outcome_log)
                 if row.get("site") == prior_site and row.get("kind") == "no_choice"]
        if (len(rows) != expected_count or len(fixed) != expected_count or
                {row.get("compute_invocation_id") for row in rows} !=
                {row.get("compute_invocation_id") for row in fixed}):
            raise ValueError(
                f"expected {expected_count} paired {mode} {request_site} "
                f"Choice and {prior_site} no_choice invocations")
        priors = [row.get("state", {}).get("current_schedule", {}).get(
            "prior_choices") for row in rows]
        if any(not isinstance(prior, list) or prior.count(expected) != 1
               for prior in priors):
            raise ValueError(f"{mode} {request_site} lacks prior {expected}: {priors}")
        evidence[mode] = priors[0] if expected_count == 1 else priors
    rows = [row for row in read_jsonl(server_log)
            if server_site(row) == request_site]
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} API {request_site} requests")
    priors = [row.get("request", {}).get("state", {}).get(
        "current_schedule", {}).get("prior_choices") for row in rows]
    if any(not isinstance(prior, list) or prior.count(expected) != 1
           for prior in priors):
        raise ValueError(f"API {request_site} lacks prior {expected}: {priors}")
    evidence["server"] = priors[0] if expected_count == 1 else priors
    return evidence


IP6_LOCAL_PREVIEW_SITES = (
    "IP6.width", "IP6.waves", "IP6.occupancy", "IP6.warp",
    "IP6.outer_unroll",
)


def check_ip6_local_previews(record: dict[str, Any], remote: dict[str, Any],
                             server_log: Path) -> dict[str, Any]:
    """Check source plan effects survive Record, HTTP, and numeric Remote.

    The original plan tuples remain in the raw analysis for the mock server.
    They cross-check the typed effects at sites that already emit them; wave
    and warp sites are checked against their descriptions and hardware.
    """
    evidence = {}
    for mode, path in (("record", Path(record["decision_log"])),
                       ("remote", Path(remote["decision_log"])),
                       ("server", server_log)):
        rows = read_jsonl(path)
        matching = {site: [row for row in rows if server_site(row) == site]
                    for site in IP6_LOCAL_PREVIEW_SITES}
        counts = {site: len(found) for site, found in matching.items()}
        if any(count != 1 for count in counts.values()):
            raise ValueError(f"{mode} did not expose all five IP6 previews: "
                             f"{counts}")
        for site, found in matching.items():
            request = found[0]["request"]
            state = request["state"]
            if state["schema"] != "nvfuser.systemone.context.v3":
                raise ValueError(f"{mode} {site} lost its semantic-v3 context")
            decision = state["decision"]
            if decision["site"] != site:
                raise ValueError(f"{mode} {site} disagrees with question site")
            candidates = decision["candidates"]
            if not candidates or [item["id"] for item in candidates] != sorted(
                    request["questions"][site]["criteria"]):
                raise ValueError(f"{mode} {site} candidate menu changed")
            plan_rows = {}
            for line in decision["analysis_inputs_text"].splitlines()[1:]:
                key, _, value = line.partition("=")
                if key.startswith("plan"):
                    plan_rows[key] = tuple(int(part) for part in value.split(","))
            hardware = state["workload"]["hardware"]
            for candidate in candidates:
                effect = candidate.get("consequence")
                if not isinstance(effect, dict) or set(effect) != {
                        "schema", "scope", "inner", "outer", "block",
                        "iteration_grid_blocks", "nvrtc_maxrregcount",
                        "planner_work"} or effect["schema"] != (
                            "nvfuser.inner_persistent.3d_local_plan.v1") or (
                            effect["scope"] !=
                            "local_preview_before_later_choices"):
                    raise ValueError(f"{mode} {site} lacks a typed local plan")
                inner, outer, block = (effect[name] for name in
                                       ("inner", "outer", "block"))
                work = effect["planner_work"]
                if (set(inner) != {"batches_per_block", "unroll", "vectorized"}
                        or set(outer) != {"batches_per_block", "unroll"}
                        or set(block) != {"padded_x", "y", "z", "threads"}
                        or set(work) != {"target_unroll", "target_iterations"}
                        or type(inner["vectorized"]) is not bool):
                    raise ValueError(f"{mode} {site} has an incomplete plan")
                measured = (inner["batches_per_block"], inner["unroll"],
                            outer["batches_per_block"], outer["unroll"],
                            block["padded_x"], block["y"], block["z"],
                            block["threads"], effect["nvrtc_maxrregcount"],
                            effect["iteration_grid_blocks"])
                if (any(type(value) is not int or value <= 0 for value in
                        (*measured, *work.values())) or
                        block["threads"] !=
                        block["padded_x"] * block["y"] * block["z"] or
                        block["threads"] > hardware["max_threads_per_block"] or
                        effect["nvrtc_maxrregcount"] * block["threads"] >
                        hardware["registers_per_block"]):
                    raise ValueError(f"{mode} {site} plan violates basic resources")
                prose = candidate["description"]
                if (f"completed CTA={block['threads']}, register cap="
                        f"{effect['nvrtc_maxrregcount']}, grid blocks="
                        f"{effect['iteration_grid_blocks']}" not in prose):
                    raise ValueError(f"{mode} {site} plan and prose disagree")
                if site in {"IP6.width", "IP6.occupancy", "IP6.outer_unroll"}:
                    key = "plan" + re.search(r"[0-9]+$", candidate["id"])[0]
                    if key not in plan_rows or plan_rows[key][:10] != measured:
                        raise ValueError(f"{mode} {site} typed/raw plan mismatch")
                effect_text = json.dumps(effect).lower()
                if "native" in effect_text or "baseline" in effect_text:
                    raise ValueError(f"{mode} {site} leaked a preferred plan")
        evidence[mode] = {site: len(found[0]["request"]["state"]
                                      ["decision"]["candidates"])
                          for site, found in matching.items()}
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--gtest-filter", required=True)
    parser.add_argument("--case-key", required=True)
    parser.add_argument("--site-prefix", required=True)
    parser.add_argument("--required-site")
    parser.add_argument("--expect-no-choice", metavar="SITE=SELECTED")
    parser.add_argument("--expect-no-choice-count", type=int, default=1)
    parser.add_argument("--expect-no-choice-reason")
    parser.add_argument("--expect-no-choice-source")
    parser.add_argument("--expect-prior-in", metavar="DOWNSTREAM_SITE")
    parser.add_argument("--protocol", choices=("text-v1", "semantic-v2", "semantic-v3"))
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--check-ip6-local-previews", action="store_true")
    parser.add_argument("--artifacts-root", required=True, type=Path)
    args = parser.parse_args(argv)
    names = args.gtest_filter.split(":")
    if not 1 <= len(names) <= 4 or len(set(names)) != len(names) or any(
        GTEST_NAME.fullmatch(name) is None for name in names
    ):
        parser.error("--gtest-filter must contain 1..4 distinct exact GTest names")
    if CASE_KEY.fullmatch(args.case_key) is None or SITE_PREFIX.fullmatch(args.site_prefix) is None:
        parser.error("invalid test case key or scheduler site prefix")
    if args.required_site is not None and (
            REQUIRED_SITE.fullmatch(args.required_site) is None or
            not args.required_site.startswith(args.site_prefix)):
        parser.error("invalid required Choice site")
    no_choice_match = (NO_CHOICE_EXPECTATION.fullmatch(args.expect_no_choice)
                       if args.expect_no_choice is not None else None)
    if args.expect_no_choice is not None and (
            no_choice_match is None or
            not no_choice_match["site"].startswith(args.site_prefix) or
            no_choice_match["site"] == args.required_site):
        parser.error("invalid no-choice site or selected ID")
    if args.expect_no_choice is None and (
            args.expect_no_choice_count != 1 or
            args.expect_no_choice_reason is not None or
            args.expect_no_choice_source is not None or
            args.expect_prior_in is not None):
        parser.error("no-choice reason/source/prior requires --expect-no-choice")
    if not 1 <= args.expect_no_choice_count <= 16:
        parser.error("--expect-no-choice-count must be in 1..16")
    if args.expect_prior_in is not None and (
            REQUIRED_SITE.fullmatch(args.expect_prior_in) is None or
            not args.expect_prior_in.startswith(args.site_prefix) or
            args.expect_prior_in == no_choice_match["site"]):
        parser.error("invalid downstream prior-choice site")
    if args.expect_prior_in is not None and args.protocol != "semantic-v3":
        parser.error("--expect-prior-in requires --protocol semantic-v3")
    if args.check_ip6_local_previews and args.protocol != "semantic-v3":
        parser.error("--check-ip6-local-previews requires --protocol semantic-v3")
    if not args.binary.is_absolute() or not args.binary.is_file():
        parser.error("--binary must name an existing absolute test executable")
    if not args.artifacts_root.is_absolute():
        parser.error("--artifacts-root must be an absolute directory")
    module_dir, loader, libraries = checked_environment()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.case_key + "-", dir=args.artifacts_root))
    manifest_path = directory / "manifest.json"
    manifest: dict[str, Any] = {
        "schema": ("nvfuser-jev-remote-analysis-v1" if args.analysis_only else
                   "nvfuser-jev-remote-numeric-v1"),
        "case_key": args.case_key, "gtests": names,
        "required_site_prefix": args.site_prefix,
        "required_site": args.required_site,
        "expected_no_choice": args.expect_no_choice,
        "expected_no_choice_count": args.expect_no_choice_count,
        "expected_no_choice_reason": args.expect_no_choice_reason,
        "expected_no_choice_source": args.expect_no_choice_source,
        "expected_prior_in": args.expect_prior_in,
        "protocol": args.protocol,
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
                             directory=directory, port=None,
                             analysis_only=args.analysis_only,
                             protocol=args.protocol)
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
                                 directory=directory, port=server.server_port,
                                 analysis_only=args.analysis_only,
                                 protocol=args.protocol)
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
        check_parity(report, args.case_key, args.site_prefix,
                     args.required_site)
        if no_choice_match is not None:
            manifest["no_choice_evidence"] = check_no_choice(
                record, remote, server_log, report, args.case_key,
                no_choice_match["site"], no_choice_match["selected"],
                args.expect_no_choice_reason, args.expect_no_choice_source,
                args.expect_no_choice_count)
            if args.expect_prior_in is not None:
                manifest["prior_choice_evidence"] = check_prior_choice(
                    record, remote, server_log, args.expect_prior_in,
                    no_choice_match["site"], no_choice_match["selected"],
                    args.expect_no_choice_count)
        if args.check_ip6_local_previews:
            manifest["ip6_local_preview_evidence"] = check_ip6_local_previews(
                record, remote, server_log)
        manifest["status"] = ("analysis_and_decision_parity_pass" if args.analysis_only
                              else "numeric_and_decision_parity_pass")
        write_manifest(manifest_path, manifest)
        print(json.dumps({("analysis_tests_per_mode" if args.analysis_only else
                           "numeric_tests_per_mode"): len(names),
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
