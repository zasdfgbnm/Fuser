#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check that Remote benchmark validation is numerical and log-isolated.

The API server and benchmark child belong to this foreground test. The server
is closed through its own object after the child exits; no process is signaled.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from threading import Thread


CASES = {
    "greedy": (
        2596,
        "NvFuserScheduler_Greedy_FourScan_fp32___GRAPH/"
        "NvFuserScheduler_Greedy_FourScan_fp32/8/2048/manual_time",
    ),
    "rmsnorm": (
        8080,
        "NvFuserScheduler_RMSNorm_fp16___GRAPH/"
        "NvFuserScheduler_RMSNorm_fp16/2048/4096/manual_time",
    ),
    "softmax": (
        9987,
        "NvFuserScheduler_Softmax_Inner_fp32___GRAPH/"
        "NvFuserScheduler_Softmax_Inner_fp32/1/160/manual_time",
    ),
    "transpose": (
        10978,
        "NvFuserScheduler_Transpose_Random_fp32_Inner_2D_01_Axis___GRAPH/"
        "NvFuserScheduler_Transpose_Random_fp32_Inner_2D_01_Axis/9/2408",
    ),
}


class ChoiceServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ChoiceHandler)
        self.events: list[tuple[str, str]] = []


class ChoiceHandler(BaseHTTPRequestHandler):
    server: ChoiceServer

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _send(self, status: int, value: dict) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self._send(404, {"detail": "unknown endpoint"})
            return
        self._send(200, {"models": [{
            "name": "jev-latest", "description": "local numeric gate test",
            "release_date": "2026-10-28"}]})

    def do_POST(self) -> None:
        if self.path != "/v1/systemone":
            self._send(404, {"detail": "unknown endpoint"})
            return
        size = self.headers.get("Content-Length", "")
        if not size.isdecimal() or int(size) > 8 * 1024 * 1024:
            self._send(400, {"detail": "invalid length"})
            return
        request = json.loads(self.rfile.read(int(size)))
        questions = request["questions"]
        if len(questions) != 1:
            self._send(400, {"detail": "expected one question"})
            return
        site, question = next(iter(questions.items()))
        choices = list(question["criteria"])
        # Force one known, ATen-checked branch to differ from its native b2.
        answer = "b4" if site == "GR1.multi_scan_batch" and "b4" in choices else choices[0]
        self.server.events.append((site, answer))
        self._send(200, {
            "model": "nvfuser-analysis-v1",
            "answers": {site: {
                "type": "choice", "choice": answer, "confidence": 1.0,
                "probabilities": {key: float(key == answer) for key in choices}}},
            "usage": {"input_tokens": 0, "output_tokens": 0},
        })


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def remote_env(server: ChoiceServer, decision: Path, outcome: Path,
               cuda_root: Path, case_key: str | None = None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({
        "CUDA_HOME": str(cuda_root),
        "CUDA_PATH": str(cuda_root),
        "PATH": str(cuda_root / "bin") + ":/usr/bin:/bin",
        "NVFUSER_JEV_MODE": "remote",
        "NVFUSER_JEV_PROTOCOL": "semantic-v3",
        "NVFUSER_JEV_LOG": str(decision),
        "NVFUSER_JEV_OUTCOME_LOG": str(outcome),
        "NVFUSER_JEV_PORT": str(server.server_port),
        "NVFUSER_JEV_TIMEOUT_MS": "3600000",
    })
    if case_key is not None:
        env["NVFUSER_JEV_CASE_KEY"] = case_key
    return env


def run_case(binary: Path, loader: Path, libraries: str, cuda_root: Path, root: Path,
             label: str, enabled: bool, log_failure: bool = False) -> dict:
    if log_failure and not enabled:
        raise ValueError("a log failure requires the numerical gate")
    ordinal, name = CASES[label]
    case_key = f"{ordinal}:{name}"
    suffix = "log_failure" if log_failure else "checked" if enabled else "plain"
    stem = f"{label}.{suffix}"
    decision = root / f"{stem}.decision.jsonl"
    outcome = root / f"{stem}.outcome.jsonl"
    numeric = root / f"{stem}.numeric.jsonl"
    benchmark = root / f"{stem}.benchmark.json"
    stdout = root / f"{stem}.stdout.txt"
    stderr = root / f"{stem}.stderr.txt"
    decision.touch(exist_ok=False)
    with ChoiceServer() as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = remote_env(server, decision, outcome, cuda_root, case_key)
        if enabled:
            env["NVFUSER_JEV_VALIDATE_NATIVE"] = "1"
            env["NVFUSER_JEV_NUMERIC_LOG"] = str(root if log_failure else numeric)
        argv = [
            str(loader), "--library-path", str(binary.parent) + ":" + libraries,
            str(binary), "--benchmark_filter=^" + re.escape(name) + "$",
            "--benchmark_min_time=0", "--benchmark_repetitions=1",
            "--benchmark_out=" + str(benchmark), "--benchmark_out_format=json",
        ]
        try:
            with stdout.open("x") as out, stderr.open("x") as err:
                result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                        check=False)
        finally:
            server.shutdown()
            thread.join()
    if log_failure:
        if result.returncode == 0 or numeric.exists() or (
                "Could not append the benchmark numerical validation result"
                not in stderr.read_text()):
            raise AssertionError("missing sidecar did not fail the benchmark")
        return {"log_failure": "failed_closed"}
    if result.returncode != 0:
        raise AssertionError(
            f"{stem} exited {result.returncode}; see {stderr}")
    measured = json.loads(benchmark.read_text())["benchmarks"]
    if len(measured) != 1 or measured[0]["name"] != name:
        raise AssertionError(f"{stem} did not run exactly its registration")
    numeric_rows = rows(numeric) if enabled else []
    if enabled and (
            len(numeric_rows) != 1 or
            numeric_rows[0].get("schema") != "nvfuser.jev.benchmark_numeric.v1" or
            numeric_rows[0].get("case_key") != case_key or
            numeric_rows[0].get("mode") != "remote" or
            numeric_rows[0].get("path") != "fusion_executor_cache" or
            numeric_rows[0].get("status") != "passed" or
            numeric_rows[0].get("reference") != "native_off_same_inputs" or
            numeric_rows[0].get("output_count", 0) < 1):
        raise AssertionError(f"{stem} lacks a passing numeric row: {numeric_rows}")
    if not enabled and numeric.exists():
        raise AssertionError(f"{stem} unexpectedly created a numeric sidecar")
    outcome_rows = rows(outcome)
    return {
        "case_key": case_key,
        "choices": [(row["site"], row["selected"]) for row in rows(decision)
                    if row.get("kind") is None],
        "outcome_kinds": sorted(
            (row.get("schema"), row.get("kind")) for row in outcome_rows),
        "server_events": server.events,
        "numeric": numeric_rows,
    }


def run_batch_fallback(binary: Path, loader: Path, libraries: str,
                       cuda_root: Path, root: Path) -> dict:
    # The first registration computes heuristics without an FEC iteration
    # helper. The second is an ATen baseline and has no model parity to prove.
    cases = [
        (2599, "NvFuserScheduler_LayerNormBackward_HeuristicCache"),
        (10252, "Baseline_Softmax_Inner_fp32/1/160/manual_time"),
    ]
    manifest = root / "batch_fallback.manifest.tsv"
    manifest.write_text("".join(f"{ordinal}\t{name}\n" for ordinal, name in cases))
    decision = root / "batch_fallback.decision.jsonl"
    outcome = root / "batch_fallback.outcome.jsonl"
    numeric = root / "batch_fallback.numeric.jsonl"
    benchmark = root / "batch_fallback.benchmark.json"
    stdout = root / "batch_fallback.stdout.txt"
    stderr = root / "batch_fallback.stderr.txt"
    decision.touch(exist_ok=False)
    with ChoiceServer() as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = remote_env(server, decision, outcome, cuda_root)
        env.update({
            "NVFUSER_JEV_BATCH_MANIFEST": str(manifest),
            "NVFUSER_JEV_VALIDATE_NATIVE": "1",
            "NVFUSER_JEV_NUMERIC_LOG": str(numeric),
        })
        filt = "^(" + "|".join(re.escape(name) for _, name in cases) + ")$"
        argv = [
            str(loader), "--library-path", str(binary.parent) + ":" + libraries,
            str(binary), "--benchmark_filter=" + filt,
            "--benchmark_min_time=0", "--benchmark_repetitions=1",
            "--benchmark_enable_random_interleaving=false",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_display_aggregates_only=false",
            "--benchmark_out=" + str(benchmark), "--benchmark_out_format=json",
        ]
        try:
            with stdout.open("x") as out, stderr.open("x") as err:
                result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                        check=False)
        finally:
            server.shutdown()
            thread.join()
    if result.returncode != 0:
        raise AssertionError(f"batch fallback exited {result.returncode}; see {stderr}")
    measured = json.loads(benchmark.read_text())["benchmarks"]
    if [row["name"] for row in measured] != [name for _, name in cases]:
        raise AssertionError("batch fallback did not run both registrations")
    numeric_rows = rows(numeric)
    if (len(numeric_rows) != 1 or
            numeric_rows[0].get("case_key") != f"{cases[0][0]}:{cases[0][1]}" or
            numeric_rows[0].get("status") != "unsupported" or
            numeric_rows[0].get("path") != "benchmark_registration" or
            numeric_rows[0].get("reason") !=
            "benchmark_did_not_use_validation_helper"):
        raise AssertionError(f"batch fallback was not explicit: {numeric_rows}")
    return {"setup_only": "unsupported", "baseline_rows": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--artifacts-root", required=True, type=Path)
    args = parser.parse_args()
    if any(key not in os.environ for key in (
            "JEV_NUMERIC_LOADER", "JEV_NUMERIC_LIBRARY_PATH",
            "JEV_NUMERIC_CUDA_ROOT")):
        raise SystemExit(77)
    binary = args.binary.resolve(strict=True)
    loader = Path(os.environ["JEV_NUMERIC_LOADER"]).resolve(strict=True)
    libraries = os.environ["JEV_NUMERIC_LIBRARY_PATH"]
    cuda_root = Path(os.environ["JEV_NUMERIC_CUDA_ROOT"]).resolve(strict=True)
    if not (cuda_root / "include/cuda/std/type_traits").is_file():
        raise ValueError("numeric CUDA root lacks the required headers")
    if any(not Path(directory).is_absolute() or not Path(directory).is_dir()
           for directory in libraries.split(":")):
        raise ValueError("invalid numeric library path")
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="numeric-gate-", dir=args.artifacts_root))
    results = {}
    for label in CASES:
        plain = run_case(binary, loader, libraries, cuda_root, root, label, False)
        checked = run_case(binary, loader, libraries, cuda_root, root, label, True)
        if (plain["choices"] != checked["choices"] or
                plain["server_events"] != checked["server_events"] or
                plain["outcome_kinds"] != checked["outcome_kinds"]):
            raise AssertionError(
                f"{label} numerical preflight changed Remote Choice/outcome lineage")
        if label == "greedy" and ("GR1.multi_scan_batch", "b4") not in checked[
                "server_events"]:
            raise AssertionError("Greedy did not exercise the changed schedule")
        results[label] = {
            "choices": len(checked["choices"]),
            "numeric_status": checked["numeric"][0]["status"],
            "output_count": checked["numeric"][0]["output_count"],
        }
    results["missing_sidecar"] = run_case(
        binary, loader, libraries, cuda_root, root, "greedy", True,
        log_failure=True)["log_failure"]
    results["batch_fallback"] = run_batch_fallback(
        binary, loader, libraries, cuda_root, root)
    print(json.dumps({"artifacts": str(root), "cases": results}, sort_keys=True))


if __name__ == "__main__":
    main()
