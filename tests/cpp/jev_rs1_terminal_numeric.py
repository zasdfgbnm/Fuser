#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise c19's terminal Resize CTA previews with a bounded local server.

The server reuses the strict workspace evaluator for upstream RS1 decisions.
For the new CTA request it checks the exact source-produced c19 consequences
and forces one legal non-native CTA. The independent v2 service parser is a
separate workspace change; this test proves the producer and numeric path.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib
import json
from pathlib import Path
import re
import sys
import tempfile
from threading import Thread
import traceback

from jev_remote_numeric import (checked_environment, file_hash, run_numeric,
                                write_manifest)


SITES = ["RS1.width", "RS1.reorder", "RS1.cta"]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_cta(request: dict) -> None:
    state = request["state"]
    assert state["schema"] == "nvfuser.systemone.context.v3"
    snapshot = state["workload"]["fusion_and_runtime"]
    assert snapshot.startswith("nvfuser.compute_invocation_input.v2\n")
    assert "pad( T0" in snapshot
    assert "sizes 3 5 7 64 strides 3 64 320 1" in snapshot
    match = re.search(r"(?m)^index_type (\d+)$", snapshot)
    assert match is not None
    index_type = int(match.group(1))
    assert state["current_schedule"]["prior_choices"] == [
        {"site": "RS1.width", "choice": "w4"},
        {"site": "RS1.reorder", "choice": "keep"},
    ]
    decision = state["decision"]
    assert decision["site"] == "RS1.cta"
    lines = decision["analysis_inputs_text"].splitlines()
    assert lines[0] == "resize.cta.v2"
    fields = dict(line.split("=", 1) for line in lines[1:])
    assert list(fields) == [
        "max_output_elements", "layout_max", "vector_width", "warp_size",
        "max_threads_x", "max_threads_per_block", "max_grid_x",
    ]
    assert [int(fields[key]) for key in
            ("max_output_elements", "layout_max", "vector_width")] == [2520, 4, 4]
    hardware = state["workload"]["hardware"]
    for key in ("warp_size", "max_threads_x", "max_threads_per_block",
                "max_grid_x"):
        assert int(fields[key]) == hardware[key]

    candidates = decision["candidates"]
    assert [candidate["id"] for candidate in candidates] == [
        "t128", "t256", "t64"]
    for candidate in candidates:
        threads = int(candidate["id"][1:])
        assert candidate["description"] == (
            f"Use {threads} threads per Resize CTA")
        assert candidate["consequence"] == {
            "final_resize_plan": {
                "index_type": index_type,
                "largest_input": 0,
                "vectorization_factor": 4,
                "cta_threads": threads,
                "split_grid_x_dim": False,
            },
            "index_type_provenance": "source_hard_analysis",
            "reference_tv": "T1",
            "input_tv": "T0",
            "loop_axis_order": [1, 0, 2],
            "read_stride_by_loop": [320, 64, 1],
            "write_stride_by_loop": [72, 504, 1],
            "read_elements": 2240,
            "write_elements": 2520,
            "pad_fill_elements": 280,
            "output_vector_iterations": 630,
            "block_x": threads,
            "grid_x": (630 + threads - 1) // threads,
            "grid_y": 1,
            "pending_choices": [],
        }


def make_handler(evaluator, target: str, events: list[dict], errors: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                assert self.path == "/v1/systemone"
                length = int(self.headers["Content-Length"])
                assert 0 < length < 8 * 1024 * 1024
                request = json.loads(self.rfile.read(length))
                site = request["state"]["decision"]["site"]
                if site == "RS1.cta":
                    validate_cta(request)
                    ids = [candidate["id"] for candidate in
                           request["state"]["decision"]["candidates"]]
                    response = {
                        "model": "nvfuser-analysis-v1",
                        "answers": {site: {
                            "type": "choice", "choice": target,
                            "confidence": 1.0,
                            "probabilities": {
                                choice: float(choice == target) for choice in ids},
                        }},
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                else:
                    assert site in SITES[:2]
                    response, _ = evaluator(request)
                events.append({"site": site, "request": request,
                               "response": response})
                body = json.dumps(response, separators=(",", ":")).encode()
                self.send_response(200)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                body = json.dumps({"error": errors[-1]}).encode()
                self.send_response(422)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, choices=(64, 256), required=True)
    args = parser.parse_args()
    assert args.binary.is_absolute() and args.binary.is_file()
    assert args.artifacts_root.is_absolute()
    module_dir, loader, libraries = checked_environment()
    sys.path.insert(0, str(module_dir))
    evaluator = importlib.import_module("jev_shim").evaluate
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    case_key = f"jev_rs_one_terminal_t{args.threads}"
    directory = Path(tempfile.mkdtemp(prefix=case_key + "-",
                                      dir=args.artifacts_root))
    manifest_path = directory / "manifest.json"
    manifest = {"schema": "nvfuser-jev-rs1-terminal-source-test-v1",
                "case_key": case_key, "binary": str(args.binary),
                "binary_sha256": file_hash(args.binary),
                "target_threads": args.threads, "runs": [], "status": "started"}
    write_manifest(manifest_path, manifest)
    print(f"RS1 terminal numeric artifacts: {directory}", flush=True)
    try:
        record = run_numeric(
            mode="record", binary=args.binary,
            names=["ResizeTest.JevDirectPadTerminalCta_CUDA"],
            case_key=case_key, loader=loader, libraries=libraries,
            directory=directory, port=None, protocol="semantic-v3")
        manifest["runs"].append(record)
        events: list[dict] = []
        errors: list[str] = []
        server = HTTPServer(("127.0.0.1", 0),
                            make_handler(evaluator, f"t{args.threads}",
                                         events, errors))
        thread = Thread(target=server.serve_forever,
                        kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            remote = run_numeric(
                mode="remote", binary=args.binary,
                names=["ResizeTest.JevDirectPadTerminalCta_CUDA"],
                case_key=case_key, loader=loader, libraries=libraries,
                directory=directory, port=server.server_port,
                protocol="semantic-v3")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
        manifest["runs"].append(remote)
        server_log = directory / "server.jsonl"
        server_log.write_text("".join(json.dumps(event, sort_keys=True) + "\n"
                                      for event in events))
        assert not errors, errors
        assert [event["site"] for event in events] == SITES
        record_rows = rows(Path(record["decision_log"]))
        remote_rows = rows(Path(remote["decision_log"]))
        assert [row["site"] for row in record_rows] == SITES
        assert [row["site"] for row in remote_rows] == SITES
        assert [row["selected"] for row in record_rows] == [
            "w4", "keep", "t128"]
        assert [row["selected"] for row in remote_rows] == [
            "w4", "keep", f"t{args.threads}"]
        for row in (record_rows[-1], remote_rows[-1]):
            validate_cta(row["request"])
        assert record_rows[-1]["request"] == remote_rows[-1]["request"]
        if args.threads == 64:
            # A symbolic rank-two Pad is outside the direct rank-three proof.
            # Record must run its native CTA and log both unsupported sites.
            unsupported_dir = directory / "unsupported"
            unsupported_dir.mkdir()
            unsupported = run_numeric(
                mode="record", binary=args.binary,
                names=["ResizeTest.JevAlternateCta_CUDA"],
                case_key=case_key + "_unsupported", loader=loader,
                libraries=libraries, directory=unsupported_dir, port=None,
                protocol="semantic-v3")
            unsupported_rows = rows(Path(unsupported["decision_log"]))
            assert [row["site"] for row in unsupported_rows] == SITES
            assert unsupported_rows[0]["selected"] == "w4"
            assert all(row["kind"] == "coverage_skip" and
                       "request" not in row for row in unsupported_rows[1:])
            manifest["unsupported_record"] = unsupported
        manifest["server_log"] = str(server_log)
        manifest["server_log_sha256"] = file_hash(server_log)
        manifest["status"] = "numeric_and_request_preview_pass"
        write_manifest(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        manifest["status"] = "failed"
        write_manifest(manifest_path, manifest)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
