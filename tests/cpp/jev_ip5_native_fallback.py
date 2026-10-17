#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Check the two softmax shapes whose native IP5 menu fails preflight.

Off, Record, and Remote must finish with the same typed scheduler plan. Remote
has no API server, so an accidental Choice fails. Record keeps the coverage
skips which identify the unproved candidate menu. Each mode also checks the
same shape against ATen. The caller owns this numeric workload and its
processes; this script sends no signals or timeouts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


CASES = (
    (9993, 160),
    (10001, 320),
)
SCHEMA = "nvfuser.jev.compute_invocation.v1"
PLAN_FIELDS = ("scheduler_type", "params_kind", "scheduler_fields", "launch",
               "compile", "tag")
COVERAGE = (
    ("IP5.width", "Native width has no proved 2D register plan"),
    ("IP5.targets", "Native 2D register policy lacks a proved batch menu"),
    ("IP5.cta_seed", "Native 2D register policy lacks a proved batch menu"),
)
GTEST_NAMES = {
    160: "Gpu1Test.JevIP5NativeFallbackSoftmax160_CUDA",
    320: "Gpu1Test.JevIP5NativeFallbackSoftmax320_CUDA",
}


def case_name(iteration: int) -> str:
    return ("NvFuserScheduler_Softmax_Inner_fp32___GRAPH/"
            "NvFuserScheduler_Softmax_Inner_fp32/32768/"
            f"{iteration}/manual_time")


def device_profile(libraries: str) -> tuple[int, int, int]:
    runtime = next((Path(directory) / name
                    for directory in libraries.split(":")
                    for name in ("libcudart.so.12", "libcudart.so")
                    if (Path(directory) / name).is_file()), None)
    if runtime is None:
        raise ValueError("CUDA runtime is absent from the numeric library path")
    cudart = ctypes.CDLL(str(runtime))
    device = ctypes.c_int()
    if cudart.cudaGetDevice(ctypes.byref(device)) != 0:
        raise RuntimeError("cudaGetDevice failed")
    attributes = []
    for attribute in (75, 76, 16):  # compute capability major/minor, SM count
        value = ctypes.c_int()
        if cudart.cudaDeviceGetAttribute(
                ctypes.byref(value), attribute, device.value) != 0:
            raise RuntimeError(f"CUDA attribute {attribute} is unavailable")
        attributes.append(value.value)
    return tuple(attributes)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_numeric_output(output: str, name: str) -> None:
    running = re.findall(r"^\[=+\] Running (\d+) tests? from", output, re.M)
    passed = re.findall(r"^\[\s+PASSED\s+\]\s+(\d+) tests?\.", output, re.M)
    run_names = re.findall(r"^\[\s+RUN\s+\]\s+(\S+)\s*$", output, re.M)
    ok_names = re.findall(r"^\[\s+OK\s+\]\s+(\S+)", output, re.M)
    failed_or_skipped = re.search(r"^\[\s+(?:FAILED|SKIPPED)\s+\]", output, re.M)
    if (running != ["1"] or passed != ["1"] or
            run_names != [name] or ok_names != [name] or failed_or_skipped):
        raise AssertionError(f"numeric GTest did not pass exactly {name}")


def run_numeric(mode: str, ordinal: int, iteration: int, binary: Path,
                loader: Path, libraries: str, root: Path) -> set[str]:
    name = GTEST_NAMES[iteration]
    key = f"{ordinal}:{case_name(iteration)}"
    prefix = f"{ordinal}-{mode}.numeric"
    stdout = root / f"{prefix}.stdout.txt"
    stderr = root / f"{prefix}.stderr.txt"
    decision_log = root / f"{prefix}.decisions.jsonl"
    outcome_log = root / f"{prefix}.outcomes.jsonl"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({"NVFUSER_JEV_MODE": mode,
                "NVFUSER_JEV_OUTCOME_LOG": str(outcome_log),
                "NVFUSER_JEV_CASE_KEY": key})
    if mode != "off":
        decision_log.touch(exist_ok=False)
        env["NVFUSER_JEV_LOG"] = str(decision_log)
        env["NVFUSER_JEV_PORT"] = "1"
    argv = [str(loader), "--library-path", str(binary.parent) + ":" + libraries,
            str(binary), "--gtest_color=no", "--gtest_filter=" + name]
    with stdout.open("x") as out, stderr.open("x") as err:
        result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                check=False)
    if result.returncode != 0:
        raise AssertionError(f"{prefix} exited {result.returncode}; see {stderr}")
    check_numeric_output(stdout.read_text(errors="replace"), name)
    if mode == "remote" and decision_log.stat().st_size:
        raise AssertionError(f"{prefix} logged an unproved IP5 Choice")
    finals = [row for row in read_rows(outcome_log)
              if row.get("schema") == "nvfuser.native_outcome.v2" and
              row.get("kind") == "final" and row.get("case_key") == key]
    if not finals or any(row.get("scheduler_type") != "inner_persistent"
                         for row in finals):
        raise AssertionError(f"{prefix} lacks completed inner-persistent plans")
    plans = {json.dumps({field: row[field] for field in PLAN_FIELDS},
                        sort_keys=True, separators=(",", ":")) for row in finals}
    if len(plans) != 1:
        raise AssertionError(f"{prefix} produced multiple typed plans")
    return plans


def run_mode(mode: str, ordinal: int, iteration: int, binary: Path,
             loader: Path, libraries: str, root: Path) -> tuple[set[str], list[dict], list[dict]]:
    name = case_name(iteration)
    key = f"{ordinal}:{name}"
    prefix = f"{ordinal}-{mode}"
    decision_log = root / f"{prefix}.decisions.jsonl"
    outcome_log = root / f"{prefix}.outcomes.jsonl"
    benchmark_json = root / f"{prefix}.benchmark.json"
    stdout = root / f"{prefix}.stdout.txt"
    stderr = root / f"{prefix}.stderr.txt"
    if mode != "off":
        decision_log.touch(exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({"NVFUSER_JEV_MODE": mode,
                "NVFUSER_JEV_OUTCOME_LOG": str(outcome_log),
                "NVFUSER_JEV_CASE_KEY": key})
    if mode != "off":
        env.update({"NVFUSER_JEV_LOG": str(decision_log),
                    "NVFUSER_JEV_PORT": "1"})
    argv = [str(loader), "--library-path", str(binary.parent) + ":" + libraries,
            str(binary), "--benchmark_filter=^" + re.escape(name) + "$",
            "--benchmark_min_time=0.01", "--benchmark_repetitions=1",
            "--benchmark_out=" + str(benchmark_json),
            "--benchmark_out_format=json"]
    with stdout.open("x") as out, stderr.open("x") as err:
        result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                check=False)
    if result.returncode != 0:
        raise AssertionError(f"{prefix} exited {result.returncode}; see {stderr}")
    measurements = json.loads(benchmark_json.read_text())["benchmarks"]
    if (len(measurements) != 1 or measurements[0]["name"] != name or
            measurements[0]["iterations"] < 1):
        raise AssertionError(f"{prefix} did not measure its exact benchmark")
    outcomes = read_rows(outcome_log)
    finals = [row for row in outcomes
              if row.get("schema") == "nvfuser.native_outcome.v2" and
              row.get("kind") == "final" and row.get("case_key") == key]
    if not finals or any(row.get("scheduler_type") != "inner_persistent"
                         for row in finals):
        raise AssertionError(f"{prefix} lacks completed inner-persistent plans")
    plans = {json.dumps({field: row[field] for field in PLAN_FIELDS},
                        sort_keys=True, separators=(",", ":")) for row in finals}
    if len(plans) != 1:
        raise AssertionError(f"{prefix} produced multiple typed plans")
    decisions = read_rows(decision_log) if mode != "off" else []
    return plans, outcomes, decisions


def check_traced_mode(mode: str, ordinal: int, iteration: int,
                      outcomes: list[dict], decisions: list[dict]) -> int:
    key = f"{ordinal}:{case_name(iteration)}"
    scopes: dict[str, list[dict]] = defaultdict(list)
    for row in outcomes:
        if row.get("schema") == SCHEMA and row.get("case_key") == key:
            if row.get("mode") != mode:
                raise AssertionError("mixed compute modes in one outcome log")
            scopes[row["compute_invocation_id"]].append(row)
    if not scopes:
        raise AssertionError(f"{mode} emitted no compute invocation for {key}")
    for invocation, events in scopes.items():
        events.sort(key=lambda row: row["compute_event_index"])
        expected_final = 5 if mode == "record" else 2
        if ([event["kind"] for event in events] !=
                ["begin", "no_choice", "final"] or
                [event["compute_event_index"] for event in events] !=
                [0, expected_final - 1, expected_final]):
            raise AssertionError(f"{mode} has incomplete IP5 scope {invocation}")
        no_choice = events[1]
        if (no_choice.get("site") != "IP5.magic_zero" or
                no_choice.get("selected") != "on"):
            raise AssertionError(f"{mode} changed the native magic-zero plan")
    if mode == "remote":
        if decisions:
            raise AssertionError("Remote emitted an unproved IP5 Choice")
    else:
        by_scope: dict[str, list[dict]] = defaultdict(list)
        for row in decisions:
            if row.get("case_key") != key or row.get("kind") != "coverage_skip":
                raise AssertionError("Record logged an unexpected decision")
            by_scope[row["compute_invocation_id"]].append(row)
        if set(by_scope) != set(scopes):
            raise AssertionError("Record coverage does not span every invocation")
        for invocation, rows in by_scope.items():
            rows.sort(key=lambda row: row["compute_event_index"])
            if ([(row["site"], row["reason"]) for row in rows] !=
                    list(COVERAGE) or
                    [row["compute_event_index"] for row in rows] != [1, 2, 3]):
                raise AssertionError(f"Record coverage changed for {invocation}")
    return len(scopes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    test_binary = binary.with_name("test_nvfuser").resolve(strict=True)
    if not test_binary.is_file():
        raise ValueError("test_nvfuser is not a sibling of nvfuser_bench")
    loader = Path(os.environ["JEV_NUMERIC_LOADER"]).resolve(strict=True)
    libraries = os.environ["JEV_NUMERIC_LIBRARY_PATH"]
    if any(not Path(path).is_absolute() or not Path(path).is_dir()
           for path in libraries.split(":")):
        raise ValueError("JEV_NUMERIC_LIBRARY_PATH has an invalid directory")
    if device_profile(libraries) != (8, 9, 128):
        print("IP5 native-menu regression requires SM89 with 128 SMs")
        raise SystemExit(77)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ip5-native-menu-", dir=args.artifacts_root))
    summary = []
    for ordinal, iteration in CASES:
        results = {mode: run_mode(mode, ordinal, iteration, binary, loader,
                                  libraries, root)
                   for mode in ("off", "record", "remote")}
        if not (results["off"][0] == results["record"][0] ==
                results["remote"][0]):
            raise AssertionError(f"typed plans differ for ordinal {ordinal}")
        counts = {mode: check_traced_mode(mode, ordinal, iteration,
                                          results[mode][1], results[mode][2])
                  for mode in ("record", "remote")}
        for mode in ("off", "record", "remote"):
            numeric_plan = run_numeric(mode, ordinal, iteration, test_binary,
                                       loader, libraries, root)
            if numeric_plan != results[mode][0]:
                raise AssertionError(f"{mode} numeric plan differs from benchmark"
                                     f" for ordinal {ordinal}")
        summary.append({"ordinal": ordinal, "iteration": iteration,
                        "typed_plan_equal": True, "compute_invocations": counts,
                        "numeric_reference_passed": ["off", "record", "remote"],
                        "numeric_plan_equal": True})
    print(json.dumps({"cases": summary, "artifacts": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
