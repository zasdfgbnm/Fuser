#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Exercise the exact batch-norm shape whose native IP6 plan fails menu preflight.

The benchmark is run in Off, Record and Remote modes. Remote has no API server:
any unintended Choice fails instead of silently selecting an unproved plan.
Each mode also checks the exact shape against an ATen tensor reference.
The enclosing CTest workload owns the processes; no timeout or signal is used.
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


CASE = ("NvFuserScheduler_BatchNorm_fp32___GRAPH/"
        "NvFuserScheduler_BatchNorm_fp32/8/32/64/manual_time")
CASE_KEY = "44:" + CASE
SCHEMA = "nvfuser.jev.compute_invocation.v1"
SITES = (
    ("IP6.width", "w4"),
    ("IP6.waves", "waves4"),
    ("IP6.occupancy", "occ28"),
    ("IP6.warp", "warp8"),
    ("IP6.work_split", "unroll_first"),
    ("IP6.outer_unroll", "u4"),
    ("IP6.reg_overhead", "r16"),
    ("IP6.cta_floor", "floor128"),
)
REASON = "native 3D plan failed complete resource preflight"
SOURCE = "native_3d_preflight_fallback"
PLAN_FIELDS = ("scheduler_type", "params_kind", "scheduler_fields", "launch",
               "compile", "tag")
GTEST_NAME = "Gpu1Test.JevIP6NativeFallbackBatchNorm_CUDA"


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
    return attributes[0], attributes[1], attributes[2]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_numeric_output(output: str) -> None:
    running = re.findall(r"^\[=+\] Running (\d+) tests? from", output, re.M)
    passed = re.findall(r"^\[\s+PASSED\s+\]\s+(\d+) tests?\.", output, re.M)
    run_names = re.findall(r"^\[\s+RUN\s+\]\s+(\S+)\s*$", output, re.M)
    ok_names = re.findall(r"^\[\s+OK\s+\]\s+(\S+)", output, re.M)
    failed_or_skipped = re.search(r"^\[\s+(?:FAILED|SKIPPED)\s+\]", output, re.M)
    if (running != ["1"] or passed != ["1"] or
            run_names != [GTEST_NAME] or ok_names != [GTEST_NAME] or
            failed_or_skipped):
        raise AssertionError(f"numeric GTest did not pass exactly {GTEST_NAME}")


def run_numeric(mode: str, binary: Path, loader: Path, libraries: str,
                root: Path) -> set[str]:
    prefix = f"{mode}.numeric"
    stdout = root / f"{prefix}.stdout.txt"
    stderr = root / f"{prefix}.stderr.txt"
    decision_log = root / f"{prefix}.decisions.jsonl"
    outcome_log = root / f"{prefix}.outcomes.jsonl"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({"NVFUSER_JEV_MODE": mode,
                "NVFUSER_JEV_OUTCOME_LOG": str(outcome_log),
                "NVFUSER_JEV_CASE_KEY": CASE_KEY})
    if mode != "off":
        decision_log.touch(exist_ok=False)
        env["NVFUSER_JEV_LOG"] = str(decision_log)
        env["NVFUSER_JEV_PORT"] = "1"
    argv = [str(loader), "--library-path", str(binary.parent) + ":" + libraries,
            str(binary), "--gtest_color=no", "--gtest_filter=" + GTEST_NAME]
    with stdout.open("x") as out, stderr.open("x") as err:
        result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                check=False)
    if result.returncode != 0:
        raise AssertionError(f"{prefix} exited {result.returncode}; see {stderr}")
    check_numeric_output(stdout.read_text(errors="replace"))
    if mode == "remote" and decision_log.stat().st_size:
        raise AssertionError(f"{prefix} logged an unproved IP6 Choice")
    finals = [row for row in rows(outcome_log)
              if row.get("schema") == "nvfuser.native_outcome.v2" and
              row.get("kind") == "final" and row.get("case_key") == CASE_KEY]
    if not finals or any(row.get("scheduler_type") != "inner_persistent"
                         for row in finals):
        raise AssertionError(f"{prefix} lacks completed inner-persistent plans")
    plans = {json.dumps({field: row[field] for field in PLAN_FIELDS},
                        sort_keys=True, separators=(",", ":")) for row in finals}
    if len(plans) != 1:
        raise AssertionError(f"{prefix} produced multiple typed plans")
    return plans


def run_mode(mode: str, binary: Path, loader: Path, libraries: str,
             root: Path) -> tuple[set[str], list[dict]]:
    decision_log = root / f"{mode}.decisions.jsonl"
    outcome_log = root / f"{mode}.outcomes.jsonl"
    benchmark_json = root / f"{mode}.benchmark.json"
    stdout = root / f"{mode}.stdout.txt"
    stderr = root / f"{mode}.stderr.txt"
    if mode != "off":
        decision_log.touch(exist_ok=False)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("NVFUSER_") and not key.startswith("LD_")}
    env.update({"NVFUSER_JEV_MODE": mode,
                "NVFUSER_JEV_OUTCOME_LOG": str(outcome_log),
                "NVFUSER_JEV_CASE_KEY": CASE_KEY})
    if mode != "off":
        env.update({"NVFUSER_JEV_LOG": str(decision_log),
                    "NVFUSER_JEV_PORT": "1"})
    # Place this checkout's built libraries first even if a caller supplied a
    # library path used for an earlier nvFuser checkout.
    library_path = str(binary.parent) + ":" + libraries
    argv = [str(loader), "--library-path", library_path, str(binary),
            "--benchmark_filter=^" + CASE + "$",
            "--benchmark_min_time=0.01", "--benchmark_repetitions=1",
            "--benchmark_out=" + str(benchmark_json),
            "--benchmark_out_format=json"]
    with stdout.open("x") as out, stderr.open("x") as err:
        result = subprocess.run(argv, env=env, stdout=out, stderr=err,
                                check=False)
    if result.returncode != 0:
        raise AssertionError(f"{mode} exited {result.returncode}; see {stderr}")
    measurements = json.loads(benchmark_json.read_text())["benchmarks"]
    if (len(measurements) != 1 or measurements[0]["name"] != CASE or
            measurements[0]["iterations"] < 1):
        raise AssertionError(f"{mode} did not measure the exact ordinal 44")
    if mode != "off" and decision_log.stat().st_size:
        raise AssertionError(f"{mode} sent an unproved IP6 Choice")
    outcome_rows = rows(outcome_log)
    finals = [row for row in outcome_rows
              if row.get("schema") == "nvfuser.native_outcome.v2" and
              row.get("kind") == "final" and row.get("case_key") == CASE_KEY]
    if not finals or any(row.get("scheduler_type") != "inner_persistent"
                         for row in finals):
        raise AssertionError(f"{mode} lacks completed inner-persistent plans")
    plans = {json.dumps({key: row[key] for key in PLAN_FIELDS},
                        sort_keys=True, separators=(",", ":")) for row in finals}
    if len(plans) != 1:
        raise AssertionError(f"{mode} produced multiple typed plans")
    return plans, outcome_rows


def check_native_only(mode: str, outcome_rows: list[dict]) -> int:
    scopes: dict[str, list[dict]] = defaultdict(list)
    for row in outcome_rows:
        if row.get("schema") == SCHEMA and row.get("case_key") == CASE_KEY:
            if row.get("mode") != mode:
                raise AssertionError("mixed compute modes in one outcome log")
            scopes[row["compute_invocation_id"]].append(row)
    if not scopes:
        raise AssertionError(f"{mode} emitted no compute invocation")
    for invocation, events in scopes.items():
        events.sort(key=lambda row: row["compute_event_index"])
        expected_kinds = ["begin"] + ["no_choice"] * len(SITES) + ["final"]
        if ([row["kind"] for row in events] != expected_kinds or
                [row["compute_event_index"] for row in events] !=
                list(range(len(expected_kinds)))):
            raise AssertionError(f"{mode} incomplete native-only {invocation}")
        actual = [(row.get("site"), row.get("selected"))
                  for row in events[1:-1]]
        if actual != list(SITES) or any(
                row.get("reason") != REASON or
                row.get("selection_source") != SOURCE
                for row in events[1:-1]):
            raise AssertionError(f"{mode} wrong IP6 fallback {invocation}: {actual}")
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
        print("ordinal 44 native-menu regression requires SM89 with 128 SMs")
        raise SystemExit(77)
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ordinal44-", dir=args.artifacts_root))
    results = {mode: run_mode(mode, binary, loader, libraries, root)
               for mode in ("off", "record", "remote")}
    if not (results["off"][0] == results["record"][0] ==
            results["remote"][0]):
        raise AssertionError("Off, Record and Remote typed plans differ")
    counts = {mode: check_native_only(mode, results[mode][1])
              for mode in ("record", "remote")}
    for mode in ("off", "record", "remote"):
        numeric_plan = run_numeric(mode, test_binary, loader, libraries, root)
        if numeric_plan != results[mode][0]:
            raise AssertionError(f"{mode} numeric plan differs from benchmark")
    print(json.dumps({"case_key": CASE_KEY, "artifacts": str(root),
                      "typed_plan_equal": True, "no_choice_sites": len(SITES),
                      "compute_invocations": counts,
                      "numeric_reference_passed": ["off", "record", "remote"],
                      "numeric_plan_equal": True},
                     sort_keys=True))


if __name__ == "__main__":
    main()
