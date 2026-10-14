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


def paired_state_rows(report: dict[str, Any], site: str
                      ) -> tuple[dict[str, Any], dict[str, Any],
                                 list[dict[str, str]]]:
    """Require one request on each side of a changed dependent state."""
    rows = [row for row in report["multiplicity"] if row["site"] == site]
    record = [row for row in rows if row["record"] and not row["remote"]]
    remote = [row for row in rows if row["remote"] and not row["record"]]
    if (len(rows) != 2 or len(record) != 1 or len(remote) != 1 or
            len(record[0]["record"]) != 1 or
            list(record[0]["record"].values()) != [1] or
            len(remote[0]["remote"]) != 1 or
            list(remote[0]["remote"].values()) != [1] or
            record[0]["request_sha256"] == remote[0]["request_sha256"]):
        raise ValueError(f"expected exactly one paired {site} state replacement: {rows}")
    return record[0], remote[0], [
        {"kind": "record_only_state", "site": site,
         "request_sha256": record[0]["request_sha256"]},
        {"kind": "remote_only_state", "site": site,
         "request_sha256": remote[0]["request_sha256"]},
    ]


def state_fields(row: dict[str, Any], version: str) -> dict[str, str]:
    lines = row["request"]["state"].splitlines()
    if not lines or lines[0] != version:
        raise ValueError(f"unexpected {row['site']} state version: {lines}")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise ValueError(f"malformed {row['site']} state: {line}")
        fields[key] = value
    return fields


def check_transpose_tile_dependency(report: dict[str, Any]
                                    ) -> list[dict[str, str]]:
    """Require the chosen tile to reach every downstream TR2 calculation."""
    tr1 = [row for row in report["multiplicity"] if row["site"] == "TR1"]
    if (len(tr1) != 1 or tr1[0]["record"] != tr1[0]["remote"] or
            sum(tr1[0]["record"].values()) != 1):
        raise ValueError(f"tile changed the native virtual-axis plan: {tr1}")

    versions = {"TR2.unroll": "transpose.unroll_cap.v1",
                "TR2.width1": "transpose.group_width.v1",
                "TR2.width2": "transpose.group_width.v1",
                "TR2.cta": "transpose.cta.v1"}
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    issues: list[dict[str, str]] = []
    for site, version in versions.items():
        before, after, replaced = paired_state_rows(report, site)
        left, right = state_fields(before, version), state_fields(after, version)
        if (left.get("tile1") != "32" or left.get("tile2") != "32" or
                right.get("tile1") != "16" or right.get("tile2") != "16"):
            raise ValueError(f"{site} did not rebuild for the selected tile")
        if (len(before["record"]) != 1 or len(after["remote"]) != 1):
            raise ValueError(f"{site} omitted a native or remote winner")
        pairs[site] = (before, after)
        issues.extend(replaced)

    unroll_before, unroll_after = pairs["TR2.unroll"]
    for site in ("TR2.width1", "TR2.width2"):
        before, after = pairs[site]
        for width_row, unroll_row in ((before, unroll_before),
                                      (after, unroll_after)):
            width_state = state_fields(width_row, versions[site])
            cap = next(iter(unroll_row["record"] or unroll_row["remote"]))
            if width_state["max_unroll_input"] != cap[1:]:
                raise ValueError(f"{site} ignored the tile-dependent unroll cap")
        left, right = state_fields(before, versions[site]), state_fields(
            after, versions[site])
        if (left["mapped_sizes"] != right["mapped_sizes"] or
                left["group"] != right["group"]):
            raise ValueError(f"{site} changed native contiguous-axis mapping")

    for cta_row, side in ((pairs["TR2.cta"][0], "record"),
                          (pairs["TR2.cta"][1], "remote")):
        cta = state_fields(cta_row, versions["TR2.cta"])
        for group in (1, 2):
            width_row = pairs[f"TR2.width{group}"][0 if side == "record" else 1]
            width = next(iter(width_row[side]))
            if cta[f"width{group}"] != width[1:]:
                raise ValueError("CTA plan ignored selected transpose width")
    return issues


IOP3_CTA_FIELDS = {
    "inner_dim", "outer_dim", "vector_width", "regs_buffer_bits",
    "tmp_dtype_bits", "warp_size", "sm_count", "max_threads_per_block",
    "max_grid_y", "smem_total_bits", "smem_limit_bits",
    "selected_outer_read_width",
}


def iop3_cta_plan_fields(row: dict[str, Any], width: int
                         ) -> list[tuple[int, ...]]:
    """Read repeated CTA plans while keeping scalar fields strict."""
    lines = row["request"]["state"].splitlines()
    if not lines or lines[0] != "inner_outer_multi_wave.cta.v2":
        raise ValueError(f"unexpected IOP3.cta state version: {lines[:1]}")
    fields: dict[str, str] = {}
    plans: list[tuple[int, ...]] = []
    for line in lines[1:]:
        key, separator, value = line.partition("=")
        if not separator or not value:
            raise ValueError(f"malformed IOP3.cta field: {line}")
        if key == "plan":
            try:
                plan = tuple(int(part) for part in value.split(":"))
            except ValueError as exc:
                raise ValueError(f"malformed IOP3.cta plan: {line}") from exc
            if len(plan) != 14 or any(part < 1 for part in plan):
                raise ValueError(f"incomplete IOP3.cta plan: {line}")
            plans.append(plan)
        elif key not in IOP3_CTA_FIELDS or key in fields:
            raise ValueError(f"unknown or repeated IOP3.cta scalar: {line}")
        else:
            fields[key] = value
    if set(fields) != IOP3_CTA_FIELDS or not 1 <= len(plans) <= 7:
        raise ValueError("IOP3.cta omitted full state or complete plans")
    try:
        scalar = {key: int(value) for key, value in fields.items()}
    except ValueError as exc:
        raise ValueError("IOP3.cta has a nonnumeric scalar") from exc
    if (scalar["inner_dim"] <= 1024 or scalar["vector_width"] < 1 or
            scalar["tmp_dtype_bits"] < 1 or
            scalar["selected_outer_read_width"] != width):
        raise ValueError("IOP3.cta omitted the selected large-inner read width")
    inner = scalar["inner_dim"]
    vector = scalar["vector_width"]
    write = min(128 // scalar["tmp_dtype_bits"], vector)
    previous_threads = 0
    for plan in plans:
        (threads, inner_vect, batch, bdimx, bdimy, grid, tmp_write,
         read, _, _, _, final_x, final_y, final_grid) = plan
        requested_x = (inner // width + grid - 1) // grid
        rounded_pow2 = 1 << (requested_x - 1).bit_length()
        rounded_eight = ((requested_x + 7) // 8) * 8
        expected_x = min(rounded_pow2, rounded_eight)
        while threads % expected_x:
            expected_x = min(expected_x + 8, threads)
        if (inner_vect != vector or tmp_write != write or read != width or
                batch != (inner // vector + threads - 1) // threads or
                bdimx != expected_x or bdimy != threads // expected_x or
                final_x != bdimx or final_y != bdimy or final_grid != grid or
                (previous_threads and previous_threads != threads * 2)):
            raise ValueError(f"IOP3.cta plan disagrees with selected width: {plan}")
        previous_threads = threads
    return plans


def iop3_rows_expected_cta(state: str, rows: int
                           ) -> tuple[dict[str, str], list[tuple[int, ...]]]:
    lines = state.splitlines()
    if not lines or lines[0] != "inner_outer_multi_wave.grid_rows.v1":
        raise ValueError("unexpected IOP3.grid_rows state version")
    scalars: dict[str, str] = {}
    plans: list[tuple[int, ...]] = []
    for line in lines[1:]:
        key, sep, value = line.partition("=")
        if not sep or not value:
            raise ValueError(f"malformed IOP3.grid_rows field: {line}")
        if key == "plan":
            fields = tuple(int(part) for part in value.split(":"))
            if len(fields) != 17:
                raise ValueError("incomplete IOP3.grid_rows CTA plan")
            if fields[0] == rows:
                plans.append(fields[3:])
        elif key in scalars:
            raise ValueError(f"repeated IOP3.grid_rows field: {key}")
        else:
            scalars[key] = value
    if set(scalars) != IOP3_CTA_FIELDS | {
            "selected_min_batch", "min_threads", "max_threads", "thread_cap"}:
        raise ValueError("IOP3.grid_rows omitted upstream or CTA measurements")
    if not plans or len(plans) != (int(scalars["max_threads"]) //
                                 int(scalars["min_threads"])).bit_length():
        raise ValueError("IOP3.grid_rows omitted a complete selected CTA search")
    return ({key: scalars[key] for key in IOP3_CTA_FIELDS}, plans)


def checked_iop3_rows_request(row: dict[str, Any]
                              ) -> tuple[dict[str, str], dict[int, list[tuple[int, ...]]]]:
    """Validate the exact rows menu, occupancy grid and complete CTA searches."""
    text = row["request"]["state"]
    lines = text.splitlines()
    if not lines or lines[0] != "inner_outer_multi_wave.grid_rows.v1":
        raise ValueError("unknown dependent grid-rows state version")
    scalars: dict[str, str] = {}
    groups: dict[int, list[tuple[int, ...]]] = {}
    previous_rows = 0
    for line in lines[1:]:
        key, sep, value = line.partition("=")
        if not sep or not value:
            raise ValueError("malformed dependent grid-rows state")
        if key == "plan":
            fields = tuple(int(part) for part in value.split(":"))
            if (len(fields) != 17 or any(part < 1 for part in fields) or
                    fields[0] < previous_rows):
                raise ValueError("incomplete dependent grid-rows plan")
            groups.setdefault(fields[0], []).append(fields)
            previous_rows = fields[0]
        elif key in scalars:
            raise ValueError("repeated dependent grid-rows scalar")
        else:
            scalars[key] = value
    if (set(scalars) != IOP3_CTA_FIELDS | {
            "selected_min_batch", "min_threads", "max_threads", "thread_cap"} or
            set(groups) != {4, 8, 16}):
        raise ValueError("dependent rows omitted full state or option menu")
    numbers = {key: int(value) for key, value in scalars.items()}
    sm_count = numbers["sm_count"]
    if (sm_count < 1 or numbers["inner_dim"] < 1 or
            numbers["outer_dim"] < 1 or numbers["vector_width"] < 1 or
            numbers["selected_outer_read_width"] < 1 or
            numbers["tmp_dtype_bits"] < 1 or numbers["warp_size"] < 1 or
            numbers["max_grid_y"] < 1 or numbers["regs_buffer_bits"] < 0 or
            numbers["inner_dim"] % numbers["vector_width"] or
            numbers["selected_min_batch"] not in {1, 2, 3, 4} or
            numbers["min_threads"] < 1 or
            numbers["max_threads"] < numbers["min_threads"] or
            numbers["thread_cap"] < numbers["max_threads"] or
            numbers["smem_total_bits"] > numbers["smem_limit_bits"]):
        raise ValueError("invalid dependent rows dimensions or resource bound")
    expected_threads = []
    threads = numbers["max_threads"]
    while threads >= numbers["min_threads"]:
        expected_threads.append(threads)
        threads //= 2
    if threads * 2 != numbers["min_threads"] or len(expected_threads) > 7:
        raise ValueError("incomplete dependent rows CTA search")
    after_vector = numbers["inner_dim"] // numbers["vector_width"]
    minimum = max(numbers["min_threads"],
                  (after_vector + numbers["selected_min_batch"] - 1) //
                  numbers["selected_min_batch"])
    maximum = min(1 << (minimum - 1).bit_length(), numbers["thread_cap"])
    if maximum != numbers["max_threads"]:
        raise ValueError("dependent rows ignored selected minimum batch")
    question = row["request"]["questions"]["IOP3.grid_rows"]
    if question != {
            "type": "choice",
            "instructions": "Choose nvFuser scheduler decision IOP3.grid_rows",
            "criteria": {f"r{rows}": (
                f"Require at least {rows} outer rows per grid-Y block "
                f"across {len(plans)} complete CTA plans")
                for rows, plans in groups.items()}}:
        raise ValueError("dependent rows menu differs from complete CTA searches")
    capacities: dict[int, tuple[int, int]] = {}
    for rows, plans in groups.items():
        if [plan[3] for plan in plans] != expected_threads:
            raise ValueError("dependent rows omitted a CTA candidate")
        cta_scalar, cta_plans = iop3_rows_expected_cta(text, rows)
        cta_state = "\n".join((
            "inner_outer_multi_wave.cta.v2",
            *(f"{key}={value}" for key, value in cta_scalar.items()),
            *("plan=" + ":".join(map(str, plan)) for plan in cta_plans)))
        iop3_cta_plan_fields({"site": "IOP3.cta", "request": {"state": cta_state}},
                             numbers["selected_outer_read_width"])
        for plan in plans:
            (_, capacity, final_capacity, tpb, vector, batch, _, _, grid,
             tmp_write, read, warps, required, available, _, _, final_grid) = plan
            if (capacity % sm_count or final_capacity % sm_count or
                    capacity > 64 * sm_count or
                    final_capacity > 64 * sm_count or
                    final_capacity != capacity or
                    capacity < grid or
                    tmp_write != min(128 // numbers["tmp_dtype_bits"], vector) or
                    read != numbers["selected_outer_read_width"] or
                    warps != ((tpb + numbers["warp_size"] - 1) //
                              numbers["warp_size"]) * (grid // sm_count) or
                    required != min((numbers["regs_buffer_bits"] //
                                     numbers["inner_dim"] * vector * batch) //
                                    32 + 40, 255) or
                    not 1 <= available <= 255):
                raise ValueError("dependent rows plan changed resource measurements")
            cap_blocks = capacity // sm_count
            maximum_grid = ((numbers["outer_dim"] + rows - 1) // rows +
                            sm_count - 1) // sm_count * sm_count
            expected_grid = min(cap_blocks,
                                max(1, maximum_grid // sm_count)) * sm_count
            if grid != expected_grid or final_grid != expected_grid:
                raise ValueError("dependent rows grid disagrees with occupancy cap")
            previous = capacities.setdefault(tpb, (capacity, final_capacity))
            if previous != (capacity, final_capacity):
                raise ValueError("dependent rows changed occupancy capacity")
    return scalars, groups


def check_iop3_grid_rows_dependency(
        report: dict[str, Any], *, batch_pair: tuple[int, int],
        width_pair: tuple[int, int], cta_pair: tuple[str, str]
        ) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Check one replaced rows request and its exact downstream CTA plans."""
    record_rows, remote_rows, rows_issues = paired_state_rows(
        report, "IOP3.grid_rows")
    states = []
    for row, batch, width, side in (
            (record_rows, batch_pair[0], width_pair[0], "record"),
            (remote_rows, batch_pair[1], width_pair[1], "remote")):
        state, groups = checked_iop3_rows_request(row)
        if (row[side] != {"r8": 1} or
                state["inner_dim"] != "2048" or
                state["outer_dim"] != "1024" or
                state["vector_width"] != "4" or
                state["min_threads"] != "128" or
                state["thread_cap"] != "512" or
                state["selected_min_batch"] != str(batch) or
                state["selected_outer_read_width"] != str(width) or
                state["max_threads"] != str(512 if batch == 1 else 256)):
            raise ValueError("dependent grid-rows request changed beyond selected upstream choice")
        states.append((state, groups))
    permitted = {"selected_min_batch", "selected_outer_read_width",
                 "max_threads"}
    record_scalars = {key: value for key, value in states[0][0].items()
                      if key not in permitted}
    remote_scalars = {key: value for key, value in states[1][0].items()
                      if key not in permitted}
    if record_scalars != remote_scalars:
        raise ValueError("upstream choice altered unrelated rows measurements")
    for rows in (4, 8, 16):
        native = states[0][1][rows]
        selected = states[1][1][rows]
        if batch_pair[0] != batch_pair[1]:
            if tuple(native) != tuple(selected[1:]):
                raise ValueError("minimum batch altered the overlapping rows plans")
        else:
            if len(native) != len(selected):
                raise ValueError("read width altered the CTA search cardinality")
            for left, right in zip(native, selected):
                invariant_columns = (0, 1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 16)
                if (tuple(left[index] for index in invariant_columns) !=
                        tuple(right[index] for index in invariant_columns)):
                    raise ValueError("read width altered unrelated rows plan metrics")
    record_cta, remote_cta, cta_issues = paired_state_rows(report, "IOP3.cta")
    for rows_row, cta_row, width, winner, side in (
            (record_rows, record_cta, width_pair[0], cta_pair[0], "record"),
            (remote_rows, remote_cta, width_pair[1], cta_pair[1], "remote")):
        scalar, plans = iop3_rows_expected_cta(rows_row["request"]["state"], 8)
        lines = cta_row["request"]["state"].splitlines()
        if not lines or lines[0] != "inner_outer_multi_wave.cta.v2":
            raise ValueError("dependent CTA omitted the rows plan version")
        cta_scalar = dict(line.split("=", 1) for line in lines[1:]
                          if not line.startswith("plan="))
        if (cta_scalar != scalar or
                iop3_cta_plan_fields(cta_row, width) != plans or
                cta_row[side] != {winner: 1}):
            raise ValueError("dependent CTA omitted exact grid-rows geometry")
    return [*rows_issues, *cta_issues], record_cta, remote_cta


def check_iop3_inner_width_dependency(
        report: dict[str, Any], event: dict[str, Any],
        intended: dict[str, str], issues: list[dict[str, str]]) -> None:
    """Pair and replay every IOP3 request after a narrower inner vector."""
    from jev_shim import (evaluate, parse_inner_outer_inner_width_state)

    target = parse_inner_outer_inner_width_state(event["state"])
    if (event["native"] != "w4" or event["requested_alternate"] != "w2" or
            target.inner_dim != 2048 or target.outer_dim != 1024 or
            not {"w2", "w4"} <= set(event["candidates"])):
        raise ValueError("inner-outer fixture lacks a complete vector-width alternate")
    target_plans = {plan[0]: plan for plan in target.plans}
    paired = {}
    expected = [intended]
    for site in ("IOP3.min_batch", "IOP3.tmp_read_width",
                 "IOP3.grid_rows", "IOP3.cta", "IOP3.tmp_write_width"):
        before, after, pair_issues = paired_state_rows(report, site)
        paired[site] = (before, after)
        expected.extend(pair_issues)
        for row, side in ((before, "record"), (after, "remote")):
            try:
                response, _ = evaluate(row["request"])
                native = response["answers"][site]["choice"]
            except Exception as exc:
                raise ValueError(f"invalid dependent {site} request") from exc
            if row[side] != {native: 1}:
                raise ValueError(f"dependent {site} changed its native winner")

    batch_rows = paired["IOP3.min_batch"]
    read_rows = paired["IOP3.tmp_read_width"]
    rows_rows = paired["IOP3.grid_rows"]
    cta_rows = paired["IOP3.cta"]
    write_rows = paired["IOP3.tmp_write_width"]
    for index, width in enumerate((4, 2)):
        batch = batch_rows[index]
        read = read_rows[index]
        rows = rows_rows[index]
        cta = cta_rows[index]
        write = write_rows[index]
        batch_state = batch["request"]["state"].splitlines()
        batch_fields = dict(line.split("=", 1) for line in batch_state[1:]
                            if not line.startswith("plan="))
        read_fields = state_fields(read, "inner_outer_multi_wave.tmp_read_width.v1")
        rows_fields, _ = checked_iop3_rows_request(rows)
        write_fields = state_fields(write, "inner_outer_multi_wave.tmp_write_width.v2")
        cta_fields, cta_plans = iop3_rows_expected_cta(
            rows["request"]["state"], 8)
        cta_lines = cta["request"]["state"].splitlines()
        actual_cta_fields = dict(line.split("=", 1) for line in cta_lines[1:]
                                 if not line.startswith("plan="))
        if (batch_state[0] != "inner_outer_multi_wave.minimum_batch.v1" or
                batch_fields.get("vector_width") != str(width) or
                batch_fields.get("after_vector") != str(2048 // width) or
                read_fields["inner_vector_width"] != str(width) or
                rows_fields["vector_width"] != str(width) or
                rows_fields["selected_min_batch"] != "2" or
                rows_fields["selected_outer_read_width"] != "2" or
                rows_fields["max_threads"] != str(512 if width == 2 else 256) or
                rows_fields["smem_total_bits"] != str(
                    target_plans[width][2] + target_plans[width][3]) or
                rows_fields["regs_buffer_bits"] != str(target_plans[width][1]) or
                actual_cta_fields != cta_fields or
                iop3_cta_plan_fields(cta, 2) != cta_plans or
                write_fields["inner_vector_width"] != str(width) or
                write_fields["outer_read_width"] != "2" or
                write_fields["tmp_dtype_bits"] != "32"):
            raise ValueError("inner width did not produce the exact dependent plan")
    for site, version, changing in (
            ("IOP3.tmp_read_width", "inner_outer_multi_wave.tmp_read_width.v1",
             {"inner_vector_width", "legal_widths"}),
            ("IOP3.tmp_write_width", "inner_outer_multi_wave.tmp_write_width.v2",
             {"inner_vector_width"})):
        before, after = paired[site]
        left, right = (state_fields(row, version) for row in (before, after))
        if ({key: value for key, value in left.items() if key not in changing} !=
                {key: value for key, value in right.items() if key not in changing}):
            raise ValueError(f"inner width changed unrelated {site} fields")
    batch_before, batch_after = (
        {key: value for key, value in
         (line.split("=", 1) for line in row["request"]["state"].splitlines()[1:]
          if not line.startswith("plan="))}
        for row in batch_rows)
    if ({key: value for key, value in batch_before.items()
         if key not in {"vector_width", "after_vector"}} !=
            {key: value for key, value in batch_after.items()
             if key not in {"vector_width", "after_vector"}}):
        raise ValueError("inner width changed unrelated batch measurements")
    left, right = (checked_iop3_rows_request(row)[0] for row in rows_rows)
    changed = {"vector_width", "max_threads", "smem_total_bits"}
    if ({key: value for key, value in left.items() if key not in changed} !=
            {key: value for key, value in right.items() if key not in changed}):
        raise ValueError("inner width changed unrelated rows measurements")
    left_cta, right_cta = (iop3_rows_expected_cta(
        row["request"]["state"], 8)[0] for row in rows_rows)
    if ({key: value for key, value in left_cta.items() if key not in changed} !=
            {key: value for key, value in right_cta.items() if key not in changed}):
        raise ValueError("inner width changed unrelated CTA measurements")
    if sorted(issues, key=str) != sorted(expected, key=str):
        raise ValueError(f"unexpected inner-width decision differences: {issues}")


OUTER_STRIDE_KEYS = {
    "reduction_numel", "iteration_numel", "dtype_bits", "sm_count",
    "min_warp_size", "bdimx", "bdimy", "iter_unroll", "redu_unroll",
    "gidim", "base_grdim",
}

OUTER_RAGGED_FORWARD_KEYS = {
    "reduction_numel", "iteration_numel", "dtype_bits", "sm_count",
    "bdimx", "bdimy", "iter_unroll", "redu_unroll", "gidim", "base_grdim",
}


def outer_stride_fields(row: dict[str, Any]) -> dict[str, str]:
    version = row["request"]["state"].splitlines()[0]
    if version not in ("reduction.outer_grid_stride.v1",
                       "reduction.outer_grid_stride.v2"):
        raise ValueError(f"unexpected RD9.stride state version: {version}")
    fields = state_fields(row, version)
    expected = OUTER_STRIDE_KEYS | ({"ragged_policy"} if version.endswith("v2")
                                    else set())
    if set(fields) != expected:
        raise ValueError(f"RD9.stride state keys differ from {version}: {fields}")
    if version.endswith("v2") and fields["ragged_policy"] != "clean":
        raise ValueError(f"dependent RD9.stride changed ragged policy: {fields}")
    return fields


def check_outer_ragged_dependency(report: dict[str, Any],
                                  record_stride: dict[str, Any],
                                  remote_stride: dict[str, Any],
                                  vector_widths: tuple[str, str]
                                  ) -> list[dict[str, str]]:
    versions = [row["request"]["state"].splitlines()[0] for row in
                (record_stride, remote_stride)]
    if versions == ["reduction.outer_grid_stride.v1"] * 2:
        if any(row["site"] == "RD9.ragged" for row in report["multiplicity"]):
            raise ValueError("v1 stride unexpectedly contains RD9.ragged requests")
        return []
    if versions != ["reduction.outer_grid_stride.v2"] * 2:
        raise ValueError(f"mixed dependent stride versions: {versions}")
    record_ragged, remote_ragged, ragged_issues = paired_state_rows(
        report, "RD9.ragged")
    for ragged, stride, width, side in (
            (record_ragged, record_stride, vector_widths[0], "record"),
            (remote_ragged, remote_stride, vector_widths[1], "remote")):
        if (ragged[side] != {"clean": 1} or
                set(ragged["request"]["questions"]["RD9.ragged"]["criteria"]) !=
                {"clean"}):
            raise ValueError("dependent RD9.ragged winner or menu changed")
        selected = state_fields(ragged, "reduction.outer_ragged.v1")
        forwarded = outer_stride_fields(stride)
        if (set(selected) != OUTER_RAGGED_FORWARD_KEYS | {
                "max_threads_per_block", "max_unroll", "vector_width"} or
                selected["vector_width"] != width or
                any(selected[key] != forwarded[key]
                    for key in OUTER_RAGGED_FORWARD_KEYS)):
            raise ValueError(f"RD9.ragged omitted selected grid plan: {selected}")
    return ragged_issues


def clean_outer_grid_waves(raw_grid: int, output_grid: int, sms: int) -> int:
    """Replay RD9's native ragged-wave contraction for a selected raw grid."""
    blocks = raw_grid * output_grid
    if blocks < sms * 16 and blocks % sms != 0 and blocks > sms:
        waves = max(blocks // sms, 1)
        cleaned = max(waves * sms // output_grid, 1)
        if ((raw_grid - cleaned) * 4 <= raw_grid and
                cleaned * output_grid % sms > blocks % sms):
            return cleaned
    return raw_grid


def check_outer_grid_plan(record: dict[str, Any], remote: dict[str, Any],
                          record_stride: dict[str, Any],
                          remote_stride: dict[str, Any]) -> None:
    if record["record"] != {"grid": 1} or remote["remote"] != {"grid": 1}:
        raise ValueError("selected grid plan changed the RD8 winner")
    for rd8, stride in ((record, record_stride), (remote, remote_stride)):
        plan = state_fields(rd8, "reduction.outer_plan.v1")
        selected = outer_stride_fields(stride)
        for plan_key, stride_key in (("grid_gidim", "gidim"),
                                     ("grid_grdim", "base_grdim"),
                                     ("grid_iter_unroll", "iter_unroll")):
            if (plan_key not in plan or stride_key not in selected or
                    plan[plan_key] != selected[stride_key]):
                raise ValueError(f"RD8 omitted selected stride plan: {plan}, {selected}")


def check_outer_unroll_dependency(
        report: dict[str, Any], changed_field: str,
        before_value: str, after_value: str) -> list[dict[str, str]]:
    """Require an upstream choice to reach RD9's complete unroll menu."""
    before, after, issues = paired_state_rows(report, "RD9.unroll")
    left = state_fields(before, "reduction.outer_unroll_distribution.v1")
    right = state_fields(after, "reduction.outer_unroll_distribution.v1")
    if (left.get(changed_field) != before_value or
            right.get(changed_field) != after_value or
            {key: value for key, value in left.items()
             if key != changed_field} !=
            {key: value for key, value in right.items()
             if key != changed_field}):
        raise ValueError(f"RD9.unroll missed selected {changed_field}: {left}, {right}")
    for row, side in ((before, "record"), (after, "remote")):
        if (len(row[side]) != 1 or
                next(iter(row[side])) not in
                row["request"]["questions"]["RD9.unroll"]["criteria"]):
            raise ValueError("dependent unroll winner is outside its complete menu")
    return issues


def check_alternate_report(report: dict[str, Any],
                           events: list[dict[str, Any]], *, site: str,
                           alternate: str, case_key: str,
                           require_grid: bool = False,
                           require_outer_width: bool = False,
                           require_outer_cta: bool = False,
                           require_outer_ragged: bool = False,
                           require_outer_grid_base: bool = False,
                           require_outer_unroll: bool = False,
                           require_inner_iter_unroll: bool = False,
                           require_inner_warp_target: bool = False,
                           require_inner_cta_envelope: bool = False,
                           require_inner3d_warp_target: bool = False,
                           require_inner3d_cta_target: bool = False,
                           require_inner3d_cta_full: bool = False,
                           require_inner3d_grid_wave: bool = False,
                           require_inner_width: bool = False,
                           require_inner_batch_cap: bool = False,
                           require_inner_targets: bool = False,
                           require_inner3d_warp: bool = False,
                           require_inner3d_occupancy: bool = False,
                           require_inner3d_width: bool = False,
                           require_inner3d_outer_unroll: bool = False,
                           require_inner3d_register_overhead: bool = False,
                           require_iop3_min_batch: bool = False,
                           require_iop3_grid_rows: bool = False,
                           require_iop3_tmp_write: bool = False,
                           require_iop3_tmp_read: bool = False,
                           require_iop3_inner_width: bool = False,
                           require_transpose_tile: bool = False,
                           require_pointwise_1d_cta: bool = False) -> dict[str, Any]:
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
    if require_transpose_tile:
        if (site != "TR2.tile" or event["native"] != "tile32" or
                alternate != "tile16" or
                set(event["candidates"]) != {"tile32", "tile16"}):
            raise ValueError("transpose fixture lacks its complete tile menu")
        expected = [intended, *check_transpose_tile_dependency(report)]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected tile-dependent decisions: {issues}")
    elif require_pointwise_1d_cta:
        from jev_shim import evaluate, parse_launch_1d_state, parse_unroll_state

        launch = parse_launch_1d_state(event["state"])
        if (site != "PW3.cta_1d" or event["native"] != "x128" or
                alternate not in {"x32", "x64"} or
                set(event["candidates"]) != {"x32", "x64", "x128"} or
                launch.n_elems != 16388 or launch.vector_width != 4 or
                launch.warp_size != 32):
            raise ValueError("1D pointwise fixture lacks its complete CTA menu")
        before, after, dependent_issues = paired_state_rows(
            report, "PW3.unroll")
        original = state_fields(before, "pointwise.unroll.v1")
        selected = state_fields(after, "pointwise.unroll.v1")
        for row, fields, threads, side in (
                (before, original, 128, "record"),
                (after, selected, int(alternate[1:]), "remote")):
            state = parse_unroll_state(row["request"]["state"])
            response, _ = evaluate(row["request"])
            blocks = (launch.n_elems // launch.vector_width + threads - 1) // threads
            if (state.break_point != 0 or state.bdimx != threads or
                    state.range_low != 1 or state.range_high != blocks or
                    row[side] != {"u1": 1} or
                    response["answers"]["PW3.unroll"]["choice"] != "u1" or
                    set(row["request"]["questions"]["PW3.unroll"]["criteria"]) !=
                    {f"u{factor}" for factor in range(1, blocks + 1)}):
                raise ValueError("1D CTA did not reach a complete dependent unroll plan")
        if ({key: value for key, value in original.items()
             if key not in {"bdimx", "range_high"}} !=
                {key: value for key, value in selected.items()
                 if key not in {"bdimx", "range_high"}}):
            raise ValueError("1D CTA changed unrelated unroll measurements")
        for row in report["multiplicity"]:
            if row["site"] not in {site, "PW3.unroll"} and (
                    row["record"] != row["remote"] or
                    sum(row["record"].values()) != 1):
                raise ValueError(f"unrelated pointwise decision changed: {row}")
        if sorted(issues, key=str) != sorted([intended, *dependent_issues], key=str):
            raise ValueError(f"unexpected 1D CTA dependencies: {issues}")
    elif site == "OP3.redu_unroll":
        if event["native"] != "r1" or set(event["candidates"]) != {
                "r1", "r2", "r4"}:
            raise ValueError("outer reduction fixture lacks its complete factor menu")
        dependent = []
        for nested_site, record_tag, remote_tag in (
                ("OP3.unroll", "outer_persistent.iter_unroll.v1\n",
                 "outer_persistent.iter_unroll.v2\n"),
                ("OP3.cta", "outer_persistent.block_cta.v1\n",
                 "outer_persistent.block_cta.v3\n")):
            rows = [entry for entry in report["multiplicity"]
                    if entry["site"] == nested_site]
            record_only = [entry for entry in rows
                           if sum(entry["record"].values()) == 1 and
                           not entry["remote"]]
            remote_only = [entry for entry in rows
                           if sum(entry["remote"].values()) == 1 and
                           not entry["record"]]
            if len(rows) != 2 or len(record_only) != 1 or len(remote_only) != 1:
                raise ValueError(f"expected exactly one replaced {nested_site}: {rows}")
            if (not record_only[0]["request"]["state"].startswith(record_tag) or
                    not remote_only[0]["request"]["state"].startswith(remote_tag)):
                raise ValueError(f"{nested_site} did not use dependent state versions")
            fields = dict(line.split("=", 1) for line in
                          remote_only[0]["request"]["state"].splitlines()[1:])
            if (fields.get("redu_unroll") != alternate[1:] or
                    (nested_site == "OP3.cta" and fields.get("iter_unroll") != "1")):
                raise ValueError(f"{nested_site} did not receive {alternate}: {fields}")
            dependent.extend((
                {"kind": "record_only_state", "site": nested_site,
                 "request_sha256": record_only[0]["request_sha256"]},
                {"kind": "remote_only_state", "site": nested_site,
                 "request_sha256": remote_only[0]["request_sha256"]},
            ))
        if sorted(issues, key=str) != sorted([intended, *dependent], key=str):
            raise ValueError(f"unexpected dependent reduction issues: {issues}")
    elif site == "OP3.unroll":
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
    elif site == "OP2.base" and alternate == "x11_y11_b16":
        # The bounded register-margin plan bypasses the native no-spill
        # estimate. It is already a complete grid/batch/unswitch plan, so the
        # dependent tail and unswitch menus intentionally stay native-only.
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "outer_persistent.grid_base.v2")
        if (event["native"] != "x8_y16_b16" or
                "x11_y11_b16" not in event["candidates"] or
                "x12_y10_b16" in event["candidates"] or
                any(fields.get(key) != value for key, value in {
                    "reduction_numel": "8192", "iteration_numel": "1024",
                    "vector_width": "4", "persistent_buffer_bits": "262144",
                    "regs_per_block": "65536"}.items())):
            raise ValueError("register-margin fixture changed its bounded plan")
        expected = [intended]
        for nested_site, version, winner in (
                ("OP2.tail", "outer_persistent.grid_tail.v1", "y16_b32"),
                ("OP2.unswitch", "outer_persistent.unswitch.v1", "u32")):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == nested_site]
            if (len(rows) != 1 or rows[0]["record"] != {winner: 1} or
                    rows[0]["remote"] or
                    not rows[0]["request"]["state"].startswith(version + "\n")):
                raise ValueError(f"unexpected dependent {nested_site}: {rows}")
            expected.append({"kind": "record_only_state", "site": nested_site,
                             "request_sha256": rows[0]["request_sha256"]})
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected register-margin issues: {issues}")
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
        record_unswitch, remote_unswitch, unswitch_issues = paired_state_rows(
            report, "OP2.unswitch")
        for row, base_id, side in (
                (record_unswitch, event["native"], "record"),
                (remote_unswitch, alternate, "remote")):
            base_match = re.fullmatch(r"x([0-9]+)_y([0-9]+)_b([0-9]+)", base_id)
            if base_match is None:
                raise ValueError(f"invalid OP2.base Choice ID: {base_id}")
            block_y = 256 // int(base_match.group(3))
            fields = state_fields(row, "outer_persistent.unswitch.v1")
            if (fields.get("block_y") != str(block_y) or
                    fields.get("persistent_batch") != "32" or
                    row[side] != {"u32": 1}):
                raise ValueError(f"dependent unswitch ignored {base_id}: {row}")
        expected.extend(unswitch_issues)
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected decision comparison issues: {issues}")
    elif site == "OP2.unswitch":
        if (event["native"] != "u32" or alternate != "u4" or
                set(event["candidates"]) != {"u1", "u2", "u4", "u8", "u32"}):
            raise ValueError("outer grid fixture lacks its bounded unswitch menu")
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "outer_persistent.unswitch.v1")
        if any(fields.get(key) != value for key, value in {
                "reduction_numel": "8192", "block_x": "16",
                "block_y": "16", "grid_y": "16",
                "persistent_batch": "32"}.items()):
            raise ValueError(f"unswitch did not receive the final grid plan: {fields}")
        for upstream in ("OP2.base", "OP2.tail"):
            rows = [row for row in report["multiplicity"] if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed: {rows}")
        if issues != [intended]:
            raise ValueError(f"unexpected unswitch comparison issues: {issues}")
    elif require_outer_ragged:
        if (site != "RD9.ragged" or event["native"] != "clean" or
                alternate != "raw" or
                set(event["candidates"]) != {"clean", "raw"}):
            raise ValueError("outer ragged fixture lacks both complete grid plans")
        for upstream in ("RD9.reg_budget", "RD9.width", "RD9.waves",
                         "RD9.warp", "RD9.cta", "RD9.grid_base",
                         "RD9.unroll"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with ragged choice: {rows}")
        strides = [row for row in report["multiplicity"]
                   if row["site"] == "RD9.stride"]
        clean = [row for row in strides if row["record"] == {"g7": 1} and
                 not row["remote"]]
        raw = [row for row in strides if row["remote"] == {"g8": 1} and
               not row["record"]]
        if len(strides) != 2 or len(clean) != 1 or len(raw) != 1:
            raise ValueError(f"ragged choice did not reach both stride grids: {strides}")
        if ("base_grdim=8\nragged_policy=clean" not in
                clean[0]["request"]["state"] or
                "base_grdim=8\nragged_policy=raw" not in
                raw[0]["request"]["state"]):
            raise ValueError("stride grid did not receive the selected ragged policy")
        allowed = {site, "RD9.stride", "RD8"}
        if any(issue["site"] not in allowed for issue in issues):
            raise ValueError(f"unrelated ragged decision issue: {issues}")
        for row in (clean[0], raw[0]):
            kind = ("record_only_state" if row is clean[0]
                    else "remote_only_state")
            if {"kind": kind, "site": "RD9.stride",
                "request_sha256": row["request_sha256"]} not in issues:
                raise ValueError(f"missing dependent stride state: {issues}")
    elif require_outer_unroll:
        if (site != "RD9.unroll" or event["native"] != "u4" or
                alternate != "u1" or
                set(event["candidates"]) != {"u1", "u2", "u4"}):
            raise ValueError("outer unroll fixture lacks its complete plan menu")
        state = state_fields({"site": site, "request": {
            "state": event["state"]}},
            "reduction.outer_unroll_distribution.v1")
        if any(state.get(key) != value for key, value in {
                "reduction_numel": "16384", "iteration_numel": "512",
                "vector_width": "4", "max_unroll": "32",
                "cta_share": "4", "grid_base": "target"}.items()):
            raise ValueError(f"outer unroll fixture changed shape: {state}")
        for upstream in ("RD9.reg_budget", "RD9.width", "RD9.waves",
                         "RD9.warp", "RD9.cta", "RD9.grid_base"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with unroll: {rows}")
        record_ragged, remote_ragged, ragged_issues = paired_state_rows(
            report, "RD9.ragged")
        record_stride, remote_stride, stride_issues = paired_state_rows(
            report, "RD9.stride")
        record_plan, remote_plan, plan_issues = paired_state_rows(report, "RD8")
        expected_factors = (("4", "8", "8", "32"),
                            ("1", "32", "32", "8"))
        for row, factors in zip((record_ragged, remote_ragged), expected_factors):
            fields = state_fields(row, "reduction.outer_ragged.v1")
            if tuple(fields.get(key) for key in (
                    "iter_unroll", "redu_unroll", "gidim", "base_grdim")) != factors:
                raise ValueError(f"ragged plan ignored selected unroll: {fields}")
        for row, factors in zip((record_stride, remote_stride), expected_factors):
            fields = outer_stride_fields(row)
            if tuple(fields.get(key) for key in (
                    "iter_unroll", "redu_unroll", "gidim", "base_grdim")) != factors:
                raise ValueError(f"stride plan ignored selected unroll: {fields}")
        if (record_stride["record"] != {"g32": 1} or
                remote_stride["remote"] != {"g8": 1}):
            raise ValueError("outer unroll did not select its complete grid")
        check_outer_grid_plan(
            record_plan, remote_plan, record_stride, remote_stride)
        before = state_fields(record_plan, "reduction.outer_plan.v1")
        after = state_fields(remote_plan, "reduction.outer_plan.v1")
        for field in ("block_gidim", "block_grdim", "block_iter_unroll"):
            if before[field] != after[field]:
                raise ValueError("outer unroll changed the block plan")
        expected = [intended, *ragged_issues, *stride_issues, *plan_issues]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected outer unroll dependencies: {issues}")
    elif require_outer_grid_base:
        if (site != "RD9.grid_base" or event["native"] != "target" or
                alternate != "wave" or
                set(event["candidates"]) != {"target", "wave"}):
            raise ValueError("outer grid baseline fixture lacks both complete plans")
        for upstream in ("RD9.reg_budget", "RD9.width", "RD9.waves",
                         "RD9.warp", "RD9.cta"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with grid baseline: {rows}")
        base_state = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.outer_grid_base.v1")
        if any(base_state.get(key) != value for key, value in {
                "reduction_numel": "16384", "iteration_numel": "513",
                "vector_width": "1", "cta_share": "4"}.items()):
            raise ValueError(f"grid baseline fixture changed shape: {base_state}")
        unroll_issues = check_outer_unroll_dependency(
            report, "grid_base", "target", "wave")
        record_ragged, remote_ragged, ragged_issues = paired_state_rows(
            report, "RD9.ragged")
        for row, side, base in ((record_ragged, "record", "8"),
                                (remote_ragged, "remote", "4")):
            state = state_fields(row, "reduction.outer_ragged.v1")
            if (state.get("base_grdim") != base or
                    row[side] != {"clean": 1} or
                    set(row["request"]["questions"]["RD9.ragged"]
                        ["criteria"]) != {"clean", "raw"}):
                raise ValueError(f"ragged plan ignored selected baseline: {row}")
        ragged_before = state_fields(record_ragged, "reduction.outer_ragged.v1")
        ragged_after = state_fields(remote_ragged, "reduction.outer_ragged.v1")
        if ({key: value for key, value in ragged_before.items()
             if key != "base_grdim"} !=
                {key: value for key, value in ragged_after.items()
                 if key != "base_grdim"}):
            raise ValueError("grid baseline changed unrelated ragged measurements")
        record_clean = clean_outer_grid_waves(
            int(ragged_before["base_grdim"]),
            int(ragged_before["gidim"]), int(ragged_before["sm_count"]))
        remote_clean = clean_outer_grid_waves(
            int(ragged_after["base_grdim"]),
            int(ragged_after["gidim"]), int(ragged_after["sm_count"]))
        if (record_clean, remote_clean) != (7, 3):
            raise ValueError("baseline fixture changed ragged-wave contraction")
        record_stride, remote_stride, stride_issues = paired_state_rows(
            report, "RD9.stride")
        stride_before = outer_stride_fields(record_stride)
        stride_after = outer_stride_fields(remote_stride)
        if (stride_before.get("base_grdim") != "8" or
                stride_after.get("base_grdim") != "4" or
                record_stride["record"] != {f"g{record_clean}": 1} or
                remote_stride["remote"] != {f"g{remote_clean}": 1} or
                {key: value for key, value in stride_before.items()
                 if key != "base_grdim"} !=
                {key: value for key, value in stride_after.items()
                 if key != "base_grdim"}):
            raise ValueError("stride grid did not follow selected baseline")
        record_plan, remote_plan, plan_issues = paired_state_rows(report, "RD8")
        plan_before = state_fields(record_plan, "reduction.outer_plan.v1")
        plan_after = state_fields(remote_plan, "reduction.outer_plan.v1")
        if (record_plan["record"] != {"grid": 1} or
                remote_plan["remote"] != {"grid": 1} or
                plan_before.get("grid_grdim") != str(record_clean) or
                plan_after.get("grid_grdim") != str(remote_clean) or
                {key: value for key, value in plan_before.items()
                 if key != "grid_grdim"} !=
                {key: value for key, value in plan_after.items()
                 if key != "grid_grdim"}):
            raise ValueError("RD8 did not receive selected baseline grid")
        expected = [intended, *unroll_issues, *ragged_issues,
                    *stride_issues, *plan_issues]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected dependent baseline issues: {issues}")
    elif require_outer_cta:
        if (site != "RD9.cta" or event["native"] != "q4" or
                alternate != "q8" or
                set(event["candidates"]) != {"q2", "q4", "q8"}):
            raise ValueError("outer CTA fixture lacks its three complete plans")
        for upstream in ("RD9.reg_budget", "RD9.width", "RD9.waves", "RD9.warp"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with CTA target: {rows}")
        base_record, base_remote, base_issues = paired_state_rows(
            report, "RD9.grid_base")
        base_before = state_fields(base_record, "reduction.outer_grid_base.v1")
        base_after = state_fields(base_remote, "reduction.outer_grid_base.v1")
        if (base_record["record"] != {"target": 1} or
                base_remote["remote"] != {"target": 1} or
                base_before.get("cta_share") != "4" or
                base_after.get("cta_share") != "8" or
                {key: value for key, value in base_before.items()
                 if key != "cta_share"} !=
                {key: value for key, value in base_after.items()
                 if key != "cta_share"}):
            raise ValueError("CTA target did not reach dependent grid baseline")
        unroll_issues = check_outer_unroll_dependency(
            report, "cta_share", "4", "8")
        record_stride, remote_stride, stride_issues = paired_state_rows(
            report, "RD9.stride")
        before, after = (outer_stride_fields(row) for row in
                         (record_stride, remote_stride))
        ragged_issues = check_outer_ragged_dependency(
            report, record_stride, remote_stride, ("4", "4"))
        if (before["bdimx"] != "16" or before["bdimy"] != "32" or
                before["base_grdim"] != "32" or
                after["bdimx"] != "16" or after["bdimy"] != "16" or
                after["base_grdim"] != "64" or
                record_stride["record"] != {"g32": 1} or
                remote_stride["remote"] != {"g64": 1}):
            raise ValueError(f"outer CTA did not reshape the grid plan: {before}, {after}")
        record_plan, remote_plan, plan_issues = paired_state_rows(report, "RD8")
        check_outer_grid_plan(record_plan, remote_plan, record_stride, remote_stride)
        plan_before = state_fields(record_plan, "reduction.outer_plan.v1")
        plan_after = state_fields(remote_plan, "reduction.outer_plan.v1")
        if (plan_before.get("grid_grdim") != "32" or
                plan_after.get("grid_grdim") != "64" or
                {key: value for key, value in plan_before.items()
                 if key != "grid_grdim"} !=
                {key: value for key, value in plan_after.items()
                 if key != "grid_grdim"}):
            raise ValueError("CTA target changed more than the RD8 grid size")
        expected = [intended, *base_issues, *unroll_issues, *ragged_issues,
                    *stride_issues, *plan_issues]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected outer CTA comparison issues: {issues}")
    elif require_outer_width:
        if (site != "RD9.width" or event["native"] != "w4" or
                alternate != "w1" or
                set(event["candidates"]) != {"w1", "w2", "w4"}):
            raise ValueError("outer width fixture lacks its aligned width menu")
        for upstream in ("RD9.reg_budget", "RD9.waves", "RD9.warp"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with width: {rows}")
        record_cta, remote_cta, cta_issues = paired_state_rows(report, "RD9.cta")
        cta_before = state_fields(record_cta, "reduction.outer_cta.v1")
        cta_after = state_fields(remote_cta, "reduction.outer_cta.v1")
        if (record_cta["record"] != {"q4": 1} or
                remote_cta["remote"] != {"q4": 1} or
                cta_before.get("vector_width") != "4" or
                cta_after.get("vector_width") != "1" or
                {key: value for key, value in cta_before.items()
                 if key != "vector_width"} !=
                {key: value for key, value in cta_after.items()
                 if key != "vector_width"}):
            raise ValueError("width changed the CTA winner or upstream CTA state")
        base_record, base_remote, base_issues = paired_state_rows(
            report, "RD9.grid_base")
        base_before = state_fields(base_record, "reduction.outer_grid_base.v1")
        base_after = state_fields(base_remote, "reduction.outer_grid_base.v1")
        if (base_record["record"] != {"target": 1} or
                base_remote["remote"] != {"target": 1} or
                base_before.get("vector_width") != "4" or
                base_after.get("vector_width") != "1" or
                {key: value for key, value in base_before.items()
                 if key != "vector_width"} !=
                {key: value for key, value in base_after.items()
                 if key != "vector_width"}):
            raise ValueError("width did not reach dependent grid baseline")
        unroll_issues = check_outer_unroll_dependency(
            report, "vector_width", "4", "1")
        record_stride, remote_stride, stride_issues = paired_state_rows(
            report, "RD9.stride")
        before, after = (outer_stride_fields(row) for row in
                         (record_stride, remote_stride))
        ragged_issues = check_outer_ragged_dependency(
            report, record_stride, remote_stride,
            (cta_before["vector_width"], cta_after["vector_width"]))
        record_unroll = int(before["iter_unroll"])
        remote_unroll = int(after["iter_unroll"])
        if record_unroll <= remote_unroll or remote_unroll != 1:
            raise ValueError("outer width did not lower dependent iteration unroll")
        if (record_stride["record"] != {"g32": 1} or
                remote_stride["remote"] != {"g8": 1}):
            raise ValueError("width changed the stride winner unexpectedly")
        record_plan, remote_plan, plan_issues = paired_state_rows(report, "RD8")
        check_outer_grid_plan(record_plan, remote_plan, record_stride, remote_stride)
        expected = [intended, *cta_issues, *base_issues, *unroll_issues,
                    *ragged_issues, *stride_issues,
                    *plan_issues]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected outer-width comparison issues: {issues}")
    elif require_inner3d_grid_wave:
        if (site != "RD5.grid" or event["native"] != "clean" or
                alternate != "raw" or
                set(event["candidates"]) != {"raw", "clean"}):
            raise ValueError("3D grid fixture lacks both complete grid plans")
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner3d_grid_wave.v1")
        if any(fields.get(key) != value for key, value in {
                "raw_inner_grid": "2", "output_grid": "65",
                "outer_grid": "1", "output_threads": "1",
                "output_unroll": "1", "sm_count": "128"}.items()):
            raise ValueError(f"3D grid fixture escaped the proved shape: {fields}")
        for row in report["multiplicity"]:
            if row["site"] != site and (
                    row["record"] != row["remote"] or
                    sum(row["record"].values()) != 1):
                raise ValueError(f"3D grid changed unrelated decision: {row}")
        if issues != [intended]:
            raise ValueError(f"unexpected 3D grid comparison issues: {issues}")
    elif require_inner3d_cta_target or require_inner3d_cta_full:
        expected_alternate = "t128" if require_inner3d_cta_full else "t64"
        expected_targets = tuple(range(8, 249, 8))
        if (site != "RD5.cta" or event["native"] != "t32" or
                alternate != expected_alternate or
                set(event["candidates"]) != {
                    f"t{target}" for target in expected_targets}):
            raise ValueError("3D CTA fixture lacks its complete hard-valid menu")
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner3d_cta_target.v1")
        if any(fields.get(key) != value for key, value in {
                "reduction_numel": "512", "iteration_numel": "32",
                "inner_numel": "128", "n_tensor_inputs": "1",
                "max_dtype_bits": "32", "vector_width": "4",
                "sm_count": "128", "logical_warp": "8",
                "hard_valid_targets": ",".join(map(str, expected_targets))
                }.items()):
            raise ValueError(f"3D CTA fixture escaped the proved shape: {fields}")
        for dependent_site in ("RD4", "RD5.iter_unroll", "RD5.width"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == dependent_site]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"3D CTA changed {dependent_site}: {rows}")
        if issues != [intended]:
            raise ValueError(f"unexpected 3D CTA comparison issues: {issues}")
    elif require_inner3d_warp_target:
        if (site != "RD5.warp" or event["native"] != "m8" or
                alternate != "m4" or
                set(event["candidates"]) != {
                    f"m{warp}" for warp in range(1, 17)}):
            raise ValueError("3D warp fixture lacks its complete logical menu")
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner3d_warp_target.v1")
        if any(fields.get(key) != value for key, value in {
                "reduction_numel": "512", "iteration_numel": "32",
                "inner_numel": "128", "n_tensor_inputs": "1",
                "max_dtype_bits": "32", "vector_width": "4",
                "sm_count": "128", "warp_size": "32"}.items()):
            raise ValueError(f"3D warp fixture escaped the proved shape: {fields}")
        cta_record, cta_remote, cta_issues = paired_state_rows(
            report, "RD5.cta")
        common = {key: fields[key] for key in (
            "reduction_numel", "iteration_numel", "inner_numel",
            "n_tensor_inputs", "max_dtype_bits", "vector_width",
            "sm_count", "max_threads_per_sm", "max_threads_per_block")}
        for row, side, choice, logical_warp, targets in (
                (cta_record, "record", "t32", "8", range(8, 249, 8)),
                (cta_remote, "remote", "t16", "4", range(4, 253, 4))):
            cta_fields = state_fields(row, "reduction.inner3d_cta_target.v1")
            expected_fields = {**common, "logical_warp": logical_warp,
                               "hard_valid_targets": ",".join(map(str, targets))}
            criteria = row["request"]["questions"]["RD5.cta"]["criteria"]
            if (row[side] != {choice: 1} or cta_fields != expected_fields or
                    set(criteria) != {f"t{target}" for target in targets}):
                raise ValueError(f"3D warp changed the dependent CTA plan: {row}")
        for dependent_site in ("RD4", "RD5.iter_unroll", "RD5.width"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == dependent_site]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"3D warp changed {dependent_site}: {rows}")
        if sorted(issues, key=str) != sorted([intended, *cta_issues], key=str):
            raise ValueError(f"unexpected 3D warp comparison issues: {issues}")
    elif require_inner_cta_envelope:
        if (site != "RD3.cta" or event["native"] != "t256" or
                alternate != "t64" or set(event["candidates"]) !=
                {"t64", "t128", "t256", "t512", "t1024"}):
            raise ValueError("short 2D fixture lacks its complete CTA menu")
        cta = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner_cta.v2")
        if any(cta.get(key) != value for key, value in {
                "reduction_numel": "64", "iteration_numel": "1024",
                "warp_size": "32", "max_threads_per_block": "1024",
                "max_threads_dim_y": "1024",
                "max_grid_x": str((1 << 31) - 1)}.items()):
            raise ValueError(f"CTA did not receive the short 2D plan: {cta}")
        after_vect = 64 // int(cta["vector_width"])
        if (64 % int(cta["vector_width"]) or after_vect > 64 or
                int(cta["target_inner_unroll"]) < 1):
            raise ValueError("short CTA fixture changed the selected width/unroll")
        warp_rows = [row for row in report["multiplicity"]
                     if row["site"] == "RD3.warp"]
        if len(warp_rows) != 1 or not warp_rows[0]["record"] or warp_rows[0]["remote"]:
            raise ValueError(f"expected only the native bounded warp plan: {warp_rows}")
        warp_fields = state_fields(warp_rows[0],
                                   "reduction.inner_warp_target.v1")
        if (warp_fields.get("target_threads_per_block") != "256" or
                warp_fields.get("after_vect") != str(after_vect)):
            raise ValueError("native warp state ignored its CTA target")
        expected = [intended, {"kind": "record_only_state", "site": "RD3.warp",
                               "request_sha256": warp_rows[0]["request_sha256"]}]
        for nested_site, tag in (
                ("RD3.grid", "reduction.inner_grid.v1"),
                ("RD3.iter_unroll", "reduction.inner_iter_unroll.v1")):
            before, after, pair_issues = paired_state_rows(report, nested_site)
            expected.extend(pair_issues)
            for row, side, target in ((before, "record", 256),
                                      (after, "remote", 64)):
                fields = state_fields(row, tag)
                bdimy = target // after_vect
                godim = (1024 + bdimy - 1) // bdimy
                checked = ({"godim": str(godim), "inner_remainder": "1"}
                           if nested_site == "RD3.grid" else
                           {"iteration_numel": "1024", "bdimy": str(bdimy),
                            "pre_unroll_godim": str(godim),
                            "grid_reduction_factor": "1"})
                if (row[side] != {"g1" if nested_site == "RD3.grid" else
                                 "u1": 1} or
                        any(fields.get(key) != value
                            for key, value in checked.items())):
                    raise ValueError(f"{nested_site} ignored selected CTA {target}: {row}")
        for row in report["multiplicity"]:
            if row["site"] not in {site, "RD3.warp", "RD3.grid",
                                   "RD3.iter_unroll"} and (
                    row["record"] != row["remote"] or
                    sum(row["record"].values()) != 1):
                raise ValueError(f"unrelated decision changed: {row}")
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected short CTA dependencies: {issues}")
    elif require_inner_warp_target:
        if (site != "RD3.warp" or event["native"] != "m8" or
                alternate != "m16" or
                set(event["candidates"]) != {
                    f"m{warp}" for warp in range(1, 17)}):
            raise ValueError("2D inner warp fixture lacks its complete target menu")
        warp = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner_warp_target.v1")
        if any(warp.get(key) != value for key, value in {
                "reduction_numel": "1025", "iteration_numel": "1024",
                "vector_width": "1", "after_vect": "1025",
                "target_threads_per_block": "256", "warp_size": "32",
                "max_grid_y": "65535"}.items()):
            raise ValueError(f"warp target did not receive the odd 2D plan: {warp}")
        if not 2 <= int(warp.get("target_inner_unroll", "0")) <= 4:
            raise ValueError(f"fixture missed the native remainder scan: {warp}")
        expected = [intended]
        for nested_site, native_x, selected_x in (
                ("RD3.remainder_x", None, None),
                ("RD3.remainder_u", "200", "192")):
            before, after, pair_issues = paired_state_rows(report, nested_site)
            expected.extend(pair_issues)
            tag = ("reduction.inner_remainder_x.v1" if native_x is None
                   else "reduction.inner_remainder_u.v1")
            record_fields = state_fields(before, tag)
            remote_fields = state_fields(after, tag)
            for fields, minimum, chosen in (
                    (record_fields, "8", native_x),
                    (remote_fields, "16", selected_x)):
                if (fields.get("min_warp_size") != minimum or
                        fields.get("after_vect") != "1025" or
                        fields.get("target_threads_per_block") != "256" or
                        fields.get("target_inner_unroll") !=
                        warp["target_inner_unroll"] or
                        (chosen is not None and
                         fields.get("chosen_bdimx") != chosen)):
                    raise ValueError(f"{nested_site} ignored selected warp: {fields}")
            if nested_site == "RD3.remainder_u" and (
                    before["record"] != after["remote"]):
                raise ValueError("inner unroll changed outside warp's x scan")
        for row in report["multiplicity"]:
            if row["site"].startswith("RD3.") and row["site"] not in {
                    site, "RD3.remainder_x", "RD3.remainder_u"} and (
                    row["record"] != row["remote"] or
                    sum(row["record"].values()) != 1):
                raise ValueError(f"unrelated 2D reduction decision changed: {row}")
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected inner warp dependencies: {issues}")
    elif site == "RD3.grid":
        if (event["native"] != "g2" or alternate != "g1" or
                set(event["candidates"]) != {"g1", "g2"}):
            raise ValueError("inner grid fixture lacks its two complete plans")
        dependent = [row for row in report["multiplicity"]
                     if row["site"] == "RD3.iter_unroll"]
        if (len(dependent) != 1 or dependent[0]["record"] or
                dependent[0]["remote"] != {"u1": 1}):
            raise ValueError(f"expected one single-grid output choice: {dependent}")
        fields = state_fields(dependent[0], "reduction.inner_iter_unroll.v1")
        if (fields.get("iteration_numel") != "32" or
                fields.get("bdimy") != "1" or
                fields.get("pre_unroll_godim") != "32" or
                fields.get("grid_reduction_factor") != "1" or
                fields.get("max_grid_x") != str((1 << 31) - 1) or
                dependent[0]["request"]["questions"]
                    ["RD3.iter_unroll"]["criteria"] != {
                        f"u{factor}":
                            f"Unroll {factor} output(s) per thread in the completed block plan"
                        for factor in (1, 2, 4)}):
            raise ValueError("dependent output unroll does not follow grid g1")
        expected = [intended, {
            "kind": "remote_only_state", "site": "RD3.iter_unroll",
            "request_sha256": dependent[0]["request_sha256"]}]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected inner grid comparison issues: {issues}")
    elif require_grid:
        if (site != "RD9.warp" or event["native"] != "warp16" or
                alternate != "warp32" or
                set(event["candidates"]) != {"warp16", "warp32"}):
            raise ValueError("outer warp test did not reach both hard-valid plans")
        for upstream in ("RD9.reg_budget", "RD9.width", "RD9.waves"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with warp: {rows}")
        cta_record, cta_remote, cta_issues = paired_state_rows(report, "RD9.cta")
        cta_before = state_fields(cta_record, "reduction.outer_cta.v1")
        cta_after = state_fields(cta_remote, "reduction.outer_cta.v1")
        if (cta_record["record"] != {"q4": 1} or
                cta_remote["remote"] != {"q4": 1} or
                cta_before.get("warp") != "16" or
                cta_after.get("warp") != "32" or
                {key: value for key, value in cta_before.items()
                 if key != "warp"} !=
                {key: value for key, value in cta_after.items()
                 if key != "warp"}):
            raise ValueError("CTA target did not receive selected warp")
        base_record, base_remote, base_issues = paired_state_rows(
            report, "RD9.grid_base")
        base_before = state_fields(base_record, "reduction.outer_grid_base.v1")
        base_after = state_fields(base_remote, "reduction.outer_grid_base.v1")
        for base, cta, row, side in (
                (base_before, cta_before, base_record, "record"),
                (base_after, cta_after, base_remote, "remote")):
            if (row[side] != {"target": 1} or base.get("cta_share") != "4" or
                    {key: value for key, value in base.items()
                     if key != "cta_share"} != cta):
                raise ValueError("baseline grid did not receive selected warp and CTA")
        unroll_issues = check_outer_unroll_dependency(
            report, "warp", "16", "32")
        rag_record, rag_remote, rag_issues = paired_state_rows(
            report, "RD9.ragged")
        stride_record, stride_remote, stride_issues = paired_state_rows(
            report, "RD9.stride")
        plan_record, plan_remote, plan_issues = paired_state_rows(report, "RD8")
        for rag, stride, plan, side, warp, block, output, raw in (
                (rag_record, stride_record, plan_record,
                 "record", "16", (16, 32), 33, 8),
                (rag_remote, stride_remote, plan_remote,
                 "remote", "32", (32, 16), 17, 16)):
            rag_fields = state_fields(rag, "reduction.outer_ragged.v1")
            stride_fields = outer_stride_fields(stride)
            plan_fields = state_fields(plan, "reduction.outer_plan.v1")
            cleaned = clean_outer_grid_waves(raw, output, int(rag_fields["sm_count"]))
            if (rag[side] != {"clean": 1} or
                    set(rag["request"]["questions"]["RD9.ragged"]
                        ["criteria"]) != {"clean", "raw"} or
                    (rag_fields.get("bdimx"), rag_fields.get("bdimy")) !=
                    tuple(map(str, block)) or
                    rag_fields.get("gidim") != str(output) or
                    rag_fields.get("base_grdim") != str(raw) or
                    stride_fields.get("min_warp_size") != warp or
                    any(rag_fields[key] != stride_fields[key]
                        for key in OUTER_RAGGED_FORWARD_KEYS) or
                    stride[side] != {f"g{cleaned}": 1} or
                    plan[side] != {"grid": 1} or
                    plan_fields.get("grid_gidim") != str(output) or
                    plan_fields.get("grid_grdim") != str(cleaned) or
                    plan_fields.get("grid_iter_unroll") !=
                    stride_fields["iter_unroll"]):
                raise ValueError("dependent grid plan ignored selected warp")
        before_plan = state_fields(plan_record, "reduction.outer_plan.v1")
        after_plan = state_fields(plan_remote, "reduction.outer_plan.v1")
        if ({key: value for key, value in before_plan.items()
             if key not in {"grid_gidim", "grid_grdim"}} !=
                {key: value for key, value in after_plan.items()
                 if key not in {"grid_gidim", "grid_grdim"}}):
            raise ValueError("warp changed unrelated RD8 plan measurements")
        expected = [intended, *cta_issues, *base_issues, *unroll_issues,
                    *rag_issues,
                    *stride_issues, *plan_issues]
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected outer warp issues: {issues}")
    elif require_inner_batch_cap:
        if (site != "IP5.batch" or alternate != "b12" or
                event["native"] == alternate):
            raise ValueError("register batch cap fixture lacks a non-native choice")
        lines = event["state"].splitlines()
        if not lines or lines[0] != "inner_persistent.reg_batch.v3":
            raise ValueError("expected the extended register batch state")
        fields = dict(line.split("=", 1) for line in lines[1:]
                      if not line.startswith("plan="))
        lower = int(fields["batch_min"])
        original = int(fields["native_batch_max"])
        upper = int(fields["batch_max"])
        if (fields["parallel_after_vectorize"] != "2048" or
                original != 10 or upper < 12 or
                set(event["candidates"]) != {
                    f"b{batch}" for batch in range(lower, upper + 1)} or
                int(event["native"][1:]) > original or
                issues != [intended]):
            raise ValueError("register batch cap did not preserve its baseline")
    elif require_inner_width:
        if (site != "IP5.width" or event["native"] != "w4" or
                alternate != "w1" or
                set(event["candidates"]) != {"w1", "w2", "w4"}):
            raise ValueError("2D register fixture lacks its checked width menu")
        expected = sorted((
            ("winner_mismatch", "IP5.width"),
            ("record_only_state", "IP5.targets"),
            ("remote_only_state", "IP5.targets"),
            ("record_only_state", "IP5.batch"),
            ("remote_only_state", "IP5.batch"),
        ))
        if sorted((issue["kind"], issue["site"]) for issue in issues) != expected:
            raise ValueError(f"unexpected dependent inner-width issues: {issues}")
        for nested_site, native_parallel, selected_parallel in (
                ("IP5.targets", 256, 1024),
                ("IP5.batch", 256, 1024)):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == nested_site]
            if len(rows) != 2:
                raise ValueError(f"expected replaced {nested_site} states: {rows}")
            for row in rows:
                record_count = sum(row["record"].values())
                remote_count = sum(row["remote"].values())
                if (record_count, remote_count) not in {(1, 0), (0, 1)}:
                    raise ValueError(f"duplicate/shared dependent state: {row}")
                fields = dict(line.split("=", 1) for line in
                              row["request"]["state"].splitlines()[1:]
                              if line.count("=") == 1 and
                              not line.startswith(("policy=", "plan=")))
                expected_parallel = (native_parallel if record_count
                                     else selected_parallel)
                if int(fields["parallel_after_vectorize"]) != expected_parallel:
                    raise ValueError(f"{nested_site} ignored selected width: {fields}")
        width_fields = dict(line.split("=", 1) for line in
                            event["state"].splitlines()[1:] if line.count("=") == 1
                            and not line.startswith("plan="))
        if (width_fields.get("max_legal_width") != "4" or
                width_fields.get("inner_numel") != "1024"):
            raise ValueError(f"unexpected layout-safe width bound: {width_fields}")
    elif require_inner_iter_unroll:
        if (site != "RD3.iter_unroll" or event["native"] != "u1" or
                alternate != "u2" or
                set(event["candidates"]) != {"u1", "u2", "u4"}):
            raise ValueError("inner reduction fixture lacks complete output tiles")
        fields = state_fields({"site": site, "request": {
            "state": event["state"]}}, "reduction.inner_iter_unroll.v1")
        if (fields.get("iteration_numel") != "1024" or
                fields.get("grid_reduction_factor") != "1" or
                fields.get("max_grid_x") != str((1 << 31) - 1) or
                int(fields.get("pre_unroll_godim", "0")) !=
                (1024 + int(fields.get("bdimy", "0")) - 1) //
                int(fields.get("bdimy", "1"))):
            raise ValueError(f"output unroll did not receive the block plan: {fields}")
        for row in report["multiplicity"]:
            if row["site"].startswith("RD3.") and row["site"] != site and (
                    row["record"] != row["remote"] or
                    sum(row["record"].values()) != 1):
                raise ValueError(f"upstream inner decision changed: {row}")
        if issues != [intended]:
            raise ValueError(f"unexpected output unroll comparison issues: {issues}")
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
    elif require_inner3d_register_overhead:
        from jev_shim import parse_inner3d_register_overhead_state

        if (site != "IP6.reg_overhead" or event["native"] != "r32" or
                alternate != "r40" or
                set(event["candidates"]) != {"r32", "r40"}):
            raise ValueError("inner 3D fixture lacks a live register alternate")
        state = parse_inner3d_register_overhead_state(event["state"])
        native, selected = state.plans[32], state.plans[40]
        if (state.has_exp_op != 1 or state.inner_numel != 5120 or
                state.iteration_numel != 256 or
                state.selected_vector_width != 4 or
                native.inner_batch <= 1 or
                selected.register_cap <= native.register_cap):
            raise ValueError("register choice did not change the complete plan")
        for upstream in ("IP6.width", "IP6.waves", "IP6.occupancy",
                         "IP6.warp", "IP6.outer_unroll"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed: {rows}")
        if issues != [intended]:
            raise ValueError(f"unexpected register-overhead comparison issues: {issues}")
    elif require_inner3d_outer_unroll:
        if (site != "IP6.outer_unroll" or event["native"] != "u4" or
                alternate != "u2" or
                not {"u4", "u2"} <= set(event["candidates"])):
            raise ValueError("inner 3D fixture lacks a legal outer unroll alternate")
        fields = dict(line.split("=", 1) for line in event["state"].splitlines()[1:]
                      if not line.startswith("plan"))
        if any(fields.get(key) != value for key, value in {
                "selected_vector_width": "4", "selected_wave_cap": "4",
                "selected_occupancy_target": "28", "selected_warp_target": "8",
                "pre_outer_threads_z": "1"}.items()):
            raise ValueError(f"outer unroll did not use selected 3D plan: {fields}")
        for upstream in ("IP6.width", "IP6.waves", "IP6.occupancy", "IP6.warp"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"upstream {upstream} changed with outer unroll: {rows}")
        before, after, dependent = paired_state_rows(
            report, "IP6.reg_overhead")
        left = state_fields(
            before, "inner_persistent.3d_register_overhead.v1")
        right = state_fields(
            after, "inner_persistent.3d_register_overhead.v1")
        if (left.get("selected_outer_unroll") != "4" or
                right.get("selected_outer_unroll") != "2" or
                before["record"] != {"r32": 1} or
                after["remote"] != {"r32": 1}):
            raise ValueError("register plan ignored selected outer unroll")
        if sorted(issues, key=str) != sorted([intended, *dependent], key=str):
            raise ValueError(f"unexpected outer unroll comparison issues: {issues}")
    elif require_inner3d_width:
        if (site != "IP6.width" or event["native"] != "w4" or
                alternate != "w2" or
                not {"w4", "w2"} <= set(event["candidates"])):
            raise ValueError("inner 3D fixture lacks a legal smaller width")
        state = dict(line.split("=", 1) for line in event["state"].splitlines()[1:]
                     if not line.startswith("plan"))
        if state.get("layout_max") != "4":
            raise ValueError(f"width did not preserve the layout maximum: {state}")
        expected = [intended]
        for nested_site, record_prefix, remote_prefix in (
                ("IP6.waves", "inner_persistent.3d_wave_cap.v1\n",
                 "inner_persistent.3d_wave_cap.v1\n"),
                ("IP6.occupancy", "inner_persistent.3d_occupancy_target.v1\n",
                 "inner_persistent.3d_occupancy_target.v2\n"),
                ("IP6.warp", "inner_persistent.3d_warp_target.v1\n",
                 "inner_persistent.3d_warp_target.v3\n"),
                ("IP6.outer_unroll", "inner_persistent.3d_outer_unroll.v1\n",
                 "inner_persistent.3d_outer_unroll.v1\n"),
                ("IP6.reg_overhead", "inner_persistent.3d_register_overhead.v1\n",
                 "inner_persistent.3d_register_overhead.v1\n")):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == nested_site]
            record_only = [row for row in rows
                           if sum(row["record"].values()) == 1 and
                           not row["remote"]]
            remote_only = [row for row in rows
                           if sum(row["remote"].values()) == 1 and
                           not row["record"]]
            if len(rows) != 2 or len(record_only) != 1 or len(remote_only) != 1:
                raise ValueError(f"expected one replaced {nested_site}: {rows}")
            if nested_site == "IP6.outer_unroll" and (
                    record_only[0]["record"] != {"u4": 1} or
                    remote_only[0]["remote"] != {"u4": 1}):
                raise ValueError("width unexpectedly changed outer unroll winner")
            for row, prefix, width in ((record_only[0], record_prefix, "4"),
                                       (remote_only[0], remote_prefix, "2")):
                text = row["request"]["state"]
                if not text.startswith(prefix):
                    raise ValueError(f"unexpected {nested_site} version: {text}")
                fields = dict(line.split("=", 1) for line in text.splitlines()[1:]
                              if not line.startswith("plan"))
                key = ("vector_width" if nested_site == "IP6.waves"
                       else "selected_vector_width")
                if width == "2" and fields.get(key) != width:
                    raise ValueError(f"{nested_site} omitted selected width: {fields}")
                if width == "4" and nested_site == "IP6.waves" and fields.get(key) != width:
                    raise ValueError(f"{nested_site} omitted native width: {fields}")
                if nested_site == "IP6.outer_unroll" and any(
                        fields.get(name) != expected for name, expected in {
                            "selected_wave_cap": "4",
                            "selected_occupancy_target": "28",
                            "selected_warp_target": "8"}.items()):
                    raise ValueError(f"outer unroll used a different plan: {fields}")
            expected.extend((
                {"kind": "record_only_state", "site": nested_site,
                 "request_sha256": record_only[0]["request_sha256"]},
                {"kind": "remote_only_state", "site": nested_site,
                 "request_sha256": remote_only[0]["request_sha256"]},
            ))
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected width-dependent issues: {issues}")
    elif require_inner3d_warp:
        if (site != "IP6.warp" or event["native"] != "warp8" or
                alternate != "warp16" or
                not {"warp8", "warp16"} <= set(event["candidates"])):
            raise ValueError("inner 3D fixture lacks a legal nonnative warp plan")
        waves = [row for row in report["multiplicity"]
                 if row["site"] == "IP6.waves"]
        if (len(waves) != 1 or waves[0]["record"] != {"waves4": 1} or
                waves[0]["remote"] != {"waves4": 1}):
            raise ValueError(f"3D wave target changed before warp Choice: {waves}")
        state = dict(line.split("=", 1) for line in
                     event["state"].splitlines()[1:])
        if state.get("selected_wave_cap") != "4":
            raise ValueError(f"warp plan used an unexpected wave cap: {state}")
        expected = [intended]
        for dependent_site, version in (
                ("IP6.outer_unroll", "inner_persistent.3d_outer_unroll.v1"),
                ("IP6.reg_overhead", "inner_persistent.3d_register_overhead.v1")):
            before, after, replaced = paired_state_rows(report, dependent_site)
            left = state_fields(before, version)
            right = state_fields(after, version)
            if (left.get("selected_warp_target") != "8" or
                    right.get("selected_warp_target") != "16"):
                raise ValueError(f"{dependent_site} ignored selected warp")
            expected.extend(replaced)
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected 3D warp comparison issues: {issues}")
    elif require_inner3d_occupancy:
        if (site != "IP6.occupancy" or event["native"] != "occ28" or
                alternate != "occ16" or
                not {"occ28", "occ16"} <= set(event["candidates"])):
            raise ValueError("inner 3D fixture lacks a legal occupancy alternate")
        waves = [row for row in report["multiplicity"]
                 if row["site"] == "IP6.waves"]
        if (len(waves) != 1 or waves[0]["record"] != {"waves4": 1} or
                waves[0]["remote"] != {"waves4": 1}):
            raise ValueError(f"3D wave target changed before occupancy Choice: {waves}")
        warps = [row for row in report["multiplicity"]
                 if row["site"] == "IP6.warp"]
        record_warp = [row for row in warps
                       if sum(row["record"].values()) == 1 and not row["remote"]]
        remote_warp = [row for row in warps
                       if sum(row["remote"].values()) == 1 and not row["record"]]
        if len(warps) != 2 or len(record_warp) != 1 or len(remote_warp) != 1:
            raise ValueError(f"expected exactly one replaced IP6.warp state: {warps}")
        record_state = record_warp[0]["request"]["state"]
        remote_state = remote_warp[0]["request"]["state"]
        if (not record_state.startswith("inner_persistent.3d_warp_target.v1\n") or
                not remote_state.startswith("inner_persistent.3d_warp_target.v2\n") or
                "\nselected_occupancy_target=16\n" not in remote_state):
            raise ValueError("IP6.warp did not receive the selected occupancy")
        expected = [intended,
            {"kind": "record_only_state", "site": "IP6.warp",
             "request_sha256": record_warp[0]["request_sha256"]},
            {"kind": "remote_only_state", "site": "IP6.warp",
             "request_sha256": remote_warp[0]["request_sha256"]},
        ]
        for dependent_site, version in (
                ("IP6.outer_unroll", "inner_persistent.3d_outer_unroll.v1"),
                ("IP6.reg_overhead", "inner_persistent.3d_register_overhead.v1")):
            before, after, replaced = paired_state_rows(report, dependent_site)
            left = state_fields(before, version)
            right = state_fields(after, version)
            if (left.get("selected_occupancy_target") != "28" or
                    right.get("selected_occupancy_target") != "16"):
                raise ValueError(f"{dependent_site} ignored selected occupancy")
            expected.extend(replaced)
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected dependent warp comparison issues: {issues}")
    elif require_iop3_inner_width:
        if site != "IOP3.inner_width" or alternate != "w2":
            raise ValueError("unexpected IOP3 inner-width alternate")
        check_iop3_inner_width_dependency(report, event, intended, issues)
    elif require_iop3_min_batch:
        if (site != "IOP3.min_batch" or event["native"] != "b2" or
                alternate != "b1" or
                not {"b1", "b2"} <= set(event["candidates"])):
            raise ValueError("inner-outer fixture lacks its alternate batch search")
        state_lines = event["state"].splitlines()
        fields = dict(line.split("=", 1) for line in state_lines[1:]
                      if line.count("=") == 1 and not line.startswith("plan="))
        plans = [line.removeprefix("plan=") for line in state_lines
                 if line.startswith("plan=")]
        if (state_lines[0] != "inner_outer_multi_wave.minimum_batch.v1" or
                fields.get("inner_dim") != "2048" or
                fields.get("outer_dim") != "1024" or
                fields.get("min_threads") != "128" or
                "1:512:3" not in plans or "2:256:2" not in plans):
            raise ValueError("batch state omitted the two complete CTA searches")
        reads = [row for row in report["multiplicity"]
                 if row["site"] == "IOP3.tmp_read_width"]
        writes = [row for row in report["multiplicity"]
                  if row["site"] == "IOP3.tmp_write_width"]
        if (len(reads) != 1 or reads[0]["record"] != {"w2": 1} or
                reads[0]["remote"] != {"w2": 1} or
                len(writes) != 1 or writes[0]["record"] != {"w4": 1} or
                writes[0]["remote"] != {"w4": 1}):
            raise ValueError("batch search changed a temporary vector width")
        dependent_issues, record_cta, remote_cta = (
            check_iop3_grid_rows_dependency(
                report, batch_pair=(2, 1), width_pair=(2, 2),
                cta_pair=("t256", "t512")))
        record_plans = iop3_cta_plan_fields(record_cta, 2)
        remote_plans = iop3_cta_plan_fields(remote_cta, 2)
        if ([plan[0] for plan in record_plans] != [256, 128] or
                [plan[0] for plan in remote_plans] != [512, 256, 128] or
                remote_plans[1:] != record_plans or
                record_cta["record"] != {"t256": 1} or
                remote_cta["remote"] != {"t512": 1}):
            raise ValueError("batch search did not extend the exact CTA menu")
        if sorted(issues, key=str) != sorted([intended, *dependent_issues], key=str):
            raise ValueError(f"unexpected minimum-batch dependencies: {issues}")
    elif require_iop3_grid_rows:
        if (site != "IOP3.grid_rows" or event["native"] != "r8" or
                alternate != "r4" or
                not {"r4", "r8"} <= set(event["candidates"])):
            raise ValueError("inner-outer fixture lacks checked rows alternatives")
        scalar, native_plans = iop3_rows_expected_cta(event["state"], 8)
        _, selected_plans = iop3_rows_expected_cta(event["state"], 4)
        upstream = dict(line.split("=", 1) for line in
                        event["state"].splitlines()[1:]
                        if not line.startswith("plan="))
        if (scalar["inner_dim"] != "2048" or scalar["outer_dim"] != "1024" or
                scalar["selected_outer_read_width"] != "2" or
                upstream["selected_min_batch"] != "2" or
                upstream["min_threads"] != "128" or
                upstream["max_threads"] != "256"):
            raise ValueError("rows fixture changed upstream selected values")
        if (len(native_plans) != 2 or len(selected_plans) != 2 or
                [p[5] for p in native_plans] != [128, 128] or
                [p[5] for p in selected_plans] != [256, 256] or
                [p[0] for p in native_plans] != [256, 128] or
                [p[0] for p in selected_plans] != [256, 128]):
            raise ValueError("rows alternate did not change the complete grid plans")
        for upstream in ("IOP3.min_batch", "IOP3.tmp_read_width"):
            rows = [row for row in report["multiplicity"]
                    if row["site"] == upstream]
            if (len(rows) != 1 or rows[0]["record"] != rows[0]["remote"] or
                    sum(rows[0]["record"].values()) != 1):
                raise ValueError(f"rows changed upstream {upstream}: {rows}")
        record_cta, remote_cta, cta_issues = paired_state_rows(
            report, "IOP3.cta")
        for row, expected, side in ((record_cta, native_plans, "record"),
                                    (remote_cta, selected_plans, "remote")):
            cta_lines = row["request"]["state"].splitlines()
            if cta_lines[0] != "inner_outer_multi_wave.cta.v2":
                raise ValueError("unexpected rows-dependent CTA version")
            actual_scalar = dict(line.split("=", 1) for line in cta_lines[1:]
                                 if not line.startswith("plan="))
            if (actual_scalar != scalar or
                    iop3_cta_plan_fields(row, 2) != expected or
                    row[side] != {"t256": 1}):
                raise ValueError("CTA did not materialize the selected rows plan")
        writes = [row for row in report["multiplicity"]
                  if row["site"] == "IOP3.tmp_write_width"]
        if (len(writes) != 1 or writes[0]["record"] != {"w4": 1} or
                writes[0]["remote"] != {"w4": 1}):
            raise ValueError("rows changed the downstream temporary write width")
        if sorted(issues, key=str) != sorted([intended, *cta_issues], key=str):
            raise ValueError(f"unexpected rows-dependent decisions: {issues}")
    elif require_iop3_tmp_write:
        if (site != "IOP3.tmp_write_width" or event["native"] != "w4" or
                alternate != "w2" or set(event["candidates"]) != {"w2", "w4"}):
            raise ValueError("inner-outer fixture lacks its checked write-width menu")
        ctas = [row for row in report["multiplicity"]
                if row["site"] == "IOP3.cta"]
        if (len(ctas) != 1 or ctas[0]["record"] != ctas[0]["remote"] or
                sum(ctas[0]["record"].values()) != 1):
            raise ValueError(f"write width changed the preceding CTA plan: {ctas}")
        if issues != [intended]:
            raise ValueError(f"unexpected inner-outer width issues: {issues}")
    elif require_iop3_tmp_read:
        if (site != "IOP3.tmp_read_width" or event["native"] != "w2" or
                alternate not in {"w1", "w4"} or
                set(event["candidates"]) != {"w1", "w2", "w4"}):
            raise ValueError("inner-outer fixture lacks its proved read-width menu")
        alternate_width = alternate[1:]
        expected = [intended]
        dependent_issues, record_cta, remote_cta = (
            check_iop3_grid_rows_dependency(
                report, batch_pair=(2, 2),
                width_pair=(2, int(alternate_width)),
                cta_pair=("t256", "t256")))
        expected.extend(dependent_issues)
        for nested_site, version, field in (
                ("IOP3.cta", "inner_outer_multi_wave.cta.v2",
                 "selected_outer_read_width"),
                ("IOP3.tmp_write_width",
                 "inner_outer_multi_wave.tmp_write_width.v2",
                 "outer_read_width")):
            if nested_site == "IOP3.cta":
                record_row, remote_row = record_cta, remote_cta
                pair_issues = []
            else:
                record_row, remote_row, pair_issues = paired_state_rows(
                    report, nested_site)
            if (record_row["record"] != remote_row["remote"] or
                    sum(record_row["record"].values()) != 1):
                raise ValueError(f"{nested_site} changed its winner: "
                                 f"{record_row}, {remote_row}")
            cta_plans: dict[str, list[tuple[int, ...]]] = {}
            for row, width in ((record_row, "2"),
                               (remote_row, alternate_width)):
                if nested_site == "IOP3.cta":
                    cta_plans[width] = iop3_cta_plan_fields(row, int(width))
                elif state_fields(row, version).get(field) != width:
                    raise ValueError(f"{nested_site} omitted selected read width")
            if nested_site == "IOP3.cta":
                record_plans = cta_plans["2"]
                remote_plans = cta_plans[alternate_width]
                invariant_columns = (0, 1, 2, 5, 6, 8, 9, 10, 13)
                if (len(record_plans) != len(remote_plans) or
                        any(tuple(record_plan[index] for index in invariant_columns)
                            != tuple(remote_plan[index] for index in invariant_columns)
                            for record_plan, remote_plan in
                            zip(record_plans, remote_plans))):
                    raise ValueError("IOP3.cta altered unrelated plan measurements")
            expected.extend(pair_issues)
        if sorted(issues, key=str) != sorted(expected, key=str):
            raise ValueError(f"unexpected read-width dependencies: {issues}")
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
    parser.add_argument("--require-outer-width", action="store_true")
    parser.add_argument("--require-outer-cta", action="store_true")
    parser.add_argument("--require-outer-ragged", action="store_true")
    parser.add_argument("--require-outer-grid-base", action="store_true")
    parser.add_argument("--require-outer-unroll", action="store_true")
    parser.add_argument("--require-inner-targets", action="store_true")
    parser.add_argument("--require-inner-iter-unroll", action="store_true")
    parser.add_argument("--require-inner-warp-target", action="store_true")
    parser.add_argument("--require-inner-cta-envelope", action="store_true")
    parser.add_argument("--require-inner3d-warp-target", action="store_true")
    parser.add_argument("--require-inner3d-cta-target", action="store_true")
    parser.add_argument("--require-inner3d-cta-full", action="store_true")
    parser.add_argument("--require-inner3d-grid-wave", action="store_true")
    parser.add_argument("--require-inner-width", action="store_true")
    parser.add_argument("--require-inner-batch-cap", action="store_true")
    parser.add_argument("--require-inner3d-warp", action="store_true")
    parser.add_argument("--require-inner3d-occupancy", action="store_true")
    parser.add_argument("--require-inner3d-width", action="store_true")
    parser.add_argument("--require-inner3d-outer-unroll", action="store_true")
    parser.add_argument("--require-inner3d-register-overhead", action="store_true")
    parser.add_argument("--require-iop3-min-batch", action="store_true")
    parser.add_argument("--require-iop3-grid-rows", action="store_true")
    parser.add_argument("--require-iop3-tmp-write", action="store_true")
    parser.add_argument("--require-iop3-tmp-read", action="store_true")
    parser.add_argument("--require-iop3-inner-width", action="store_true")
    parser.add_argument("--require-transpose-tile", action="store_true")
    parser.add_argument("--require-pointwise-1d-cta", action="store_true")
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
            require_outer_width=args.require_outer_width,
            require_outer_cta=args.require_outer_cta,
            require_outer_ragged=args.require_outer_ragged,
            require_outer_grid_base=args.require_outer_grid_base,
            require_outer_unroll=args.require_outer_unroll,
            require_inner_targets=args.require_inner_targets,
            require_inner_iter_unroll=args.require_inner_iter_unroll,
            require_inner_warp_target=args.require_inner_warp_target,
            require_inner_cta_envelope=args.require_inner_cta_envelope,
            require_inner3d_warp_target=args.require_inner3d_warp_target,
            require_inner3d_cta_target=args.require_inner3d_cta_target,
            require_inner3d_cta_full=args.require_inner3d_cta_full,
            require_inner3d_grid_wave=args.require_inner3d_grid_wave,
            require_inner_width=args.require_inner_width,
            require_inner_batch_cap=args.require_inner_batch_cap,
            require_inner3d_warp=args.require_inner3d_warp,
            require_inner3d_occupancy=args.require_inner3d_occupancy,
            require_inner3d_width=args.require_inner3d_width,
            require_inner3d_outer_unroll=args.require_inner3d_outer_unroll,
            require_inner3d_register_overhead=args.require_inner3d_register_overhead,
            require_iop3_min_batch=args.require_iop3_min_batch,
            require_iop3_grid_rows=args.require_iop3_grid_rows,
            require_iop3_tmp_write=args.require_iop3_tmp_write,
            require_iop3_tmp_read=args.require_iop3_tmp_read,
            require_iop3_inner_width=args.require_iop3_inner_width,
            require_transpose_tile=args.require_transpose_tile,
            require_pointwise_1d_cta=args.require_pointwise_1d_cta)
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
