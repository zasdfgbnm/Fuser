# SPDX-License-Identifier: BSD-3-Clause
"""Strictly group Pointwise Record Choices by returned computeHeuristics plan.

The ordinary candidate may be evaluated after a TMA candidate, even when the
TMA plan is returned. A per-site Choice is therefore not a backend outcome.
This module is diagnostic-only and does not participate in scheduling.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


SCHEMA = "nvfuser.jev.pointwise.outcome.v1"
FINAL_FIELDS = frozenset({
    "scheduler_type", "tag", "break_point", "split_block", "symbolic_1d_cta",
    "split_grid_y_dim", "flip_grid_binding", "vectorization_factor",
    "vectorize_casts", "unroll_factor_outer", "unroll_factor_inner",
    "use_tma_load", "use_tma_store", "tma_domain_inner", "tma_tile_outer",
    "tma_tile_inner",
})
BOOLEAN_FINAL_FIELDS = frozenset({
    "split_block", "symbolic_1d_cta", "split_grid_y_dim",
    "flip_grid_binding", "vectorize_casts", "use_tma_load", "use_tma_store",
})
TEXT_FINAL_FIELDS = frozenset({"scheduler_type", "tag"})
MODIFIED_ONLY_FINAL_FIELDS = frozenset({"scheduler_type", "symbolic_1d_cta"})
NATIVE_FINAL_FIELDS = FINAL_FIELDS - MODIFIED_ONLY_FINAL_FIELDS
LAUNCH_FIELDS = frozenset({
    "grid_x_raw", "grid_y_raw", "grid_z_raw", "block_x_raw", "block_y_raw",
    "block_z_raw", "dynamic_smem_bytes",
})
COMPILE_FIELDS = frozenset({
    "index_type", "maxrregcount", "enable_magic_zero", "enable_ptxas_verbose",
    "device", "include_paths", "block_x", "block_y", "block_z",
    "compute_block_x", "compute_block_y", "compute_block_z",
})


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def read_jsonl_strict(path: Path) -> list[dict]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"{path}: incomplete final JSONL row")
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"{path}:{number}: duplicate key {key}")
                result[key] = value
            return result

        try:
            row = json.loads(line, object_pairs_hook=unique,
                             parse_constant=_reject_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(row)
    return rows


def group_record_invocations(
    decision_rows: list[dict], outcome_rows: list[dict]
) -> dict[str, dict]:
    """Require one complete terminal snapshot per Pointwise invocation.

    Returns realized and speculative Choice rows separately. `event_index`
    merges the two files into one strict begin/site/skip/final sequence.
    Callers must compare native decisions to realized rows only; PW1 is
    optional when the ordinary candidate could not be materialized.
    """
    invocations: dict[str, dict] = defaultdict(
        lambda: {"begin": None, "final": None, "choices": [],
                 "coverage_skips": [], "events": []})
    last_outcome_index: dict[str, int] = {}
    for row in outcome_rows:
        if row.get("schema") != SCHEMA or row.get("mode") != "record":
            raise ValueError("unexpected Pointwise outcome schema or mode")
        identifier = row.get("invocation_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Pointwise outcome lacks invocation_id")
        index = row.get("event_index")
        if type(index) is not int or index < 0:
            raise ValueError(f"Pointwise outcome {identifier} lacks event_index")
        if index <= last_outcome_index.get(identifier, -1):
            raise ValueError(f"Pointwise outcome {identifier} is out of order")
        last_outcome_index[identifier] = index
        event = row.get("kind")
        if event not in ("begin", "final", "coverage_skip"):
            raise ValueError(f"unexpected Pointwise outcome kind {event!r}")
        if not isinstance(row.get("case_key"), str):
            raise ValueError("Pointwise outcome lacks case_key")
        invocations[identifier]["events"].append((index, event))
        if event == "coverage_skip":
            if not isinstance(row.get("site"), str) or not row["site"].startswith("PW"):
                raise ValueError("Pointwise coverage skip lacks a PW site")
            invocations[identifier]["coverage_skips"].append(row)
        elif invocations[identifier][event] is not None:
            raise ValueError(f"duplicate {event} for Pointwise invocation {identifier}")
        else:
            invocations[identifier][event] = row

    last_decision_index: dict[str, int] = {}
    for row in decision_rows:
        site = row.get("site", "")
        if not isinstance(site, str) or not site.startswith("PW"):
            continue
        if row.get("kind") == "coverage_skip":
            continue  # The correlated copy is in the outcome stream.
        if row.get("mode") != "record":
            raise ValueError("Pointwise Choice must be in Record mode")
        identifier = row.get("invocation_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"Pointwise Choice {site} lacks invocation_id")
        index = row.get("event_index")
        if type(index) is not int or index <= 0:
            raise ValueError(f"Pointwise Choice {site} lacks event_index")
        if index <= last_decision_index.get(identifier, -1):
            raise ValueError(f"Pointwise Choice {identifier} is out of order")
        last_decision_index[identifier] = index
        invocations[identifier]["choices"].append(row)
        invocations[identifier]["events"].append((index, "choice"))

    result = {}
    for identifier, item in invocations.items():
        begin, final = item["begin"], item["final"]
        if begin is None or final is None:
            raise ValueError(f"incomplete Pointwise invocation {identifier}")
        events = sorted(item["events"])
        if ([index for index, _ in events] != list(range(len(events))) or
                events[0][1] != "begin" or events[-1][1] != "final"):
            raise ValueError(f"Pointwise invocation {identifier} has no terminal order")
        if begin["case_key"] != final["case_key"]:
            raise ValueError(f"Pointwise invocation {identifier} changed case_key")
        for row in item["choices"] + item["coverage_skips"]:
            if row.get("case_key") != begin["case_key"]:
                raise ValueError(f"Pointwise invocation {identifier} has mixed case keys")
        for name, fields in (("final_params", FINAL_FIELDS),
                             ("launch", LAUNCH_FIELDS), ("compile", COMPILE_FIELDS)):
            value = final.get(name)
            if not isinstance(value, dict) or not fields <= value.keys():
                raise ValueError(f"Pointwise invocation {identifier} lacks {name} fields")
        params = final["final_params"]
        if params["scheduler_type"] != "PointWise" or not isinstance(params["tag"], str):
            raise ValueError(f"Pointwise invocation {identifier} has invalid identity")
        for field in BOOLEAN_FINAL_FIELDS:
            if type(params[field]) is not bool:
                raise ValueError(f"Pointwise invocation {identifier} has invalid {field}")
        for field in FINAL_FIELDS - BOOLEAN_FINAL_FIELDS - TEXT_FINAL_FIELDS:
            if type(params[field]) is not int:
                raise ValueError(f"Pointwise invocation {identifier} has invalid {field}")
        if any(type(value) is not int for value in final["launch"].values()):
            raise ValueError(f"Pointwise invocation {identifier} has invalid launch dimensions")
        backend = final.get("backend")
        if (backend not in ("tma", "ordinary", "empty") or
                type(params["use_tma_load"]) is not bool or
                (backend == "tma") != params["use_tma_load"]):
            raise ValueError(f"Pointwise invocation {identifier} has inconsistent backend")
        if backend == "empty" and item["choices"]:
            raise ValueError(f"empty Pointwise invocation {identifier} has Choices")
        realized, speculative = [], []
        for row in item["choices"]:
            site = row["site"]
            if site == "PW1":
                if row.get("selected") != backend:
                    raise ValueError(f"PW1 disagrees with returned backend in {identifier}")
                realized.append(row)
            elif site.startswith(("PW2", "PW3")):
                (speculative if backend == "tma" else realized).append(row)
            elif site.startswith("PW4"):
                (realized if backend == "tma" else speculative).append(row)
            else:
                raise ValueError(f"unknown Pointwise site {site}")
        result[identifier] = {
            "case_key": begin["case_key"], "backend": backend,
            "final": final, "realized_choices": realized,
            "speculative_choices": speculative,
            "coverage_skips": item["coverage_skips"],
        }
    return result


def project_record_final_for_native(final: dict) -> dict:
    """Project the modified snapshot onto the exact vanilla observer schema.

    Vanilla PointwiseParams has no `symbolic_1d_cta` member, and the vanilla
    observer does not emit `scheduler_type`. All other returned fields, plus
    raw launch and compile fields, must agree with the native diagnostic.
    """
    params = final["final_params"]
    return {
        "final_params": {key: params[key] for key in sorted(NATIVE_FINAL_FIELDS)},
        "launch": {key: final["launch"][key] for key in sorted(LAUNCH_FIELDS)},
        "compile": {key: final["compile"][key] for key in sorted(COMPILE_FIELDS)},
    }
