#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay a forced RD9.warp report with its dependent baseline-grid Choice."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


def load_report() -> tuple[dict, list, dict]:
    path = os.environ.get("JEV_OUTER_WARP_MANIFEST")
    if not path:
        raise unittest.SkipTest("set JEV_OUTER_WARP_MANIFEST to replay the CTest")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {
        "site": "RD9.warp", "alternate": "warp32",
        "case_key": manifest["case_key"], "require_grid": True,
    }
    return report, events, options


def remote_row(report: dict, site: str) -> dict:
    rows = [row for row in report["multiplicity"]
            if row["site"] == site and row["remote"] and not row["record"]]
    if len(rows) != 1:
        raise ValueError(f"expected one remote-only {site} row: {rows}")
    return rows[0]


class OuterWarpReportTest(unittest.TestCase):
    def test_real_alternate_preserves_numerics_and_all_decisions(self) -> None:
        report, events, options = load_report()
        result = check_alternate_report(report, events, **options)
        self.assertEqual(result["native"], "warp16")
        self.assertEqual(result["alternate"], "warp32")
        self.assertEqual(result["record_events"], 10)
        self.assertEqual(result["remote_events"], 10)
        self.assertEqual(result["server_successes"], 10)

    def test_dependent_baseline_state_and_issue_are_required(self) -> None:
        report, events, options = load_report()
        broken = deepcopy(report)
        row = remote_row(broken, "RD9.grid_base")
        row["request"]["state"] = row["request"]["state"].replace(
            "warp=32", "warp=16")
        with self.assertRaises(ValueError):
            check_alternate_report(broken, events, **options)
        broken = deepcopy(report)
        broken["issues"] = [issue for issue in broken["issues"]
                            if issue["site"] != "RD9.grid_base"]
        with self.assertRaises(ValueError):
            check_alternate_report(broken, events, **options)

    def test_dependent_ragged_and_final_grid_are_required(self) -> None:
        report, events, options = load_report()
        for site, old, new in (
                ("RD9.unroll", "warp=32", "warp=16"),
                ("RD9.ragged", "base_grdim=16", "base_grdim=8"),
                ("RD9.stride", "min_warp_size=32", "min_warp_size=16"),
                ("RD8", "grid_grdim=15", "grid_grdim=16")):
            broken = deepcopy(report)
            row = remote_row(broken, site)
            row["request"]["state"] = row["request"]["state"].replace(old, new)
            self.assertNotEqual(row["request"]["state"],
                                remote_row(report, site)["request"]["state"])
            with self.subTest(site=site), self.assertRaises(ValueError):
                check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
