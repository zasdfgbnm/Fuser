#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay and challenge an actual RD9.grid_base numerical decision report."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


def load_report() -> tuple[dict, list, dict]:
    path = os.environ.get("JEV_OUTER_GRID_BASE_MANIFEST")
    if not path:
        raise unittest.SkipTest("set JEV_OUTER_GRID_BASE_MANIFEST to replay the CTest")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {
        "site": "RD9.grid_base", "alternate": "wave",
        "case_key": manifest["case_key"], "require_outer_grid_base": True,
    }
    return report, events, options


def remote_row(report: dict, site: str) -> dict:
    rows = [row for row in report["multiplicity"]
            if row["site"] == site and row["remote"] and not row["record"]]
    if len(rows) != 1:
        raise ValueError(f"expected one remote-only {site} row: {rows}")
    return rows[0]


class OuterGridBaseReportTest(unittest.TestCase):
    def test_real_record_and_remote_numeric_report(self) -> None:
        report, events, options = load_report()
        result = check_alternate_report(report, events, **options)
        self.assertEqual(result["native"], "target")
        self.assertEqual(result["alternate"], "wave")
        self.assertEqual(result["record_events"], 9)
        self.assertEqual(result["remote_events"], 9)
        self.assertEqual(result["server_successes"], 9)

    def test_rejects_changed_dependent_grid_state(self) -> None:
        report, events, options = load_report()
        for site, old, new in (
                ("RD9.ragged", "base_grdim=4", "base_grdim=5"),
                ("RD9.stride", "base_grdim=4", "base_grdim=5"),
                ("RD8", "grid_grdim=3", "grid_grdim=4")):
            broken = deepcopy(report)
            row = remote_row(broken, site)
            row["request"]["state"] = row["request"]["state"].replace(old, new)
            self.assertNotEqual(row["request"]["state"],
                                remote_row(report, site)["request"]["state"])
            with self.subTest(site=site), self.assertRaises(ValueError):
                check_alternate_report(broken, events, **options)

    def test_rejects_missing_or_extra_dependent_issue(self) -> None:
        report, events, options = load_report()
        broken = deepcopy(report)
        broken["issues"] = [issue for issue in broken["issues"]
                            if issue["site"] != "RD9.stride"]
        with self.assertRaises(ValueError):
            check_alternate_report(broken, events, **options)
        broken = deepcopy(report)
        broken["issues"].append({"kind": "winner_mismatch", "site": "RD9.warp",
                                 "request_sha256": "forged"})
        with self.assertRaises(ValueError):
            check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
