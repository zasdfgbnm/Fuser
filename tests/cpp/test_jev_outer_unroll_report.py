#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay and challenge the SM89 RD9.unroll numeric decision report."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


def load_report() -> tuple[dict, list, dict]:
    path = os.environ.get("JEV_OUTER_UNROLL_MANIFEST")
    if not path:
        raise unittest.SkipTest("set JEV_OUTER_UNROLL_MANIFEST to replay the CTest")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {
        "site": "RD9.unroll", "alternate": "u1",
        "case_key": manifest["case_key"], "require_outer_unroll": True,
    }
    return report, events, options


def remote_row(report: dict, site: str) -> dict:
    rows = [row for row in report["multiplicity"]
            if row["site"] == site and row["remote"] and not row["record"]]
    if len(rows) != 1:
        raise ValueError(f"expected one remote-only {site} row: {rows}")
    return rows[0]


class OuterUnrollReportTest(unittest.TestCase):
    def test_real_record_and_remote_numeric_report(self) -> None:
        report, events, options = load_report()
        result = check_alternate_report(report, events, **options)
        self.assertEqual(result["native"], "u4")
        self.assertEqual(result["alternate"], "u1")
        self.assertEqual(result["candidates"], ["u1", "u2", "u4"])
        self.assertEqual(result["remote_events"], result["server_successes"])

    def test_rejects_missing_dependent_state_or_changed_plan(self) -> None:
        report, events, options = load_report()
        broken = deepcopy(report)
        broken["issues"] = [issue for issue in broken["issues"]
                            if issue["site"] != "RD9.stride"]
        with self.assertRaises(ValueError):
            check_alternate_report(broken, events, **options)
        for site, old, new in (
                ("RD9.ragged", "iter_unroll=1", "iter_unroll=2"),
                ("RD9.stride", "redu_unroll=32", "redu_unroll=16"),
                ("RD8", "grid_iter_unroll=1", "grid_iter_unroll=2")):
            broken = deepcopy(report)
            row = remote_row(broken, site)
            row["request"]["state"] = row["request"]["state"].replace(
                old, new)
            self.assertNotEqual(row["request"]["state"],
                                remote_row(report, site)["request"]["state"])
            with self.subTest(site=site), self.assertRaises(ValueError):
                check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
