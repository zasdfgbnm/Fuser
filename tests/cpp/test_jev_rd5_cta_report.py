#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay real RD5.warp numeric logs with the dependent RD5.cta decision."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


def load_warp_report() -> tuple[dict, list[dict], dict]:
    path = os.environ.get("JEV_RD5_WARP_MANIFEST")
    if not path:
        raise unittest.SkipTest("set JEV_RD5_WARP_MANIFEST to replay real logs")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {"site": "RD5.warp", "alternate": "m4",
               "case_key": manifest["case_key"],
               "require_inner3d_warp_target": True}
    return report, events, options


def cta_rows(report: dict) -> list[dict]:
    return [row for row in report["multiplicity"] if row["site"] == "RD5.cta"]


class Rd5WarpDependentCtaReportTest(unittest.TestCase):
    def test_replays_real_numeric_report(self) -> None:
        report, events, options = load_warp_report()
        summary = check_alternate_report(report, events, **options)
        self.assertEqual(summary["record_events"], 5)
        self.assertEqual(summary["remote_events"], 5)
        self.assertEqual(summary["server_successes"], 5)

    def test_missing_duplicate_or_wrong_cta_plan_is_rejected(self) -> None:
        report, events, options = load_warp_report()
        for defect in ("missing", "duplicate", "warp", "menu", "winner"):
            broken = deepcopy(report)
            rows = cta_rows(broken)
            remote = next(row for row in rows if row["remote"])
            if defect == "missing":
                broken["multiplicity"].remove(remote)
            elif defect == "duplicate":
                broken["multiplicity"].append(deepcopy(remote))
            elif defect == "warp":
                remote["request"]["state"] = remote["request"]["state"].replace(
                    "logical_warp=4", "logical_warp=8")
            elif defect == "menu":
                remote["request"]["questions"]["RD5.cta"]["criteria"].pop("t8")
            else:
                remote["remote"] = {"t32": 1}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                check_alternate_report(broken, events, **options)

    def test_extra_or_missing_issue_is_rejected(self) -> None:
        report, events, options = load_warp_report()
        for defect in ("missing", "extra", "duplicate"):
            broken = deepcopy(report)
            if defect == "missing":
                broken["issues"] = [issue for issue in broken["issues"]
                                    if not (issue["site"] == "RD5.cta" and
                                            issue["kind"] == "remote_only_state")]
            elif defect == "extra":
                broken["issues"].append({"kind": "record_only_state",
                                         "site": "RD5.width",
                                         "request_sha256": "unexpected"})
            else:
                broken["issues"].append(deepcopy(broken["issues"][0]))
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
