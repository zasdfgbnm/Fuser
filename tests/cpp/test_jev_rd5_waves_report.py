#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay equivalent RD5 wave caps after forced warp and work split."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


CASES = (
    ("JEV_RD5_WARP_MANIFEST", "RD5.warp", "m4",
     "require_inner3d_warp_target", 4),
    ("JEV_RD5_WORK_SPLIT_MANIFEST", "RD5.work_split", "iter_first",
     "require_reduction_inner3d_work_split", 5),
)


def load_case(case: tuple[str, str, str, str, int]
              ) -> tuple[dict, list[dict], dict, int]:
    variable, site, alternate, required, count = case
    path = os.environ.get(variable)
    if not path:
        raise unittest.SkipTest(f"set {variable} to replay real numeric logs")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {"site": site, "alternate": alternate,
               "case_key": manifest["case_key"], required: True,
               "trace_paths": tuple(Path(run["decision_log"])
                                    for run in manifest["runs"])}
    return report, events, options, count


def wave_rows(report: dict) -> list[dict]:
    return [row for row in report["multiplicity"]
            if row["site"] == "RD5.waves"]


class Rd5WaveDependentReportTest(unittest.TestCase):
    def test_replays_both_suppressed_equivalent_wave_plans(self) -> None:
        for case in CASES:
            with self.subTest(site=case[1]):
                report, events, options, count = load_case(case)
                summary = check_alternate_report(report, events, **options)
                self.assertEqual(summary["record_events"], count)
                self.assertEqual(summary["remote_events"], count)
                self.assertEqual(summary["server_successes"], count)

    def test_forged_wave_request_or_missing_cta_issue_is_rejected(self) -> None:
        for case in CASES:
            report, events, options, _ = load_case(case)
            self.assertEqual(wave_rows(report), [])
            for defect in ("extra_wave_request", "extra_wave_issue",
                           "missing_cta_issue"):
                with self.subTest(site=case[1], defect=defect):
                    broken = deepcopy(report)
                    if defect == "extra_wave_request":
                        broken["multiplicity"].append({
                            "site": "RD5.waves", "record": {},
                            "remote": {"waves4": 1},
                            "request_sha256": "forged-wave",
                            "request": {"state": "forged"},
                        })
                    elif defect == "extra_wave_issue":
                        broken["issues"].append({
                            "kind": "remote_only_state", "site": "RD5.waves",
                            "request_sha256": "forged-wave",
                        })
                    else:
                        broken["issues"] = [
                            issue for issue in broken["issues"]
                            if not (issue["site"] == "RD5.cta" and
                                    issue["kind"] == "remote_only_state")]
                    with self.assertRaises(ValueError):
                        check_alternate_report(broken, events, **options)

    def test_selected_cta_request_is_required(self) -> None:
        for case in CASES:
            with self.subTest(site=case[1]):
                report, events, options, _ = load_case(case)
                broken = deepcopy(report)
                broken["multiplicity"] = [
                    row for row in broken["multiplicity"]
                    if not (row["site"] == "RD5.cta" and row["remote"])]
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
