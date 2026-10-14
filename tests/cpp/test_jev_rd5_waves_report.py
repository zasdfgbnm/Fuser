#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay RD5.waves dependencies in the forced warp and work-split reports."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


CASES = (
    ("JEV_RD5_WARP_MANIFEST", "RD5.warp", "m4",
     "require_inner3d_warp_target", 6),
    ("JEV_RD5_WORK_SPLIT_MANIFEST", "RD5.work_split", "iter_first",
     "require_reduction_inner3d_work_split", 7),
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
               "case_key": manifest["case_key"], required: True}
    return report, events, options, count


def wave_rows(report: dict) -> list[dict]:
    return [row for row in report["multiplicity"]
            if row["site"] == "RD5.waves"]


class Rd5WaveDependentReportTest(unittest.TestCase):
    def test_replays_both_complete_dependent_wave_plans(self) -> None:
        for case in CASES:
            with self.subTest(site=case[1]):
                report, events, options, count = load_case(case)
                summary = check_alternate_report(report, events, **options)
                self.assertEqual(summary["record_events"], count)
                self.assertEqual(summary["remote_events"], count)
                self.assertEqual(summary["server_successes"], count)

    def test_forged_dependent_state_menu_or_winner_is_rejected(self) -> None:
        for case in CASES:
            report, events, options, _ = load_case(case)
            for defect in ("state", "menu", "winner"):
                with self.subTest(site=case[1], defect=defect):
                    broken = deepcopy(report)
                    remote = next(row for row in wave_rows(broken)
                                  if row["remote"])
                    if defect == "state":
                        state = remote["request"]["state"]
                        remote["request"]["state"] = (
                            state.replace("logical_warp=4", "logical_warp=8")
                            if case[1] == "RD5.warp" else
                            state.replace("work_split=iter_first",
                                          "work_split=unroll_first"))
                    elif defect == "menu":
                        remote["request"]["questions"]["RD5.waves"][
                            "criteria"].pop("waves2")
                    else:
                        remote["remote"] = {"waves8": 1}
                    with self.assertRaises(ValueError):
                        check_alternate_report(broken, events, **options)

    def test_missing_dependent_state_issue_is_rejected(self) -> None:
        for case in CASES:
            with self.subTest(site=case[1]):
                report, events, options, _ = load_case(case)
                broken = deepcopy(report)
                broken["issues"] = [
                    issue for issue in broken["issues"]
                    if not (issue["site"] == "RD5.waves" and
                            issue["kind"] == "remote_only_state")]
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
