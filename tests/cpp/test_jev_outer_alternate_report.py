#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Replay numeric A/B logs, then prove the strict comparison rejects corruptions.

Set JEV_OUTER_WIDTH_MANIFEST and JEV_OUTER_CTA_MANIFEST to the manifests
written by the corresponding governed CTests. No GPU work is done here.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

from jev_alternate_numeric import check_alternate_report


CASES = {
    "width": ("JEV_OUTER_WIDTH_MANIFEST", "RD9.width", "w1",
              "require_outer_width"),
    "cta": ("JEV_OUTER_CTA_MANIFEST", "RD9.cta", "q8",
            "require_outer_cta"),
}


def load_case(kind: str) -> tuple[dict, list, dict]:
    variable, site, alternate, flag = CASES[kind]
    path = os.environ.get(variable)
    if not path:
        raise unittest.SkipTest(f"set {variable} to replay the numeric decision log")
    manifest = json.loads(Path(path).read_text())
    report = json.loads(Path(manifest["decision_comparison"]).read_text())
    events = json.loads(Path(manifest["forced_choice_log"]).read_text())
    options = {"site": site, "alternate": alternate,
               "case_key": manifest["case_key"], flag: True}
    return report, events, options


def rows(report: dict, site: str) -> list[dict]:
    return [row for row in report["multiplicity"] if row["site"] == site]


class OuterAlternateReportTest(unittest.TestCase):
    def test_replay_real_numeric_reports(self) -> None:
        for kind in CASES:
            with self.subTest(kind=kind):
                report, events, options = load_case(kind)
                result = check_alternate_report(report, events, **options)
                expected_events = 7 + bool(rows(report, "RD9.ragged"))
                self.assertEqual(result["record_events"], expected_events)
                self.assertEqual(result["remote_events"], expected_events)
                self.assertEqual(result["server_successes"], expected_events)

    def test_missing_or_duplicate_dependent_state_is_rejected(self) -> None:
        for kind, site in (("width", "RD9.cta"), ("width", "RD9.stride"),
                           ("width", "RD8"), ("cta", "RD9.stride"),
                           ("cta", "RD8"), ("width", "RD9.ragged"),
                           ("cta", "RD9.ragged")):
            report, events, options = load_case(kind)
            if not rows(report, site):
                continue  # Prior v1 fixtures have no ragged Choice.
            with self.subTest(kind=kind, site=site, defect="missing"):
                broken = deepcopy(report)
                broken["multiplicity"].remove(rows(broken, site)[0])
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)
            with self.subTest(kind=kind, site=site, defect="duplicate"):
                broken = deepcopy(report)
                broken["multiplicity"].append(deepcopy(rows(broken, site)[0]))
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)

    def test_downstream_winners_and_selected_plan_are_checked(self) -> None:
        for kind, site, choice in (("width", "RD9.cta", "q2"),
                                   ("width", "RD8", "block"),
                                   ("cta", "RD8", "block")):
            report, events, options = load_case(kind)
            broken = deepcopy(report)
            remote = next(row for row in rows(broken, site) if row["remote"])
            remote["remote"] = {choice: 1}
            with self.subTest(kind=kind, site=site):
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)
        for kind in CASES:
            report, events, options = load_case(kind)
            broken = deepcopy(report)
            remote = next(row for row in rows(broken, "RD8") if row["remote"])
            remote["request"]["state"] = remote["request"]["state"].replace(
                "grid_grdim=", "grid_grdim=999")
            with self.subTest(kind=kind, defect="wrong_grid_plan"):
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)
        for kind, site, old, new in (
                ("width", "RD9.cta", "vector_width=1", "vector_width=2"),
                ("cta", "RD9.stride", "bdimy=16", "bdimy=32")):
            report, events, options = load_case(kind)
            broken = deepcopy(report)
            remote = next(row for row in rows(broken, site) if row["remote"])
            remote["request"]["state"] = remote["request"]["state"].replace(
                old, new)
            with self.subTest(kind=kind, defect="wrong_dependent_state"):
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)

    def test_unrelated_or_unpaired_issues_are_rejected(self) -> None:
        for kind in CASES:
            report, events, options = load_case(kind)
            for issue in ({"kind": "record_only_state", "site": "RD9.warp",
                           "request_sha256": "unrelated"},
                          deepcopy(report["issues"][0])):
                broken = deepcopy(report)
                broken["issues"].append(issue)
                with self.subTest(kind=kind, issue=issue):
                    with self.assertRaises(ValueError):
                        check_alternate_report(broken, events, **options)
            broken = deepcopy(report)
            broken["issues"].remove(next(issue for issue in broken["issues"]
                if issue["site"] == "RD8" and issue["kind"] == "remote_only_state"))
            with self.subTest(kind=kind, defect="missing_pair_issue"):
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)

    def test_v2_stride_policy_and_ragged_state_are_checked(self) -> None:
        for kind in CASES:
            report, events, options = load_case(kind)
            if not rows(report, "RD9.ragged"):
                continue  # This check applies to the v2 numeric fixture.
            for old, new in (("ragged_policy=clean", "ragged_policy=raw"),
                             ("ragged_policy=clean", ""),
                             ("outer_grid_stride.v2", "outer_grid_stride.v1")):
                broken = deepcopy(report)
                remote = next(row for row in rows(broken, "RD9.stride")
                              if row["remote"])
                remote["request"]["state"] = remote["request"]["state"].replace(
                    old, new)
                with self.subTest(kind=kind, mutation=(old, new)):
                    with self.assertRaises(ValueError):
                        check_alternate_report(broken, events, **options)
            for site, mutation in (("RD9.ragged", "winner"),
                                   ("RD9.ragged", "state")):
                broken = deepcopy(report)
                remote = next(row for row in rows(broken, site) if row["remote"])
                if mutation == "winner":
                    remote["remote"] = {"raw": 1}
                else:
                    remote["request"]["state"] = remote["request"]["state"].replace(
                        "base_grdim=", "base_grdim=999")
                with self.subTest(kind=kind, mutation=mutation):
                    with self.assertRaises(ValueError):
                        check_alternate_report(broken, events, **options)
            broken = deepcopy(report)
            broken["issues"].remove(next(issue for issue in broken["issues"]
                if issue["site"] == "RD9.ragged" and
                issue["kind"] == "remote_only_state"))
            with self.subTest(kind=kind, mutation="missing_ragged_issue"):
                with self.assertRaises(ValueError):
                    check_alternate_report(broken, events, **options)


if __name__ == "__main__":
    unittest.main()
