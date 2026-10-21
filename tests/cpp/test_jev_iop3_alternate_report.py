"""Regression checks for IOP3 read-width-dependent decision logs."""
from __future__ import annotations

import unittest

from jev_alternate_numeric import check_alternate_report, state_fields
from jev_iop3_rows_report_fixture import (cta_state as complete_cta_state,
                                          rows_request)

CASE = "jev_alternate_numeric_inner_outer_tmp_read_width"


def cta_state(read: int) -> str:
    return complete_cta_state(2, read)


def write_state(read: int) -> str:
    return ("inner_outer_multi_wave.tmp_write_width.v2\n"
            "inner_dim=2048\ninner_vector_width=4\n"
            f"outer_read_width={read}\ntmp_dtype_bits=32")


def row(site: str, hash_: str, state: str,
        record: dict[str, int], remote: dict[str, int]) -> dict:
    return {"site": site, "request_sha256": hash_,
            "request": {"state": state},
            "record": record, "remote": remote}


def fixture(alternate: str = "w1") -> tuple[dict, list[dict]]:
    remote_width = int(alternate[1:])
    entries = [
        row("IOP3.tmp_read_width", "read", "read-state",
            {"w2": 1}, {alternate: 1}),
        row("IOP3.grid_rows", "rows-record", "", {"r8": 1}, {}),
        row("IOP3.grid_rows", "rows-remote", "", {}, {"r8": 1}),
        row("IOP3.cta", "cta-record", cta_state(2), {"t256": 1}, {}),
        row("IOP3.cta", "cta-remote", cta_state(remote_width),
            {}, {"t256": 1}),
        row("IOP3.tmp_write_width", "write-record", write_state(2),
            {"w4": 1}, {}),
        row("IOP3.tmp_write_width", "write-remote", write_state(remote_width),
            {}, {"w4": 1}),
    ]
    entries[1]["request"] = rows_request(2, 2)
    entries[2]["request"] = rows_request(2, remote_width)
    issues = [{"kind": "winner_mismatch", "site": "IOP3.tmp_read_width",
               "request_sha256": "read"}]
    for site, prefix in (("IOP3.grid_rows", "rows"),
                         ("IOP3.cta", "cta"),
                         ("IOP3.tmp_write_width", "write")):
        for side in ("record", "remote"):
            issues.append({"kind": f"{side}_only_state", "site": site,
                           "request_sha256": f"{prefix}-{side}"})
    report = {"multiplicity": entries, "issues": issues,
              "summary": {"remote_events": 4, "server_successes": 4,
                          "record_events": 4},
              "case_coverage": {CASE: {}}}
    event = {"site": "IOP3.tmp_read_width", "native": "w2",
             "candidates": ["w1", "w2", "w4"], "forced": True,
             "request_sha256": "read"}
    return report, [event]


class Iop3ReadReportTests(unittest.TestCase):
    def check(self, report: dict, events: list[dict],
              alternate: str = "w1") -> dict:
        return check_alternate_report(
            report, events, site="IOP3.tmp_read_width", alternate=alternate,
            case_key=CASE, require_iop3_tmp_read=True)

    def test_accepts_two_complete_cta_plans_and_exact_downstream_pair(self):
        report, events = fixture()
        self.assertEqual(self.check(report, events)["record_events"], 4)
        with self.assertRaisesRegex(ValueError, "malformed IOP3.cta"):
            state_fields(report["multiplicity"][3],
                         "inner_outer_multi_wave.cta.v2")

    def test_accepts_wider_read_and_rejects_wrong_cta_geometry(self):
        report, events = fixture("w4")
        self.assertEqual(self.check(report, events, "w4")["remote_events"], 4)
        report["multiplicity"][4]["request"]["state"] = (
            cta_state(4).replace("plan=256:4:2:4:64", "plan=256:4:2:8:32"))
        with self.assertRaisesRegex(ValueError, "plan disagrees"):
            self.check(report, events, "w4")

    def test_rejects_duplicate_scalar_and_missing_plan(self):
        for corrupt in (
                cta_state(2) + "\nselected_outer_read_width=2",
                cta_state(2).split("\nplan=")[0]):
            report, events = fixture()
            report["multiplicity"][3]["request"]["state"] = corrupt
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.check(report, events)

    def test_rejects_wrong_geometry_and_unrelated_plan_change(self):
        for old, new in (("plan=256:4:2:16:16", "plan=256:4:2:8:32"),
                         ("plan=128:4:4:16:8:128:4:1:4:56:255",
                          "plan=128:4:4:16:8:128:4:1:4:56:254")):
            report, events = fixture()
            report["multiplicity"][4]["request"]["state"] = (
                cta_state(1).replace(old, new))
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.check(report, events)

    def test_rejects_unexpected_downstream_winner(self):
        report, events = fixture()
        report["multiplicity"][6]["remote"] = {"w2": 1}
        with self.assertRaisesRegex(ValueError, "changed its winner"):
            self.check(report, events)

    def test_rejects_rows_request_without_selected_read_width(self):
        for corrupt in ("width", "capacity", "menu"):
            report, events = fixture("w4")
            remote = report["multiplicity"][2]
            if corrupt == "width":
                remote["request"]["state"] = remote["request"]["state"].replace(
                    "selected_outer_read_width=4", "selected_outer_read_width=2")
            elif corrupt == "capacity":
                remote["request"]["state"] = remote["request"]["state"].replace(
                    "plan=8:640:640:256", "plan=8:128:640:256")
            else:
                remote["request"]["questions"]["IOP3.grid_rows"][
                    "criteria"].pop("r16")
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.check(report, events, "w4")


if __name__ == "__main__":
    unittest.main()
