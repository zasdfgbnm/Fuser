"""Check the exact IOP3 CTA search extension after a forced batch."""

from __future__ import annotations

import unittest

from jev_alternate_numeric import check_alternate_report
from jev_iop3_rows_report_fixture import (cta_state as complete_cta_state,
                                          rows_request)


CASE = "jev_alternate_numeric_inner_outer_min_batch"


def cta_state(extra: bool) -> str:
    return complete_cta_state(1 if extra else 2, 2)


def row(site: str, digest: str, state: str,
        record: dict[str, int], remote: dict[str, int]) -> dict:
    return {"site": site, "request_sha256": digest,
            "request": {"state": state}, "record": record, "remote": remote}


def fixture() -> tuple[dict, list[dict]]:
    batch_state = "\n".join((
        "inner_outer_multi_wave.minimum_batch.v1", "inner_dim=2048",
        "outer_dim=1024", "vector_width=4", "after_vector=512",
        "min_threads=128", "thread_cap=1024", "tmp_dtype_bits=32",
        "baseline_read_width=2", "plan=1:512:3", "plan=2:256:2",
        "plan=3:256:2", "plan=4:128:1"))
    rows = [
        row("IOP3.min_batch", "batch", batch_state, {"b2": 1}, {"b1": 1}),
        row("IOP3.tmp_read_width", "read", "read-state", {"w2": 1},
            {"w2": 1}),
        row("IOP3.grid_rows", "rows-record", "", {"r8": 1}, {}),
        row("IOP3.grid_rows", "rows-remote", "", {}, {"r8": 1}),
        row("IOP3.cta", "cta-record", cta_state(False), {"t256": 1}, {}),
        row("IOP3.cta", "cta-remote", cta_state(True), {}, {"t512": 1}),
        row("IOP3.tmp_write_width", "write", "write-state", {"w4": 1},
            {"w4": 1}),
    ]
    rows[2]["request"] = rows_request(2, 2)
    rows[3]["request"] = rows_request(1, 2)
    issues = [
        {"kind": "winner_mismatch", "site": "IOP3.min_batch",
         "request_sha256": "batch"},
        {"kind": "record_only_state", "site": "IOP3.cta",
         "request_sha256": "cta-record"},
        {"kind": "remote_only_state", "site": "IOP3.cta",
         "request_sha256": "cta-remote"},
        {"kind": "record_only_state", "site": "IOP3.grid_rows",
         "request_sha256": "rows-record"},
        {"kind": "remote_only_state", "site": "IOP3.grid_rows",
         "request_sha256": "rows-remote"},
    ]
    report = {"multiplicity": rows, "issues": issues,
              "summary": {"record_events": 5, "remote_events": 5,
                          "server_successes": 5},
              "case_coverage": {CASE: {}}}
    event = {"site": "IOP3.min_batch", "native": "b2",
             "candidates": ["b1", "b2", "b3", "b4"], "forced": True,
             "state": batch_state, "request_sha256": "batch"}
    return report, [event]


class MinBatchReportTests(unittest.TestCase):
    def check(self, report: dict, events: list[dict]) -> dict:
        return check_alternate_report(
            report, events, site="IOP3.min_batch", alternate="b1",
            case_key=CASE, require_iop3_min_batch=True)

    def test_accepts_exact_extended_cta_search(self):
        report, events = fixture()
        self.assertEqual(self.check(report, events)["remote_events"], 5)

    def test_rejects_missing_extra_cta_or_unrelated_width_change(self):
        report, events = fixture()
        report["multiplicity"][5]["request"]["state"] = cta_state(False)
        with self.assertRaises(ValueError):
            self.check(report, events)
        report, events = fixture()
        report["multiplicity"][6]["remote"] = {"w2": 1}
        with self.assertRaisesRegex(ValueError, "vector width"):
            self.check(report, events)

    def test_rejects_unreported_downstream_state(self):
        report, events = fixture()
        report["issues"].append({"kind": "remote_only_state",
                                  "site": "IOP3.tmp_write_width",
                                  "request_sha256": "rogue"})
        with self.assertRaisesRegex(ValueError, "unexpected minimum-batch"):
            self.check(report, events)

    def test_rejects_wrong_rows_answer_menu_or_grid(self):
        for corrupt in ("winner", "menu", "grid"):
            report, events = fixture()
            remote = report["multiplicity"][3]
            if corrupt == "winner":
                remote["remote"] = {"r4": 1}
            elif corrupt == "menu":
                remote["request"]["questions"]["IOP3.grid_rows"][
                    "criteria"]["r8"] = "winner"
            else:
                remote["request"]["state"] = remote["request"]["state"].replace(
                    "plan=8:640:640:512:4:1:8:64:128",
                    "plan=8:640:640:512:4:1:8:64:256")
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.check(report, events)


if __name__ == "__main__":
    unittest.main()
