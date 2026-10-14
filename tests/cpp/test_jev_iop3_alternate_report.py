"""Regression checks for IOP3 read-width-dependent decision logs."""
from __future__ import annotations

from copy import deepcopy
import unittest

from jev_alternate_numeric import check_alternate_report, state_fields

CASE = "jev_alternate_numeric_inner_outer_tmp_read_width"


def cta_state(read: int) -> str:
    x = 8 if read == 2 else 16
    return "\n".join((
        "inner_outer_multi_wave.cta.v2",
        "inner_dim=2048", "outer_dim=1024", "vector_width=4",
        "regs_buffer_bits=65536", "tmp_dtype_bits=32", "warp_size=32",
        "sm_count=128", "max_threads_per_block=1024", "max_grid_y=65535",
        "smem_total_bits=140288", "smem_limit_bits=811008",
        f"selected_outer_read_width={read}",
        f"plan=256:4:2:{x}:{256 // x}:128:4:{read}:8:48:255:"
        f"{x}:{256 // x}:128",
        f"plan=128:4:4:{x}:{128 // x}:128:4:{read}:4:56:255:"
        f"{x}:{128 // x}:128",
    ))


def write_state(read: int) -> str:
    return ("inner_outer_multi_wave.tmp_write_width.v2\n"
            "inner_dim=2048\ninner_vector_width=4\n"
            f"outer_read_width={read}\ntmp_dtype_bits=32")


def row(site: str, hash_: str, state: str,
        record: dict[str, int], remote: dict[str, int]) -> dict:
    return {"site": site, "request_sha256": hash_,
            "request": {"state": state},
            "record": record, "remote": remote}


def fixture() -> tuple[dict, list[dict]]:
    entries = [
        row("IOP3.tmp_read_width", "read", "read-state",
            {"w2": 1}, {"w1": 1}),
        row("IOP3.cta", "cta-record", cta_state(2), {"t256": 1}, {}),
        row("IOP3.cta", "cta-remote", cta_state(1), {}, {"t256": 1}),
        row("IOP3.tmp_write_width", "write-record", write_state(2),
            {"w4": 1}, {}),
        row("IOP3.tmp_write_width", "write-remote", write_state(1),
            {}, {"w4": 1}),
    ]
    issues = [{"kind": "winner_mismatch", "site": "IOP3.tmp_read_width",
               "request_sha256": "read"}]
    for site, prefix in (("IOP3.cta", "cta"),
                         ("IOP3.tmp_write_width", "write")):
        for side in ("record", "remote"):
            issues.append({"kind": f"{side}_only_state", "site": site,
                           "request_sha256": f"{prefix}-{side}"})
    report = {"multiplicity": entries, "issues": issues,
              "summary": {"remote_events": 3, "server_successes": 3,
                          "record_events": 3},
              "case_coverage": {CASE: {}}}
    event = {"site": "IOP3.tmp_read_width", "native": "w2",
             "candidates": ["w1", "w2", "w4"], "forced": True,
             "request_sha256": "read"}
    return report, [event]


class Iop3ReadReportTests(unittest.TestCase):
    def check(self, report: dict, events: list[dict]) -> dict:
        return check_alternate_report(
            report, events, site="IOP3.tmp_read_width", alternate="w1",
            case_key=CASE, require_iop3_tmp_read=True)

    def test_accepts_two_complete_cta_plans_and_exact_downstream_pair(self):
        report, events = fixture()
        self.assertEqual(self.check(report, events)["record_events"], 3)
        with self.assertRaisesRegex(ValueError, "malformed IOP3.cta"):
            state_fields(report["multiplicity"][1],
                         "inner_outer_multi_wave.cta.v2")

    def test_rejects_duplicate_scalar_and_missing_plan(self):
        for corrupt in (
                cta_state(2) + "\nselected_outer_read_width=2",
                cta_state(2).split("\nplan=")[0]):
            report, events = fixture()
            report["multiplicity"][1]["request"]["state"] = corrupt
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.check(report, events)

    def test_rejects_wrong_geometry_and_unrelated_plan_change(self):
        for old, new in (("plan=256:4:2:16:16", "plan=256:4:2:8:32"),
                         ("plan=128:4:4:16:8:128:4:1:4:56:255",
                          "plan=128:4:4:16:8:128:4:1:4:56:254")):
            report, events = fixture()
            report["multiplicity"][2]["request"]["state"] = (
                cta_state(1).replace(old, new))
            with self.subTest(old=old), self.assertRaises(ValueError):
                self.check(report, events)

    def test_rejects_unexpected_downstream_winner(self):
        report, events = fixture()
        report["multiplicity"][4]["remote"] = {"w2": 1}
        with self.assertRaisesRegex(ValueError, "changed its winner"):
            self.check(report, events)


if __name__ == "__main__":
    unittest.main()
