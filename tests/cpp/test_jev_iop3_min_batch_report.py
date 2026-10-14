"""Check the exact IOP3 CTA search extension after a forced batch."""

from __future__ import annotations

import unittest

from jev_alternate_numeric import check_alternate_report


CASE = "jev_alternate_numeric_inner_outer_min_batch"


def cta_state(extra: bool) -> str:
    plans = [
        "plan=256:4:2:8:32:128:4:2:8:48:255:8:32:128",
        "plan=128:4:4:8:16:128:4:2:4:56:255:8:16:128",
    ]
    if extra:
        plans.insert(0, "plan=512:4:1:8:64:128:4:2:16:44:128:8:64:128")
    return "\n".join((
        "inner_outer_multi_wave.cta.v2", "inner_dim=2048", "outer_dim=1024",
        "vector_width=4", "regs_buffer_bits=65536", "tmp_dtype_bits=32",
        "warp_size=32", "sm_count=128", "max_threads_per_block=1024",
        "max_grid_y=65535", "smem_total_bits=140288",
        "smem_limit_bits=811008", "selected_outer_read_width=2", *plans))


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
        row("IOP3.cta", "cta-record", cta_state(False), {"t256": 1}, {}),
        row("IOP3.cta", "cta-remote", cta_state(True), {}, {"t512": 1}),
        row("IOP3.tmp_write_width", "write", "write-state", {"w4": 1},
            {"w4": 1}),
    ]
    issues = [
        {"kind": "winner_mismatch", "site": "IOP3.min_batch",
         "request_sha256": "batch"},
        {"kind": "record_only_state", "site": "IOP3.cta",
         "request_sha256": "cta-record"},
        {"kind": "remote_only_state", "site": "IOP3.cta",
         "request_sha256": "cta-remote"},
    ]
    report = {"multiplicity": rows, "issues": issues,
              "summary": {"record_events": 4, "remote_events": 4,
                          "server_successes": 4},
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
        self.assertEqual(self.check(report, events)["remote_events"], 4)

    def test_rejects_missing_extra_cta_or_unrelated_width_change(self):
        report, events = fixture()
        report["multiplicity"][3]["request"]["state"] = cta_state(False)
        with self.assertRaises(ValueError):
            self.check(report, events)
        report, events = fixture()
        report["multiplicity"][4]["remote"] = {"w2": 1}
        with self.assertRaisesRegex(ValueError, "vector width"):
            self.check(report, events)

    def test_rejects_unreported_downstream_state(self):
        report, events = fixture()
        report["issues"].append({"kind": "remote_only_state",
                                  "site": "IOP3.tmp_write_width",
                                  "request_sha256": "rogue"})
        with self.assertRaisesRegex(ValueError, "unexpected minimum-batch"):
            self.check(report, events)


if __name__ == "__main__":
    unittest.main()
