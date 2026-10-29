# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only checks for Pointwise's terminal Record diagnostic contract."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_pointwise_record_contract import (  # noqa: E402
    BOOLEAN_FINAL_FIELDS, COMPILE_FIELDS, FINAL_FIELDS, LAUNCH_FIELDS,
    NATIVE_FINAL_FIELDS, SCHEMA, group_record_invocations,
    project_record_final_for_native, read_jsonl_strict,
)


def outcome(identifier: str, backend: str = "tma",
            event_count: int = 0) -> list[dict]:
    params = {field: 0 for field in FINAL_FIELDS}
    params.update(scheduler_type="PointWise", tag="Pointwise heuristics",
                  use_tma_load=backend == "tma")
    params.update({field: False for field in BOOLEAN_FINAL_FIELDS - {"use_tma_load"}})
    begin = {"schema": SCHEMA, "kind": "begin", "mode": "record",
             "invocation_id": identifier, "event_index": 0, "case_key": "c01"}
    final = {**begin, "kind": "final", "event_index": event_count + 1,
             "backend": backend,
             "final_params": params,
             "launch": {field: -1 for field in LAUNCH_FIELDS},
             "compile": {field: None for field in COMPILE_FIELDS}}
    return [begin, final]


def choice(identifier: str, site: str, selected: str,
           event_index: int = 1) -> dict:
    return {"site": site, "mode": "record", "case_key": "c01",
            "invocation_id": identifier, "event_index": event_index,
            "selected": selected}


class PointwiseRecordContractTest(unittest.TestCase):
    def test_tma_return_ignores_discarded_ordinary_candidate(self):
        # A valid TMA plan can return without PW1. The ordinary PW2/PW3 work
        # is still logged while trying to materialize a second plan.
        decisions = [choice("1", "PW4.width", "w4", 1),
                     choice("1", "PW2", "bp1_normal", 2),
                     choice("1", "PW3.width", "w8", 3)]
        grouped = group_record_invocations(decisions, outcome("1", event_count=3))
        self.assertEqual([row["site"] for row in grouped["1"]["realized_choices"]],
                         ["PW4.width"])
        self.assertEqual([row["site"] for row in grouped["1"]["speculative_choices"]],
                         ["PW2", "PW3.width"])
        self.assertEqual(grouped["1"]["backend"], "tma")

    def test_repeated_case_key_is_segmented_by_invocation(self):
        rows = outcome("1", event_count=1) + outcome("2", "ordinary", 1)
        decisions = [choice("1", "PW1", "tma"),
                     choice("2", "PW2", "bp0")]
        grouped = group_record_invocations(decisions, rows)
        self.assertEqual({key: value["backend"] for key, value in grouped.items()},
                         {"1": "tma", "2": "ordinary"})

    def test_incomplete_or_mismatched_outcome_fails_closed(self):
        for rows, decisions in (
            (outcome("1")[:1], [choice("1", "PW4.width", "w4")]),
            (outcome("1", event_count=1), [choice("1", "PW1", "ordinary")]),
            (outcome("1"), [choice("2", "PW4.width", "w4")]),
        ):
            with self.subTest(rows=rows, decisions=decisions), self.assertRaises(ValueError):
                group_record_invocations(decisions, rows)

    def test_snapshot_requires_every_returned_field(self):
        for section, field in (("final_params", "vectorization_factor"),
                               ("launch", "block_x_raw"),
                               ("compile", "index_type")):
            rows = outcome("1")
            del rows[1][section][field]
            with self.subTest(section=section), self.assertRaises(ValueError):
                group_record_invocations([], rows)

    def test_sequence_rejects_final_before_begin_and_events_after_final(self):
        begin, final = outcome("1")
        skip_after_final = {**begin, "kind": "coverage_skip", "event_index": 2,
                            "site": "PW4.tile", "reason": "no safe tile"}
        for rows, decisions in (
            ([final, begin], []),
            ([begin, final, skip_after_final], []),
            ([begin, final], [choice("1", "PW4.width", "w4", 2)]),
            (outcome("1", event_count=2),
             [choice("1", "PW4.width", "w4", 1),
              choice("1", "PW4.tile", "i128", 1)]),
            (outcome("1", event_count=3),
             [choice("1", "PW4.width", "w4", 1),
              choice("1", "PW4.tile", "i128", 3)]),
        ):
            with self.subTest(rows=rows, decisions=decisions), self.assertRaises(ValueError):
                group_record_invocations(decisions, rows)

    def test_sequence_accepts_interleaved_choice_and_coverage(self):
        begin, final = outcome("1", event_count=2)
        coverage = {**begin, "kind": "coverage_skip", "event_index": 2,
                    "site": "PW4.tile", "reason": "no safe tile"}
        grouped = group_record_invocations(
            [choice("1", "PW4.width", "w4", 1)], [begin, coverage, final])
        self.assertEqual(len(grouped["1"]["coverage_skips"]), 1)

    def test_native_projection_drops_only_modified_fields(self):
        final = group_record_invocations([], outcome("1"))["1"]["final"]
        native = project_record_final_for_native(final)
        self.assertEqual(set(native["final_params"]), NATIVE_FINAL_FIELDS)
        self.assertNotIn("scheduler_type", native["final_params"])
        self.assertNotIn("symbolic_1d_cta", native["final_params"])
        modified_only = deepcopy(final)
        modified_only["final_params"]["symbolic_1d_cta"] = True
        self.assertEqual(project_record_final_for_native(modified_only), native)
        for section, field, value in (
            ("final_params", "vectorization_factor", 8),
            ("launch", "block_x_raw", 128),
            ("compile", "index_type", 1),
        ):
            changed = deepcopy(final)
            changed[section][field] = value
            with self.subTest(section=section, field=field):
                self.assertNotEqual(project_record_final_for_native(changed), native)

    def test_strict_jsonl_rejects_truncation_and_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcome.jsonl"
            path.write_text(json.dumps(outcome("1")[0]), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl_strict(path)
            path.write_text('{"kind":"begin","kind":"final"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl_strict(path)


if __name__ == "__main__":
    unittest.main()
