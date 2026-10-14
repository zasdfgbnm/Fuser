# SPDX-License-Identifier: BSD-3-Clause
"""Host-only guard for complete modified scheduler outcome coverage."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[2]
SCHEDULERS = (
    "no_op", "pointwise", "reduction", "normalization_inner",
    "normalization_outer", "normalization_inner_outer", "transpose",
    "matmul", "expr_eval_sched", "resize", "greedy", "communication",
    "cutlass",
)


class JevOutcomeCoverageTest(unittest.TestCase):
    def test_every_scheduler_implements_only_the_observed_entry(self):
        scheduler_dir = ROOT / "csrc/scheduler"
        base = (scheduler_dir / "registry.h").read_text()
        registry = (scheduler_dir / "registry.cpp").read_text()
        self.assertIn("computeHeuristicsImpl(fusion, runtime_info, data_cache)", registry)
        self.assertEqual(registry.count("scheduler::native_diagnostic::recordOutcome(outcome)"), 1)
        self.assertIn("params != nullptr && scheduler::native_diagnostic::enabled()", registry)
        self.assertRegex(base, r"std::unique_ptr<HeuristicParams> computeHeuristics\(")
        self.assertRegex(base, r"virtual std::unique_ptr<HeuristicParams> computeHeuristicsImpl\(")
        derived = {
            path.stem for path in scheduler_dir.glob("*.h")
            if re.search(r"class\s+\w+\s*:\s*public SchedulerEntry", path.read_text())
        }
        self.assertEqual(derived, set(SCHEDULERS))
        for name in SCHEDULERS:
            with self.subTest(scheduler=name):
                header = (scheduler_dir / f"{name}.h").read_text()
                body = (scheduler_dir / f"{name}.cpp").read_text()
                self.assertEqual(header.count("computeHeuristicsImpl("), 1)
                self.assertEqual(len(re.findall(r"::\s*computeHeuristicsImpl\(", body)), 1)
                self.assertNotRegex(body, r"::\s*computeHeuristics\(")

    def test_direct_users_still_call_the_public_entry(self):
        for relative in (
            "csrc/fusion_segmenter.cpp", "csrc/scheduler/heuristic.cpp",
            "csrc/scheduler/registry.cpp", "python/python_direct/schedule.cpp",
            "benchmarks/cpp/softmax.cpp",
        ):
            with self.subTest(source=relative):
                source = (ROOT / relative).read_text()
                self.assertIn("->computeHeuristics(", source)
                self.assertNotIn("->computeHeuristicsImpl(", source)


if __name__ == "__main__":
    unittest.main()
