"""Read-only checks for the generated MP16 handoff; run replay before these tests."""
from pathlib import Path
import csv
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class MP16ArtifactTests(unittest.TestCase):
    def test_required_module_outputs_exist(self):
        names = [
            "config/system_integration_config.json",
            "firmware/mp16_project_config.h",
            "results/metrics/mp16_acceptance_metrics.json",
            "results/predictions/mp16_end_to_end_predictions.csv",
            "deployment/mp16_final_project_manifest.json",
            "results/figures/mp16_latency_breakdown.png",
            "results/figures/mp16_acceptance_summary.png",
            "results/figures/mp16_command_timeline.png",
            "project_status.json",
        ]
        for name in names:
            with self.subTest(path=name):
                self.assertGreater((ROOT / name).stat().st_size, 0)

    def test_model_matches_frozen_handoff(self):
        model = read("deployment/mp15_manifest.json")["selected_model"]
        for name in ["deployment/mp16_final_project_manifest.json",
                     "results/metrics/mp16_acceptance_metrics.json",
                     "config/system_integration_config.json",
                     "firmware/esp32/include/package_manifest.json"]:
            with self.subTest(path=name):
                self.assertEqual(read(name)["selected_model"], model)
        self.assertEqual(read("project_status.json")["mp16"]["selected_model"], model)

    def test_frozen_and_final_package_hashes(self):
        for name in ["deployment/mp15_manifest.json",
                     "deployment/mp16_final_project_manifest.json"]:
            for path, expected in read(name)["sha256"].items():
                with self.subTest(manifest=name, path=path):
                    actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                    self.assertEqual(actual, expected)

    def test_prediction_audit_matches_frozen_test_ids(self):
        with (ROOT / "results/predictions/mp16_end_to_end_predictions.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        expected = read("splits/mp09_split.json")["test_ids"]
        self.assertCountEqual([r["sample_id"] for r in rows], expected)
        self.assertEqual(len(rows), read("results/metrics/mp16_acceptance_metrics.json")["n_replay_samples"])

    def test_replay_is_not_claimed_as_physical_acceptance(self):
        for name in ["deployment/mp16_final_project_manifest.json",
                     "results/metrics/mp16_acceptance_metrics.json"]:
            report = read(name)
            with self.subTest(path=name):
                self.assertEqual(report["completion_scope"], "reference_replay")
                self.assertFalse(report["physical_deployment_complete"])
                self.assertFalse(report["device_benchmark_performed"])
        metrics = read("results/metrics/mp16_acceptance_metrics.json")
        self.assertEqual(metrics["required_acceptance_pass"], all(metrics["acceptance_checks"].values()))


if __name__ == "__main__":
    unittest.main()
