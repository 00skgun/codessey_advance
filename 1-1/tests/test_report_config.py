import json
import unittest
from unittest.mock import patch

from scripts.report_evidence import load_config


class ReportConfigTests(unittest.TestCase):
    def test_overrides_merge_with_defaults(self):
        with patch("scripts.report_evidence.Path") as path:
            path.return_value.read_text.return_value = json.dumps({"iqr_threshold": 3.0})
            actual = load_config("override.json")
        self.assertEqual(actual["iqr_threshold"], 3.)
        self.assertEqual(actual["missing_strategy"], "median")

    def test_invalid_overrides_rejected(self):
        with patch("scripts.report_evidence.Path") as path:
            for override in [{"iqr_threshold": -1}, {"scatter_sample_size": 0},
                             {"misspelled_parameter": 1}, {"benchmark_repeats": 1.5}]:
                with self.subTest(override=override), self.assertRaises(ValueError):
                    path.return_value.read_text.return_value = json.dumps(override)
                    load_config("override.json")


if __name__ == "__main__":
    unittest.main()
