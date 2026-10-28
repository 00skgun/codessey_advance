"""Small accounting and array fixtures catch errors hidden by a large dataset."""
import unittest

import numpy as np
import pandas as pd

from src.pipeline import DataAnalyzer


def fixture():
    return pd.DataFrame({
        "order_id": ["1", "1", "2", "C3", "4", "5", "6", "7"],
        "product_id": ["a", "b", "a", "a", "c", "d", "d", "e"],
        "product_name": ["A", "B", None, "A", "C", "D", "D", "E"],
        "quantity": [1, 2, 1, -1, 1, 1, 1, 1],
        "price": [10., 20., 10., 10., 1000., 0., 30., np.nan],
        "customer_id": ["x", "x", "x", "x", "y", "z", None, "z"],
        "order_date": ["2020-01-01", "2020-01-01", "2020-01-10", "2020-01-11",
                       "2020-01-10", "2020-01-10", "2020-01-10", "2020-01-10"],
        "country": ["UK"] * 8,
    })


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = DataAnalyzer()
        self.analyzer.load_data(fixture())

    def test_frequency_counts_orders_not_lines_and_uses_raw_money(self):
        a = self.analyzer
        a.handle_missing_values()
        a.detect_outliers("price", treatment="clip")
        rfm = a.calculate_rfm()
        self.assertEqual(rfm.loc["x", "frequency"], 2)
        self.assertEqual(rfm.loc["x", "monetary"], 60.)
        self.assertEqual(rfm.loc["y", "monetary"], 1000.)
        self.assertNotIn("z", rfm.index)
        self.assertEqual(rfm.loc["x", "recency"], 0)

    def test_group_imputation_and_all_missing_group_fallback(self):
        a = self.analyzer
        a.handle_missing_values()
        self.assertEqual(a.df.loc[5, "price"], 30.)
        self.assertEqual(a.df.loc[7, "price"], 15.)
        self.assertTrue(pd.isna(a.raw.loc[7, "price"]))
        self.assertTrue(pd.isna(a.df.loc[6, "customer_id"]))

    def test_vector_features_avoid_uint8_subtraction_wraparound(self):
        images = np.array([[[[0], [255]], [[0], [255]]]], dtype=np.uint8)
        result = DataAnalyzer.image_features(images).iloc[0]
        self.assertEqual(result.image_mean, 127.5)
        self.assertEqual(result.image_std, 127.5)
        self.assertEqual(result.image_edge, 127.5)

    def test_zero_iqr_flags_deviations(self):
        a = self.analyzer
        a.df["value"] = [1.] * 7 + [9.]
        self.assertEqual(len(a.detect_outliers("value")), 1)
        self.assertEqual(a.audit["value_iqr"]["iqr"], 0.)

    def test_bad_reference_date_is_rejected(self):
        with self.assertRaises(ValueError):
            self.analyzer.calculate_rfm(reference_date="2019-01-01")

    def test_tied_scores_are_equal(self):
        values = pd.Series([1, 1, 1, 2, 3])
        scores = DataAnalyzer._score(values)
        self.assertEqual(scores.iloc[:3].nunique(), 1)

    def test_invalid_transactions_excluded(self):
        df = fixture()
        df.loc[0, "order_date"] = "bad"
        df.loc[2, "order_id"] = None
        a = DataAnalyzer()
        a.load_data(df)
        valid = a.valid_transactions()
        self.assertEqual(set(valid.index), {1, 4})

    def test_all_missing_imputation_rejected(self):
        self.analyzer.df["price"] = np.nan
        with self.assertRaises(ValueError):
            self.analyzer.handle_missing_values()

    def test_reload_does_not_mutate_caller(self):
        source = fixture()
        before = source.copy(deep=True)
        self.analyzer.load_data(source)
        self.analyzer.handle_missing_values()
        pd.testing.assert_frame_equal(source, before)

    def test_missing_alias_preserves_strategy_and_group_audit(self):
        a = self.analyzer
        a.handle_missing(strategy="mean", group_col="product_id", column="price")
        self.assertEqual(a.df.loc[5, "price"], 30.)
        audit = a.audit["price_imputation"]
        self.assertEqual(audit["group_imputed_rows"], 1)
        self.assertEqual(audit["global_fallback_rows"], 1)
        self.assertEqual(audit["all_missing_groups"], 1)
        self.assertEqual(audit["strategy"], "mean")
        self.assertEqual(a.imputation_details["price"].loc["d", "missing_ratio"], 0.5)
        self.assertAlmostEqual(a.df.loc[7, "price"], (10 + 20 + 10 + 10 + 1000 + 30) / 6)

    def test_image_values_outside_documented_range_rejected(self):
        for value in [np.nan, np.inf, -1., 256.]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                DataAnalyzer.image_features(np.full((1, 2, 2, 3), value))

    def test_noncontiguous_images_match_contiguous_reference(self):
        source = np.arange(48, dtype=np.uint8).reshape(2, 2, 4, 3)
        view = source[:, :, ::2, :]
        self.assertFalse(view.flags.c_contiguous)
        pd.testing.assert_frame_equal(DataAnalyzer.image_features(view),
                                      DataAnalyzer.image_features(np.ascontiguousarray(view)))


if __name__ == "__main__":
    unittest.main()
