"""
tests/test_evaluation.py
========================
Unit tests for src/evaluation.py.

All tests use small, synthetic score/label lists.  No dataset files are read.
"""

from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (
    confusion_counts,
    precision,
    recall,
    f1_score,
    f_beta_score,
    evaluate_at_threshold,
    evaluate_thresholds,
    threshold_table_str,
    evaluate_predictions,
)


# ---------------------------------------------------------------------------
# 1. confusion_counts
# ---------------------------------------------------------------------------

class TestConfusionCounts(unittest.TestCase):

    def _counts(self, y_true, y_scores, threshold):
        return confusion_counts(y_true, y_scores, threshold)

    def test_all_correct(self):
        c = self._counts([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
        self.assertEqual(c, {"tp": 2, "fp": 0, "fn": 0, "tn": 2})

    def test_all_wrong(self):
        c = self._counts([1, 1, 0, 0], [0.1, 0.2, 0.9, 0.8], 0.5)
        self.assertEqual(c, {"tp": 0, "fp": 2, "fn": 2, "tn": 0})

    def test_threshold_at_boundary(self):
        # Score exactly equals threshold -> predicted positive
        c = self._counts([1, 0], [0.5, 0.5], 0.5)
        self.assertEqual(c["tp"], 1)
        self.assertEqual(c["fp"], 1)

    def test_all_negative_predictions(self):
        # Very high threshold -> no positives predicted
        c = self._counts([1, 0, 1], [0.3, 0.4, 0.5], 0.99)
        self.assertEqual(c["tp"], 0)
        self.assertEqual(c["fp"], 0)
        self.assertEqual(c["fn"], 2)
        self.assertEqual(c["tn"], 1)

    def test_all_positive_predictions(self):
        # Very low threshold -> everything predicted positive
        c = self._counts([1, 0, 0], [0.3, 0.4, 0.5], 0.0)
        self.assertEqual(c["tp"], 1)
        self.assertEqual(c["fp"], 2)
        self.assertEqual(c["fn"], 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self._counts([1, 0], [0.5], 0.5)

    def test_empty_inputs(self):
        c = self._counts([], [], 0.5)
        self.assertEqual(c, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

    def test_single_tp(self):
        c = self._counts([1], [1.0], 0.5)
        self.assertEqual(c, {"tp": 1, "fp": 0, "fn": 0, "tn": 0})

    def test_single_tn(self):
        c = self._counts([0], [0.0], 0.5)
        self.assertEqual(c, {"tp": 0, "fp": 0, "fn": 0, "tn": 1})


# ---------------------------------------------------------------------------
# 2. precision
# ---------------------------------------------------------------------------

class TestPrecision(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(precision(10, 0), 1.0)

    def test_zero_precision(self):
        self.assertAlmostEqual(precision(0, 5), 0.0)

    def test_no_predictions(self):
        self.assertAlmostEqual(precision(0, 0), 0.0)  # safe default

    def test_mixed(self):
        self.assertAlmostEqual(precision(3, 1), 0.75)

    def test_all_fp(self):
        self.assertAlmostEqual(precision(0, 10), 0.0)


# ---------------------------------------------------------------------------
# 3. recall
# ---------------------------------------------------------------------------

class TestRecall(unittest.TestCase):

    def test_perfect(self):
        self.assertAlmostEqual(recall(10, 0), 1.0)

    def test_zero_recall(self):
        self.assertAlmostEqual(recall(0, 5), 0.0)

    def test_no_actual_positives(self):
        self.assertAlmostEqual(recall(0, 0), 0.0)  # safe default

    def test_mixed(self):
        self.assertAlmostEqual(recall(3, 1), 0.75)

    def test_all_fn(self):
        self.assertAlmostEqual(recall(0, 10), 0.0)


# ---------------------------------------------------------------------------
# 4. f1_score
# ---------------------------------------------------------------------------

class TestF1Score(unittest.TestCase):

    def test_perfect_f1(self):
        self.assertAlmostEqual(f1_score(1.0, 1.0), 1.0)

    def test_zero_f1_both_zero(self):
        self.assertAlmostEqual(f1_score(0.0, 0.0), 0.0)

    def test_zero_f1_precision_zero(self):
        self.assertAlmostEqual(f1_score(0.0, 1.0), 0.0)

    def test_zero_f1_recall_zero(self):
        self.assertAlmostEqual(f1_score(1.0, 0.0), 0.0)

    def test_balanced(self):
        # P=0.8, R=0.8 -> F1=0.8
        self.assertAlmostEqual(f1_score(0.8, 0.8), 0.8)

    def test_harmonic_mean(self):
        p, r = 0.6, 0.9
        expected = 2 * p * r / (p + r)
        self.assertAlmostEqual(f1_score(p, r), expected, places=6)


# ---------------------------------------------------------------------------
# 5. f_beta_score
# ---------------------------------------------------------------------------

class TestFBetaScore(unittest.TestCase):

    def test_beta1_equals_f1(self):
        p, r = 0.7, 0.5
        self.assertAlmostEqual(f_beta_score(p, r, beta=1.0),
                               f1_score(p, r), places=6)

    def test_beta2_favours_recall(self):
        # With high beta, high-recall / low-precision should score better.
        p, r = 0.5, 0.9
        fb2 = f_beta_score(p, r, beta=2.0)
        fb05 = f_beta_score(p, r, beta=0.5)
        self.assertGreater(fb2, fb05)

    def test_perfect_f_beta(self):
        self.assertAlmostEqual(f_beta_score(1.0, 1.0, beta=2.0), 1.0)

    def test_zero_f_beta(self):
        self.assertAlmostEqual(f_beta_score(0.0, 0.0, beta=2.0), 0.0)


# ---------------------------------------------------------------------------
# 6. evaluate_at_threshold
# ---------------------------------------------------------------------------

class TestEvaluateAtThreshold(unittest.TestCase):

    # Perfect classifier
    def setUp(self):
        self.y_true   = [1, 1, 0, 0, 1, 0]
        self.y_scores = [0.9, 0.8, 0.1, 0.2, 0.95, 0.05]

    def test_perfect_at_0_5(self):
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.5)
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["f1"], 1.0)

    def test_tp_fp_fn_values(self):
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.5)
        self.assertEqual(r["tp"], 3)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertEqual(r["tn"], 3)

    def test_support_equals_len(self):
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.5)
        self.assertEqual(r["support"], len(self.y_true))

    def test_threshold_stored(self):
        t = 0.7
        r = evaluate_at_threshold(self.y_true, self.y_scores, t)
        self.assertAlmostEqual(r["threshold"], t)

    def test_high_threshold_low_recall(self):
        # At threshold 0.92 only score 0.95 fires -> recall drops
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.92)
        self.assertEqual(r["tp"], 1)
        self.assertLess(r["recall"], 1.0)

    def test_zero_threshold_all_positive(self):
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.0)
        self.assertEqual(r["n_predicted_positive"], len(self.y_true))

    def test_n_actual_positive(self):
        r = evaluate_at_threshold(self.y_true, self.y_scores, 0.5)
        self.assertEqual(r["n_actual_positive"], sum(self.y_true))


# ---------------------------------------------------------------------------
# 7. evaluate_thresholds
# ---------------------------------------------------------------------------

class TestEvaluateThresholds(unittest.TestCase):

    def setUp(self):
        self.y_true   = [1, 1, 0, 0, 1, 0]
        self.y_scores = [0.9, 0.8, 0.1, 0.2, 0.95, 0.05]

    def test_default_thresholds_length(self):
        rows = evaluate_thresholds(self.y_true, self.y_scores)
        # Default: 5% to 95% in 5% steps = 19 thresholds
        self.assertEqual(len(rows), 19)

    def test_custom_thresholds(self):
        thresholds = [0.3, 0.5, 0.7]
        rows = evaluate_thresholds(self.y_true, self.y_scores,
                                   thresholds=thresholds)
        self.assertEqual(len(rows), 3)
        # Sorted ascending
        self.assertEqual(rows[0]["threshold"], 0.3)
        self.assertEqual(rows[-1]["threshold"], 0.7)

    def test_monotone_precision_as_threshold_rises(self):
        # As threshold increases, precision generally non-decreasing
        # (may stay flat -- not necessarily strict monotone).
        rows = evaluate_thresholds(self.y_true, self.y_scores)
        for i in range(1, len(rows)):
            # Allow equal -- only ensure it never dramatically drops
            self.assertGreaterEqual(
                rows[i]["precision"] + 0.01, rows[i - 1]["precision"]
            )

    def test_monotone_recall_as_threshold_falls(self):
        rows = evaluate_thresholds(self.y_true, self.y_scores)
        # Recall should generally decrease as threshold rises
        for i in range(1, len(rows)):
            self.assertGreaterEqual(
                rows[i - 1]["recall"] + 0.01, rows[i]["recall"]
            )

    def test_all_rows_have_required_keys(self):
        rows = evaluate_thresholds(self.y_true, self.y_scores)
        required = {"threshold", "precision", "recall", "f1",
                    "tp", "fp", "fn", "tn", "support"}
        for row in rows:
            for k in required:
                self.assertIn(k, row)

    def test_no_best_threshold_selected(self):
        # The function must NOT include a 'best_threshold' or 'selected' key.
        rows = evaluate_thresholds(self.y_true, self.y_scores)
        for row in rows:
            self.assertNotIn("best_threshold", row)
            self.assertNotIn("selected", row)


# ---------------------------------------------------------------------------
# 8. threshold_table_str
# ---------------------------------------------------------------------------

class TestThresholdTableStr(unittest.TestCase):

    def test_returns_string(self):
        rows = evaluate_thresholds([1, 0], [0.9, 0.1])
        self.assertIsInstance(threshold_table_str(rows), str)

    def test_contains_header(self):
        rows = evaluate_thresholds([1, 0], [0.9, 0.1])
        t = threshold_table_str(rows)
        self.assertIn("threshold", t)
        self.assertIn("precision", t)
        self.assertIn("recall", t)
        self.assertIn("f1", t)

    def test_row_count_matches(self):
        rows = evaluate_thresholds([1, 0], [0.9, 0.1], thresholds=[0.3, 0.7])
        t = threshold_table_str(rows)
        # header + divider + 2 data rows
        self.assertEqual(len(t.strip().split("\n")), 4)


# ---------------------------------------------------------------------------
# 9. Edge cases – all positive / all negative ground truth
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_all_labels_positive(self):
        # No true negatives or FP possible
        y_true   = [1, 1, 1]
        y_scores = [0.8, 0.6, 0.4]
        r = evaluate_at_threshold(y_true, y_scores, 0.5)
        self.assertEqual(r["tn"], 0)
        self.assertEqual(r["fp"], 0)

    def test_all_labels_negative(self):
        y_true   = [0, 0, 0]
        y_scores = [0.8, 0.6, 0.4]
        r = evaluate_at_threshold(y_true, y_scores, 0.5)
        self.assertEqual(r["tp"], 0)
        self.assertEqual(r["fn"], 0)
        self.assertAlmostEqual(r["precision"], 0.0)
        self.assertAlmostEqual(r["recall"], 0.0)

    def test_single_pair_match(self):
        r = evaluate_at_threshold([1], [0.9], 0.5)
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 1.0)
        self.assertAlmostEqual(r["f1"], 1.0)

    def test_single_pair_non_match(self):
        r = evaluate_at_threshold([0], [0.1], 0.5)
        self.assertAlmostEqual(r["precision"], 0.0)
        self.assertAlmostEqual(r["recall"], 0.0)
        self.assertAlmostEqual(r["f1"], 0.0)

    def test_no_positive_predictions_precision_zero(self):
        y_true   = [1, 0, 1]
        y_scores = [0.1, 0.2, 0.3]
        r = evaluate_at_threshold(y_true, y_scores, 0.99)
        self.assertAlmostEqual(r["precision"], 0.0)
        self.assertEqual(r["n_predicted_positive"], 0)


# ---------------------------------------------------------------------------
# 10. evaluate_predictions convenience wrapper
# ---------------------------------------------------------------------------

class TestEvaluatePredictions(unittest.TestCase):

    def test_basic(self):
        preds = [(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)]
        r = evaluate_predictions(preds, threshold=0.5)
        self.assertAlmostEqual(r["precision"], 1.0)
        self.assertAlmostEqual(r["recall"], 1.0)

    def test_threshold_respected(self):
        preds = [(0.9, 1), (0.8, 0), (0.2, 1), (0.1, 0)]
        r = evaluate_predictions(preds, threshold=0.85)
        # Only (0.9, 1) fires: 1 TP, 0 FP, 1 FN
        self.assertEqual(r["tp"], 1)
        self.assertEqual(r["fp"], 0)
        self.assertEqual(r["fn"], 1)

    def test_empty_predictions(self):
        r = evaluate_predictions([], threshold=0.5)
        self.assertEqual(r["support"], 0)
        self.assertAlmostEqual(r["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
