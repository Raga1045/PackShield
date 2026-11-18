"""Unit tests for ML evaluation metric calculations."""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import compute_metrics


class TestMetricsCalculation(unittest.TestCase):
    def test_perfect_predictions(self):
        # 3 intact (0), 3 damaged (1) - all predicted correctly
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["total_samples"], 6)
        self.assertEqual(metrics["true_positives"], 3)
        self.assertEqual(metrics["true_negatives"], 3)
        self.assertEqual(metrics["false_positives"], 0)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[3, 0], [0, 3]])

    def test_known_confusion_matrix(self):
        # y_true: 5 intact (0), 5 damaged (1)
        # y_pred: 4 TN, 1 FP (0->1), 2 FN (1->0), 3 TP (1->1)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["true_positives"], 3)
        self.assertEqual(metrics["true_negatives"], 4)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 2)

        # Accuracy = (3 + 4) / 10 = 0.70
        self.assertAlmostEqual(metrics["accuracy"], 0.70, places=4)
        # Precision = 3 / (3 + 1) = 0.75
        self.assertAlmostEqual(metrics["precision"], 0.75, places=4)
        # Recall = 3 / (3 + 2) = 0.60
        self.assertAlmostEqual(metrics["recall"], 0.60, places=4)
        # Specificity = 4 / (4 + 1) = 0.80
        self.assertAlmostEqual(metrics["specificity"], 0.80, places=4)
        # F1 = 2 * (0.75 * 0.60) / (0.75 + 0.60) = 0.90 / 1.35 = 0.6667
        self.assertAlmostEqual(metrics["f1_score"], 0.6667, places=4)

    def test_zero_positives_division_guard(self):
        # Model predicts all 0 (no positive predictions)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        metrics = compute_metrics(y_true, y_pred)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

