"""Unit tests for Trustworthy AI components: Grad-CAM, Uncertainty, and Safe Triage."""

import unittest
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.resnet34 import build_resnet34
from ml.gradcam import compute_gradcam_heatmap, generate_gradcam_overlay, find_target_conv_layer
from ml.uncertainty import compute_predictive_entropy, compute_mc_dropout, quantify_uncertainty
from ml.triage import evaluate_triage


class TestTrustworthyAIComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a tiny ResNet-34 model for fast testing
        cls.model = build_resnet34(input_shape=(64, 64, 3), num_classes=2, dropout_rate=0.2)
        cls.dummy_img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        cls.dummy_tensor = tf.convert_to_tensor(
            np.random.uniform(0.0, 1.0, size=(1, 64, 64, 3)), dtype=tf.float32
        )

    def test_find_target_conv_layer(self):
        layer_name = find_target_conv_layer(self.model)
        self.assertIsNotNone(layer_name)
        layer = self.model.get_layer(layer_name)
        self.assertEqual(len(layer.output.shape), 4)

    def test_gradcam_computation_and_overlay(self):
        heatmap = compute_gradcam_heatmap(self.model, self.dummy_tensor, class_index=0)
        self.assertEqual(len(heatmap.shape), 2)
        self.assertGreaterEqual(float(np.min(heatmap)), 0.0)
        self.assertLessEqual(float(np.max(heatmap)), 1.0)

        overlay_img, data_uri = generate_gradcam_overlay(self.dummy_img, heatmap)
        self.assertEqual(overlay_img.size, (64, 64))
        self.assertTrue(data_uri.startswith("data:image/jpeg;base64,"))

    def test_predictive_entropy_boundary_values(self):
        # Deterministic prediction -> 0.0 entropy
        zero_entropy = compute_predictive_entropy(np.array([1.0, 0.0]))
        self.assertAlmostEqual(zero_entropy, 0.0, places=3)

        # Uniform distribution [0.5, 0.5] -> 1.0 maximum normalized entropy
        max_entropy = compute_predictive_entropy(np.array([0.5, 0.5]))
        self.assertAlmostEqual(max_entropy, 1.0, places=3)

        # Skewed prediction
        skewed = compute_predictive_entropy(np.array([0.85, 0.15]))
        self.assertGreater(skewed, 0.0)
        self.assertLess(skewed, 1.0)

    def test_mc_dropout_execution(self):
        mean_probs, variance, uncertainty = compute_mc_dropout(
            self.model, self.dummy_tensor, num_samples=5
        )
        self.assertEqual(len(mean_probs), 2)
        self.assertGreaterEqual(variance, 0.0)
        self.assertGreaterEqual(uncertainty, 0.0)

    def test_quantify_uncertainty_structure(self):
        dummy_probs = np.array([0.9, 0.1])
        res = quantify_uncertainty(self.model, self.dummy_tensor, dummy_probs, mc_samples=5)
        self.assertIn("uncertainty_score", res)
        self.assertIn("entropy", res)
        self.assertIn("mc_variance", res)
        self.assertIn("is_ood", res)
        self.assertIsInstance(res["uncertainty_score"], float)

    def test_triage_safe_automated_intact(self):
        triage = evaluate_triage(
            label="intact",
            confidence=0.95,
            uncertainty_score=0.05,
        )
        self.assertEqual(triage["triage_status"], "AUTO_APPROVED")
        self.assertFalse(triage["requires_human_review"])
        self.assertEqual(triage["safety_level"], "OPTIMAL")

    def test_triage_safe_automated_damaged(self):
        triage = evaluate_triage(
            label="damaged",
            confidence=0.92,
            uncertainty_score=0.08,
        )
        self.assertEqual(triage["triage_status"], "AUTO_REJECTED")
        self.assertFalse(triage["requires_human_review"])

    def test_triage_fallback_on_high_uncertainty(self):
        # High uncertainty should trigger human fallback even if confidence seems moderate
        triage = evaluate_triage(
            label="damaged",
            confidence=0.86,
            uncertainty_score=0.25,  # Exceeds 0.20 threshold
        )
        self.assertEqual(triage["triage_status"], "FLAGGED_FOR_HUMAN_REVIEW")
        self.assertTrue(triage["requires_human_review"])
        self.assertEqual(triage["safety_level"], "CAUTION")

    def test_triage_fallback_on_low_confidence(self):
        triage = evaluate_triage(
            label="intact",
            confidence=0.62,  # Below 0.85
            uncertainty_score=0.10,
        )
        self.assertEqual(triage["triage_status"], "FLAGGED_FOR_HUMAN_REVIEW")
        self.assertTrue(triage["requires_human_review"])


if __name__ == "__main__":
    unittest.main()
