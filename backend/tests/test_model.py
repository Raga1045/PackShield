"""Unit tests for preprocessing and model architecture components."""

import unittest
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model.model import preprocess_image
from ml.resnet34 import build_resnet34, build_baseline_cnn


class TestModelComponents(unittest.TestCase):
    def test_preprocess_image_shape_and_range(self):
        # Create dummy 100x150 RGB image
        img = Image.new("RGB", (100, 150), color=(255, 128, 0))
        tensor = preprocess_image(img)

        self.assertEqual(tensor.shape, (1, 256, 256, 3))
        self.assertEqual(tensor.dtype, tf.float32)

        min_val = float(tf.reduce_min(tensor))
        max_val = float(tf.reduce_max(tensor))
        self.assertGreaterEqual(min_val, 0.0)
        self.assertLessEqual(max_val, 1.0)

    def test_preprocess_grayscale_conversion(self):
        # Create Grayscale 'L' mode image
        img = Image.new("L", (80, 80), color=200)
        tensor = preprocess_image(img)
        self.assertEqual(tensor.shape, (1, 256, 256, 3))

    def test_resnet34_architecture_structure(self):
        model = build_resnet34(input_shape=(256, 256, 3), num_classes=2)
        self.assertEqual(model.input_shape, (None, 256, 256, 3))
        self.assertEqual(model.output_shape, (None, 2))

        # Check total parameter count is non-zero
        self.assertGreater(model.count_params(), 1_000_000)

    def test_baseline_cnn_architecture_structure(self):
        model = build_baseline_cnn(input_shape=(256, 256, 3), num_classes=2)
        self.assertEqual(model.input_shape, (None, 256, 256, 3))
        self.assertEqual(model.output_shape, (None, 2))
        self.assertGreater(model.count_params(), 100_000)


if __name__ == "__main__":
    unittest.main()

