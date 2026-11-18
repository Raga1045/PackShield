"""Trustworthy AI Benchmarking & Robustness Evaluation Module for DefektGuard.

Evaluates:
1. Expected Calibration Error (ECE) - Confidence Reliability
2. Robustness Under Synthetic Corruptions (Noise, Blur, Illumination Shift)
3. Epistemic Uncertainty vs. Corruption Level Correlation
4. Human-in-the-Loop Safe Triage Interception Rate
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model.model import get_model, preprocess_image, predict
from ml.dataset import scan_and_deduplicate, split_dataset
from ml.uncertainty import quantify_uncertainty
from ml.triage import evaluate_triage

logger = logging.getLogger("defektguard.benchmark.trustworthy")


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE).

    Measures the gap between model confidence and actual accuracy.
    ECE = sum(|acc(B_m) - conf(B_m)| * |B_m| / N)
    """
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def apply_synthetic_corruption(image: Image.Image, corruption_type: str, severity: int) -> Image.Image:
    """Applies realistic warehouse sensor corruptions."""
    img = image.convert("RGB")

    if corruption_type == "gaussian_noise":
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, severity * 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    elif corruption_type == "blur":
        radius = severity * 1.5
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    elif corruption_type == "low_light":
        factor = max(0.1, 1.0 - (severity * 0.2))
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)

    elif corruption_type == "low_contrast":
        factor = max(0.1, 1.0 - (severity * 0.2))
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)

    return img


def run_trustworthy_evaluation(sample_limit: int = 30) -> Dict[str, Any]:
    """Runs end-to-end Trustworthy AI validation on the test split."""
    print("=" * 70)
    print("🛡️  DefektGuard Trustworthy AI Verification & Robustness Benchmark")
    print("=" * 70)

    # 1. Load dataset split
    manifest = scan_and_deduplicate(config.DATA_DIR)
    train_set, val_set, test_set = split_dataset(manifest)
    test_files = test_set[:sample_limit]

    print(f"Loaded {len(test_files)} test samples for validation.")
    model = get_model()

    clean_confidences = []
    clean_accuracies = []
    clean_uncertainties = []

    corrupted_confidences = []
    corrupted_uncertainties = []
    intercepted_by_triage = 0
    total_corrupted_samples = 0

    print("Evaluating clean test performance & calibration...")
    for item in test_files:
        try:
            img_path, true_idx = item
            img = Image.open(img_path).convert("RGB")
            true_label = config.CLASS_NAMES[true_idx]

            res = predict(img, explain=False, quantify_risk=True)
            pred_label = res["label"].lower()
            conf = res["confidence"]
            unc = res["uncertainty_score"]

            is_correct = 1.0 if pred_label == true_label else 0.0
            clean_confidences.append(conf)
            clean_accuracies.append(is_correct)
            clean_uncertainties.append(unc)

            # Test synthetic corruption: Low-light + Gaussian noise (Warehouse camera failure)
            corrupted_img = apply_synthetic_corruption(img, "gaussian_noise", severity=2)
            corrupted_img = apply_synthetic_corruption(corrupted_img, "low_light", severity=2)

            res_corr = predict(corrupted_img, explain=False, quantify_risk=True)
            corrupted_confidences.append(res_corr["confidence"])
            corrupted_uncertainties.append(res_corr["uncertainty_score"])
            total_corrupted_samples += 1

            if res_corr["requires_human_review"]:
                intercepted_by_triage += 1

        except Exception as e:
            logger.warning(f"Error evaluating sample {item['path']}: {e}")

    clean_conf_arr = np.array(clean_confidences)
    clean_acc_arr = np.array(clean_accuracies)
    clean_unc_arr = np.array(clean_uncertainties)
    corr_unc_arr = np.array(corrupted_uncertainties)

    clean_ece = compute_expected_calibration_error(clean_conf_arr, clean_acc_arr)
    clean_accuracy = float(np.mean(clean_acc_arr)) if len(clean_acc_arr) > 0 else 0.0
    avg_clean_uncertainty = float(np.mean(clean_unc_arr)) if len(clean_unc_arr) > 0 else 0.0
    avg_corrupted_uncertainty = float(np.mean(corr_unc_arr)) if len(corr_unc_arr) > 0 else 0.0

    triage_interception_rate = (
        round(intercepted_by_triage / total_corrupted_samples * 100, 2)
        if total_corrupted_samples > 0
        else 0.0
    )

    # Uncertainty increase verification (Trustworthy AI criterion)
    uncertainty_increase_delta = round(avg_corrupted_uncertainty - avg_clean_uncertainty, 4)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "test_samples_evaluated": len(clean_confidences),
        "clean_test_accuracy": round(clean_accuracy * 100, 2),
        "calibration": {
            "expected_calibration_error_ece": clean_ece,
            "status": "CALIBRATED" if clean_ece < 0.15 else "UNRELIABLE_CONFIDENCE",
        },
        "robustness": {
            "avg_clean_uncertainty": round(avg_clean_uncertainty * 100, 2),
            "avg_corrupted_uncertainty": round(avg_corrupted_uncertainty * 100, 2),
            "uncertainty_increase_delta": uncertainty_increase_delta,
            "epistemic_safety_validated": bool(avg_corrupted_uncertainty > avg_clean_uncertainty),
        },
        "human_in_the_loop_safety": {
            "total_corrupted_tested": total_corrupted_samples,
            "intercepted_by_triage": intercepted_by_triage,
            "interception_rate_percent": triage_interception_rate,
            "status": "PASS" if triage_interception_rate >= 70.0 else "REVIEW_THRESHOLDS",
        },
    }

    report_path = config.REPORTS_DIR / "trustworthy_ai_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Verification Results:")
    print(f"   Clean Accuracy:            {report['clean_test_accuracy']}%")
    print(f"   Expected Calibration Error: {report['calibration']['expected_calibration_error_ece']}")
    print(f"   Clean Uncertainty:         {report['robustness']['avg_clean_uncertainty']}%")
    print(f"   Corrupted Uncertainty:     {report['robustness']['avg_corrupted_uncertainty']}%")
    print(f"   Triage Interception Rate:  {report['human_in_the_loop_safety']['interception_rate_percent']}%")
    print(f"📄 Report saved to: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trustworthy AI Benchmark for DefektGuard")
    parser.add_argument("--samples", type=int, default=20, help="Number of test samples to evaluate")
    args = parser.parse_args()
    run_trustworthy_evaluation(sample_limit=args.samples)
