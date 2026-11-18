"""Reproducible evaluation script for DefektGuard.

Calculates actual, verified metrics on the test set:
- Accuracy
- Precision (Positive = 'damaged')
- Recall (Sensitivity)
- Specificity
- F1-Score
- Confusion Matrix
- Error Analysis (Misclassified cases)

Outputs JSON summary and Confusion Matrix plot.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from ml.dataset import load_image_and_label, scan_and_deduplicate, split_dataset


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes exact binary classification metrics with zero external dependency."""
    # Positive class is damaged (index 1), Negative is intact (index 0)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "total_samples": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": [
            [tn, fp],  # Row 0: Actual Intact -> [Pred Intact, Pred Damaged]
            [fn, tp],  # Row 1: Actual Damaged -> [Pred Intact, Pred Damaged]
        ],
    }


def plot_confusion_matrix(cm: List[List[int]], save_path: Path):
    """Generates and saves a clean confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm_arr = np.array(cm)
    
    im = ax.imshow(cm_arr, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax)
    
    classes = ["Intact", "Damaged"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title("DefektGuard Test Confusion Matrix", fontsize=13, fontweight="bold", pad=12)

    # Label cells
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_arr[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"📊 Saved confusion matrix visualization: {save_path}")


def evaluate(
    model_path: Path = config.DEFAULT_MODEL_PATH,
    data_dir: Path = config.DATA_DIR,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    model_path = Path(model_path)
    if not model_path.exists():
        # Check fallback
        fallback = model_path.with_suffix(".h5")
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {fallback}")

    print(f"🔍 Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Determine test items
    split_meta_path = config.REPORTS_DIR / "split_manifest.json"
    test_items = []
    if split_meta_path.exists():
        with open(split_meta_path, "r") as f:
            manifest = json.load(f)
            test_items = [(Path(item["path"]), item["label"]) for item in manifest.get("test_items", [])]

    if not test_items:
        print("ℹ️ Split manifest not found. Generating test split from dataset...")
        class_images = scan_and_deduplicate(data_dir)
        _, _, test_items = split_dataset(
            class_images,
            val_split=config.VAL_SPLIT,
            test_split=config.TEST_SPLIT,
            seed=seed,
        )

    if not test_items:
        raise ValueError(f"No test items found in {data_dir}")

    print(f"🧪 Evaluating on {len(test_items)} test images...")

    y_true, y_pred, confidences, errors = [], [], [], []

    for img_path, label in test_items:
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(config.IMG_SIZE)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        batch = np.expand_dims(img_arr, axis=0)

        preds = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        conf = float(np.max(preds))

        y_true.append(label)
        y_pred.append(pred_idx)
        confidences.append(conf)

        if pred_idx != label:
            errors.append({
                "image": str(img_path),
                "true_label": config.CLASS_NAMES[label],
                "pred_label": config.CLASS_NAMES[pred_idx],
                "confidence": round(conf, 4),
                "error_type": "False Positive (Intact marked Damaged)" if label == 0 else "False Negative (Damaged marked Intact)",
            })

    metrics = compute_metrics(np.array(y_true), np.array(y_pred))
    metrics["average_confidence"] = round(float(np.mean(confidences)), 4)
    metrics["errors"] = errors
    metrics["model_path"] = str(model_path)

    # Print summary table
    print("\n" + "=" * 50)
    print("📈 DEFEKTGUARD TEST EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Test Samples: {metrics['total_samples']}")
    print(f"Accuracy:           {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (Damaged): {metrics['precision'] * 100:.2f}%")
    print(f"Recall (Damaged):    {metrics['recall'] * 100:.2f}%")
    print(f"Specificity:        {metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score:           {metrics['f1_score'] * 100:.2f}%")
    print(f"Avg Confidence:     {metrics['average_confidence'] * 100:.2f}%")
    print("-" * 50)
    print(f"True Positives (Damaged detected): {metrics['true_positives']}")
    print(f"True Negatives (Intact detected):  {metrics['true_negatives']}")
    print(f"False Positives (Intact -> Damaged): {metrics['false_positives']}")
    print(f"False Negatives (Damaged -> Intact): {metrics['false_negatives']}")
    print("=" * 50 + "\n")

    # Save outputs
    cm_path = config.REPORTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)

    results_path = config.REPORTS_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved evaluation results to: {results_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DefektGuard Model")
    parser.add_argument("--model-path", type=str, default=str(config.DEFAULT_MODEL_PATH), help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR), help="Path to dataset")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    evaluate(model_path=Path(args.model_path), data_dir=Path(args.data_dir), seed=args.seed)

