"""Uncertainty quantification and Out-of-Distribution (OOD) detection module.

Provides:
1. Normalized Shannon Predictive Entropy
2. Monte Carlo (MC) Dropout epistemic uncertainty estimation
3. Combined calibrated uncertainty metric and OOD indicator
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger("defektguard.ml.uncertainty")


def compute_predictive_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """Computes normalized Shannon entropy in range [0, 1].

    For C classes:
        H(p) = - sum(p_i * log2(p_i)) / log2(C)
    """
    probs = np.clip(np.array(probabilities, dtype=np.float64), eps, 1.0)
    probs = probs / np.sum(probs)  # Ensure exact normalization
    num_classes = len(probs)
    if num_classes <= 1:
        return 0.0

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_classes)
    normalized_entropy = float(np.clip(entropy / max_entropy, 0.0, 1.0))
    return round(normalized_entropy, 4)


def compute_mc_dropout(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    num_samples: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """Runs Monte Carlo Dropout inference by executing stochastic forward passes.

    Args:
        model: tf.keras.Model containing a Dropout layer.
        input_tensor: Preprocessed tensor (1, H, W, C).
        num_samples: Number of stochastic passes (default: 10).

    Returns:
        Tuple of (mean_probabilities, mean_variance, epistemic_uncertainty).
    """
    predictions: List[np.ndarray] = []

    # Check if model has a dropout layer
    has_dropout = any(isinstance(l, tf.keras.layers.Dropout) for l in model.layers)

    if not has_dropout:
        # Fallback to standard prediction if no dropout layer exists
        raw_pred = model(input_tensor, training=False).numpy()[0]
        return raw_pred, 0.0, 0.0

    # Repeat forward passes with dropout active
    # We pass training=True to keep dropout sampling stochastic
    for _ in range(num_samples):
        out = model(input_tensor, training=True)
        predictions.append(out.numpy()[0])

    preds_arr = np.array(predictions)  # Shape: (num_samples, num_classes)
    mean_probs = np.mean(preds_arr, axis=0)
    variances = np.var(preds_arr, axis=0)
    mean_variance = float(np.mean(variances))

    # Normalized epistemic uncertainty metric based on maximum possible variance for binary (0.25)
    max_possible_var = 0.25
    epistemic_uncertainty = float(np.clip(mean_variance / max_possible_var, 0.0, 1.0))

    return mean_probs, round(mean_variance, 6), round(epistemic_uncertainty, 4)


def quantify_uncertainty(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    standard_probabilities: np.ndarray,
    mc_samples: int = 10,
    entropy_weight: float = 0.6,
    mc_weight: float = 0.4,
) -> Dict[str, Any]:
    """Computes a comprehensive Trustworthy AI uncertainty evaluation.

    Combines predictive entropy and MC Dropout variance into a calibrated
    uncertainty score between 0.0 (fully certain) and 1.0 (highly uncertain).
    """
    # 1. Predictive entropy from standard inference
    entropy_score = compute_predictive_entropy(standard_probabilities)

    # 2. MC Dropout estimation
    mc_probs, mc_var, mc_uncertainty = compute_mc_dropout(
        model, input_tensor, num_samples=mc_samples
    )

    # 3. Combined epistemic + aleatoric uncertainty metric
    if mc_var > 0:
        combined_uncertainty = (
            entropy_weight * entropy_score + mc_weight * mc_uncertainty
        )
    else:
        combined_uncertainty = entropy_score

    combined_uncertainty = round(float(np.clip(combined_uncertainty, 0.0, 1.0)), 4)

    # 4. Out-of-Distribution / High Ambiguity Flag
    # High uncertainty (> 0.20) or low margin indicates potential OOD or degraded image
    is_ood = bool(combined_uncertainty > 0.20 or np.max(standard_probabilities) < 0.70)

    return {
        "uncertainty_score": combined_uncertainty,
        "entropy": entropy_score,
        "mc_variance": mc_var,
        "mc_uncertainty": mc_uncertainty,
        "mc_probabilities": [round(float(p), 4) for p in mc_probs],
        "is_ood": is_ood,
    }
