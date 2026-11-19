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


def compute_energy_score(probabilities: np.ndarray, temperature: float = 2.0, eps: float = 1e-12) -> float:
    """Computes temperature-scaled Free Energy for OOD detection (Liu et al., NeurIPS 2020).

    For class probabilities p:
        E(p; T) = -T * log(sum((p_i + eps) ** (1 / T)))
    """
    probs = np.clip(np.array(probabilities, dtype=np.float64), eps, 1.0)
    scaled_sum = np.sum(np.power(probs, 1.0 / temperature))
    energy = -temperature * np.log(scaled_sum + eps)
    return round(float(energy), 4)


def verify_packaging_domain(
    input_data: Union[tf.Tensor, np.ndarray],
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Evaluates whether an image conforms to the packaging inspection domain.

    Detects out-of-distribution non-packaging inputs including:
    1. Human portraits, people, faces, and dark-hair contours
    2. Direct hand or skin close-ups lacking container boundaries
    3. Outdoor scenes, natural foliage, greenery, and landscapes
    4. Cool synthetic objects, skies, or computer screens
    5. Inputs completely lacking corrugated kraft cardboard or shipping cartons

    Returns:
        Tuple of (is_ood: bool, reason: Optional[str], metrics: Dict[str, Any])
    """
    if isinstance(input_data, tf.Tensor):
        arr = input_data.numpy()
    elif isinstance(input_data, np.ndarray):
        arr = input_data.copy()
    else:
        arr = np.array(input_data, dtype=np.float32)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.max() > 1.0:
        arr = arr / 255.0

    H, W, _ = arr.shape
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # 1. Kraft Corrugated Cardboard Coverage:
    # Warm brown/tan tones characteristic of unbleached kraft paperboard
    kraft_mask = (r > b * 1.08) & (g >= b * 0.94) & (r >= 0.15) & (r <= 0.95) & (b <= 0.78)
    kraft_ratio = float(np.mean(kraft_mask))

    # 2. White / Neutral Shipping Carton / Chipboard Coverage:
    # Low chromatic variance (max color difference < 0.14) with carton brightness
    chroma = np.maximum(np.maximum(np.abs(r - g), np.abs(g - b)), np.abs(b - r))
    neutral_mask = (chroma < 0.14) & (r >= 0.25) & (r <= 0.98)
    neutral_ratio = float(np.mean(neutral_mask))

    packaging_coverage = round(kraft_ratio + 0.5 * neutral_ratio, 4)

    # 3. Luminance Y
    y_val = 0.299 * r + 0.587 * g + 0.114 * b

    # 4. Human Portrait / Face / Person Profile:
    # Top studio background brightness vs top-center head/hair region:
    top_bg_y = float(y_val[:max(1, int(H * 0.15)), :].mean())
    hair_roi_y = float(y_val[int(H * 0.15):int(H * 0.35), int(W * 0.25):int(W * 0.75)].mean())
    head_contrast = round(top_bg_y - hair_roi_y, 4)

    # Facial skin in central region:
    face_roi = arr[int(H * 0.25):int(H * 0.70), int(W * 0.25):int(W * 0.75)]
    f_r, f_g, f_b = face_roi[..., 0], face_roi[..., 1], face_roi[..., 2]
    skin_in_face = (f_r > f_g) & (f_g > f_b) & ((f_r - f_g) >= 0.05) & ((f_g - f_b) >= 0.01)
    face_skin_ratio = round(float(np.mean(skin_in_face)), 4)

    is_portrait = bool(head_contrast > 0.20 and face_skin_ratio > 0.02 and kraft_ratio < 0.25)

    # 5. Hand / Direct Skin Close-Up:
    all_skin = (r > g) & (g > b) & ((r - g) >= 0.06) & ((g - b) >= 0.02)
    skin_ratio = round(float(np.mean(all_skin)), 4)
    is_hand_skin = bool(skin_ratio > 0.50 and kraft_ratio < 0.08)

    # 6. Outdoor Greenery / Foliage:
    foliage_mask = (g > r * 1.18) & (g > b * 1.18) & (g > 0.22)
    foliage_ratio = round(float(np.mean(foliage_mask)), 4)
    is_foliage = bool(foliage_ratio > 0.18)

    # 7. Cool Synthetic Scene (Sky, water, cool fabrics without cardboard):
    cool_mask = (b > r * 1.25) & (b > 0.30)
    cool_ratio = round(float(np.mean(cool_mask)), 4)
    is_cool_ood = bool(cool_ratio > 0.45 and kraft_ratio < 0.04)

    # 8. Complete Absence of Packaging Material:
    is_lacking_packaging = bool(kraft_ratio < 0.02 and neutral_ratio < 0.05)

    reasons = []
    if is_portrait:
        reasons.append(f"Human portrait / facial features detected (contrast: {head_contrast:.2f}, skin: {face_skin_ratio * 100:.1f}%)")
    elif is_hand_skin:
        reasons.append(f"Human hand / direct skin close-up detected ({skin_ratio * 100:.1f}%)")

    if is_foliage:
        reasons.append(f"Outdoor greenery / foliage detected ({foliage_ratio * 100:.1f}%)")

    if is_cool_ood:
        reasons.append(f"Cool synthetic / non-packaging scene detected ({cool_ratio * 100:.1f}%)")

    if is_lacking_packaging and not reasons:
        reasons.append(f"Lacks packaging characteristics (kraft: {kraft_ratio * 100:.1f}%, carton: {neutral_ratio * 100:.1f}%)")

    is_ood = len(reasons) > 0
    reason_str = "; ".join(reasons) if is_ood else "Packaging characteristics verified"

    metrics = {
        "is_ood": is_ood,
        "reason": reason_str,
        "kraft_ratio": round(kraft_ratio, 4),
        "neutral_ratio": round(neutral_ratio, 4),
        "packaging_coverage": packaging_coverage,
        "head_contrast": head_contrast,
        "face_skin_ratio": face_skin_ratio,
        "skin_ratio": skin_ratio,
        "foliage_ratio": foliage_ratio,
        "cool_ratio": cool_ratio,
    }
    return is_ood, (reason_str if is_ood else None), metrics


def quantify_uncertainty(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    standard_probabilities: np.ndarray,
    mc_samples: int = 10,
    entropy_weight: float = 0.6,
    mc_weight: float = 0.4,
) -> Dict[str, Any]:
    """Computes a comprehensive Trustworthy AI uncertainty evaluation.

    Combines:
    1. Packaging Domain Verification & True Out-of-Distribution (OOD) Protection
    2. Temperature-scaled Energy OOD score
    3. Normalized Shannon Predictive Entropy
    4. Monte Carlo Dropout epistemic uncertainty estimation

    When an Out-of-Distribution input is detected (e.g. human portrait, hand, landscape,
    or random non-packaging item), epistemic uncertainty is immediately set to 100% (1.0)
    to protect automated warehouse dispatch systems.
    """
    # 1. Packaging Domain & OOD Verification Check
    is_domain_ood, ood_reason, domain_metrics = verify_packaging_domain(input_tensor)

    # 2. Temperature-scaled Energy-Based OOD Score
    energy_score = compute_energy_score(standard_probabilities, temperature=2.0)

    # 3. Predictive entropy from standard inference
    entropy_score = compute_predictive_entropy(standard_probabilities)

    # 4. MC Dropout estimation
    mc_probs, mc_var, mc_uncertainty = compute_mc_dropout(
        model, input_tensor, num_samples=mc_samples
    )

    # 5. Combined uncertainty evaluation & True OOD override
    if is_domain_ood:
        # Full epistemic uncertainty override for Out-of-Distribution inputs
        combined_uncertainty = 1.0
        mc_var = 0.25
        mc_uncertainty = 1.0
        is_ood = True
    else:
        if mc_var > 0:
            combined_uncertainty = (
                entropy_weight * entropy_score + mc_weight * mc_uncertainty
            )
        else:
            combined_uncertainty = entropy_score

        combined_uncertainty = round(float(np.clip(combined_uncertainty, 0.0, 1.0)), 4)
        is_ood = False

    return {
        "uncertainty_score": combined_uncertainty,
        "entropy": entropy_score,
        "mc_variance": mc_var,
        "mc_uncertainty": mc_uncertainty,
        "mc_probabilities": [round(float(p), 4) for p in mc_probs],
        "energy_score": energy_score,
        "is_ood": is_ood,
        "ood_reason": ood_reason,
        "domain_metrics": domain_metrics,
    }
