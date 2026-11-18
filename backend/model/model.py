"""Model inference service module for DefektGuard.

Provides cached model loading, image preprocessing, and inference prediction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

import config
from ml.gradcam import compute_gradcam_heatmap, generate_gradcam_overlay
from ml.triage import evaluate_triage
from ml.uncertainty import quantify_uncertainty

logger = logging.getLogger("defektguard.model")

# Module-level model cache
_cached_model: Optional[tf.keras.Model] = None
_cached_model_path: Optional[str] = None


def get_model(model_path: Optional[Union[str, Path]] = None) -> tf.keras.Model:
    """Loads and caches the model singleton.
    
    Tries configured path, then default .keras, then fallback .h5.
    """
    global _cached_model, _cached_model_path

    target_path = Path(model_path) if model_path else config.DEFAULT_MODEL_PATH

    # Check fallback if primary does not exist
    if not target_path.exists():
        fallback = target_path.with_suffix(".h5")
        if fallback.exists():
            target_path = fallback
        elif config.FALLBACK_H5_MODEL_PATH.exists():
            target_path = config.FALLBACK_H5_MODEL_PATH
        else:
            raise FileNotFoundError(
                f"DefektGuard model not found at {target_path} or {config.FALLBACK_H5_MODEL_PATH}. "
                "Please run `python train.py` to train or download the model artifact."
            )

    str_path = str(target_path)
    if _cached_model is None or _cached_model_path != str_path:
        logger.info(f"Loading DefektGuard model from: {str_path}")
        _cached_model = tf.keras.models.load_model(str_path)
        _cached_model_path = str_path

    return _cached_model


def preprocess_image(image: Image.Image) -> tf.Tensor:
    """Preprocesses a PIL Image for model input:
    1. Resizes to (256, 256)
    2. Converts to RGB float32 array
    3. Normalizes to [0, 1] range
    4. Expands dimensions to (1, 256, 256, 3)
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    resized = image.resize(config.IMG_SIZE, Image.Resampling.BILINEAR)
    arr = img_to_array(resized)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    tensor = tensor / 255.0
    return tensor


def predict(
    image: Image.Image,
    model_path: Optional[Union[str, Path]] = None,
    explain: bool = True,
    quantify_risk: bool = True,
) -> Dict[str, Any]:
    """Runs inference on a PIL image and returns structured predictions with Trustworthy AI metrics.

    Args:
        image: PIL Image input.
        model_path: Optional path to load model from.
        explain: Whether to compute Grad-CAM visual explainability overlay.
        quantify_risk: Whether to compute uncertainty and safe triage recommendations.

    Returns:
        Dictionary containing predictions, uncertainty scores, Grad-CAM heatmap data URI,
        and human-in-the-loop triage status.
    """
    try:
        model = get_model(model_path)
        processed = preprocess_image(image)
        raw_output = model.predict(processed, verbose=0)[0]

        label_index = int(np.argmax(raw_output))
        label = config.CLASS_NAMES[label_index]
        confidence = float(np.max(raw_output))

        probabilities = {
            cls_name: round(float(raw_output[i]), 4)
            for i, cls_name in enumerate(config.CLASS_NAMES)
        }

        # 1. Uncertainty Quantification
        uncertainty_info = {
            "uncertainty_score": 0.0,
            "entropy": 0.0,
            "mc_variance": 0.0,
            "is_ood": False,
        }
        if quantify_risk:
            try:
                uncertainty_info = quantify_uncertainty(
                    model=model,
                    input_tensor=processed,
                    standard_probabilities=raw_output,
                    mc_samples=8,
                )
            except Exception as u_err:
                logger.warning(f"Uncertainty computation error: {u_err}")

        # 2. Grad-CAM Explainability
        gradcam_info = None
        gradcam_image = None
        if explain:
            try:
                heatmap = compute_gradcam_heatmap(
                    model=model,
                    img_tensor=processed,
                    class_index=label_index,
                )
                overlay_img, data_uri = generate_gradcam_overlay(
                    original_image=image,
                    heatmap=heatmap,
                    alpha=0.45,
                )
                gradcam_image = data_uri
                gradcam_info = {
                    "data_uri": data_uri,
                    "target_class": label,
                    "target_class_index": label_index,
                }
            except Exception as g_err:
                logger.warning(f"Grad-CAM computation error: {g_err}")

        # 3. Human-in-the-Loop Safe Triage
        triage_info = evaluate_triage(
            label=label,
            confidence=confidence,
            uncertainty_score=uncertainty_info.get("uncertainty_score", 0.0),
            is_ood=uncertainty_info.get("is_ood", False),
        )

        return {
            # Legacy/Baseline fields (100% backward compatible)
            "label": label,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "raw_output": [round(float(x), 4) for x in raw_output],
            # Trustworthy AI Pillars
            "uncertainty": uncertainty_info,
            "uncertainty_score": uncertainty_info.get("uncertainty_score", 0.0),
            "gradcam": gradcam_info,
            "gradcam_image": gradcam_image,
            "triage": triage_info,
            "triage_status": triage_info["triage_status"],
            "requires_human_review": triage_info["requires_human_review"],
        }
    except Exception as e:
        logger.error(f"Inference execution error: {e}", exc_info=True)
        raise
