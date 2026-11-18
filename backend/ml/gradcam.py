"""Grad-CAM (Gradient-weighted Class Activation Mapping) module for DefektGuard.

Provides visual explainability for ResNet-34 predictions by highlighting
image regions that contributed most to a specific classification decision.
"""

import base64
import io
import logging
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import tensorflow as tf

logger = logging.getLogger("defektguard.ml.gradcam")


def find_target_conv_layer(model: tf.keras.Model) -> str:
    """Finds the last convolutional or residual output layer suitable for Grad-CAM."""
    # Look for known ResNet-34 terminal stage layers first
    preferred_layers = [
        "stage4_block3_out",
        "stage4_block3_conv2",
        "stage4_block3_add",
        "stage4_block2_out",
    ]
    for name in preferred_layers:
        try:
            layer = model.get_layer(name)
            if len(layer.output.shape) == 4:
                return name
        except (ValueError, AttributeError):
            continue

    # Fallback: search backwards for the last layer with 4D output (batch, h, w, channels)
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("Could not find a 4D convolutional feature map layer for Grad-CAM.")


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_tensor: tf.Tensor,
    class_index: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Computes Grad-CAM 2D heatmap normalized to [0, 1].

    Args:
        model: Trained tf.keras.Model.
        img_tensor: Preprocessed image tensor with shape (1, H, W, 3).
        class_index: Target class index to explain. If None, uses top predicted class.
        layer_name: Target conv layer name. If None, auto-detected.

    Returns:
        2D numpy array with values in [0, 1] representing activation intensity.
    """
    if layer_name is None:
        layer_name = find_target_conv_layer(model)

    target_layer = model.get_layer(layer_name)

    # Build gradient model connecting input -> [target_layer_output, final_prediction]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_index]

    # Compute gradients of class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        logger.warning("Gradient is None; using uniform fallback heatmap.")
        return np.zeros((img_tensor.shape[1], img_tensor.shape[2]), dtype=np.float32)

    # Global average pooling of gradients -> importance weights alpha_k
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature map channels
    conv_outputs_val = conv_outputs[0]
    heatmap = tf.tensordot(conv_outputs_val, pooled_grads, axes=([2], [0]))

    # Apply ReLU to retain features that positively correlate with the target class
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 1e-8:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()


def generate_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, str]:
    """Overlays the Grad-CAM heatmap onto the original image.

    Args:
        original_image: Source PIL Image.
        heatmap: 2D numpy array [0, 1].
        alpha: Blending weight for heatmap overlay (0 to 1).
        colormap_name: Matplotlib colormap name ('jet', 'viridis', 'magma', etc.).

    Returns:
        Tuple of (overlay_pil_image, base64_data_uri).
    """
    rgb_img = original_image.convert("RGB")
    width, height = rgb_img.size

    # Resize heatmap to match original image dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_resized = heatmap_img.resize((width, height), resample=Image.Resampling.BILINEAR)
    heatmap_resized_arr = np.array(heatmap_resized) / 255.0

    # Colorize heatmap using matplotlib colormap
    try:
        cmap = matplotlib.colormaps[colormap_name]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap_resized_arr)[:, :, :3]  # Drop alpha channel -> RGB in [0, 1]
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)

    # Blend original and colored heatmap
    orig_arr = np.array(rgb_img, dtype=np.float32)
    overlay_arr = orig_arr * (1.0 - alpha) + colored_heatmap_uint8 * alpha
    overlay_arr = np.clip(overlay_arr, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlay_arr)

    # Encode as Base64 data URI
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="JPEG", quality=90)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64_str}"

    return overlay_img, data_uri
