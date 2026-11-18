"""Inference route module for DefektGuard.

Handles image uploads, security validation, inference execution, and scan logging.
"""

import io
import logging
from flask import Blueprint, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

import config
from model.model import predict
from storage import storage

logger = logging.getLogger("defektguard.routes.predict")
predict_bp = Blueprint("predict_bp", __name__)


def is_allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has a permitted image extension."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS


@predict_bp.route("/predict", methods=["POST"])
def predict_route():
    # 1. Check for file attachment
    file_key = "image" if "image" in request.files else ("file" if "file" in request.files else None)
    if not file_key:
        return jsonify({
            "error": "No image file provided in request",
            "code": "MISSING_IMAGE_PAYLOAD",
            "required_field": "image",
        }), 400

    file = request.files[file_key]
    if not file or file.filename == "":
        return jsonify({
            "error": "Empty filename or no file selected",
            "code": "EMPTY_FILE",
        }), 400

    # 2. Sanitize and validate filename extension
    safe_name = secure_filename(file.filename)
    if not is_allowed_file(safe_name):
        return jsonify({
            "error": f"Unsupported file extension. Allowed formats: {sorted(list(config.ALLOWED_EXTENSIONS))}",
            "code": "UNSUPPORTED_MEDIA_TYPE",
            "filename": safe_name,
        }), 415

    # 3. Read and validate image content bytes
    try:
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return jsonify({
                "error": "Uploaded file is 0 bytes (empty)",
                "code": "EMPTY_IMAGE_BYTES",
            }), 400

        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verifies file integrity
        # Reopen after verify() since verify() can mutate internal state
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as e:
        logger.warning(f"Corrupted or invalid image uploaded: {safe_name} ({e})")
        return jsonify({
            "error": "Uploaded file is corrupted or not a valid decodeable image",
            "code": "INVALID_IMAGE_DATA",
            "details": str(e),
        }), 422
    except Exception as e:
        logger.error(f"Unexpected error decoding image: {e}")
        return jsonify({
            "error": "Failed to process image payload",
            "code": "IMAGE_DECODE_FAILED",
            "details": str(e),
        }), 400

    # 4. Run ML Inference with Trustworthy AI capabilities
    try:
        result = predict(image)
        # 5. Log to storage
        sku = request.form.get("sku")
        location = request.form.get("location")
        record = storage.log_scan(
            label=result["label"],
            confidence=result["confidence"],
            filename=safe_name,
            probabilities=result["probabilities"],
            sku=sku,
            location=location,
            uncertainty_score=result.get("uncertainty_score"),
            triage_status=result.get("triage_status"),
            requires_human_review=result.get("requires_human_review"),
            gradcam_image=result.get("gradcam_image"),
        )
        result["scan_id"] = record["id"]
        result["timestamp"] = record["timestamp"]
        return jsonify(result), 200

    except FileNotFoundError as e:
        logger.error(f"Model file not loaded: {e}")
        return jsonify({
            "error": "Model artifact not available. Please train or provide weights.",
            "code": "MODEL_UNAVAILABLE",
            "details": str(e),
        }), 503
    except Exception as e:
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        return jsonify({
            "error": "Inference computation failed",
            "code": "INFERENCE_ERROR",
            "details": str(e),
        }), 500
