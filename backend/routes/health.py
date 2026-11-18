"""Health check endpoint for DefektGuard API."""

from flask import Blueprint, jsonify
import config
from model.model import _cached_model, get_model

health_bp = Blueprint("health_bp", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    model_loaded = False
    model_path = str(config.DEFAULT_MODEL_PATH)
    try:
        model = get_model()
        model_loaded = model is not None
    except Exception:
        model_loaded = False

    return jsonify({
        "status": "healthy",
        "service": "DefektGuard API",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "default_model_path": model_path,
        "input_shape": list(config.INPUT_SHAPE),
        "classes": config.CLASS_NAMES,
    }), 200

