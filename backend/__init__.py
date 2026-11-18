"""Flask Application Factory for DefektGuard."""

import logging
import sys
from pathlib import Path
from flask import Flask, jsonify
from flask_cors import CORS

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from routes.health import health_bp
from routes.predict import predict_bp
from routes.stats import stats_bp


def create_app(test_config=None):
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)
    
    # Configure app
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["DEBUG"] = config.DEBUG

    if test_config:
        app.config.update(test_config)

    # Enable CORS for all routes
    CORS(app)

    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Register Blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(stats_bp)

    # Error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            "error": "Payload exceeds maximum allowed size (10 MB limit)",
            "code": "PAYLOAD_TOO_LARGE",
        }), 413

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "error": "The requested endpoint does not exist",
            "code": "ENDPOINT_NOT_FOUND",
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            "error": "An internal server error occurred",
            "code": "INTERNAL_SERVER_ERROR",
        }), 500

    return app
