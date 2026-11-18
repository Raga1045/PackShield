"""Dashboard stats, scan history, and Human-in-the-Loop review endpoints for DefektGuard."""

from flask import Blueprint, jsonify, request
from storage import storage

stats_bp = Blueprint("stats_bp", __name__)


@stats_bp.route("/stats", methods=["GET"])
@stats_bp.route("/dashboard", methods=["GET"])
def get_stats():
    """Returns aggregated inventory and scanning statistics including Trustworthy AI metrics."""
    stats = storage.get_dashboard_stats()
    return jsonify(stats), 200


@stats_bp.route("/history", methods=["GET"])
@stats_bp.route("/details", methods=["GET"])
def get_history():
    """Returns historical scan audit records with optional pagination."""
    limit = request.args.get("limit", default=100, type=int)
    records = storage.get_records(limit=min(limit, 500))
    return jsonify(records), 200


@stats_bp.route("/scans/flagged", methods=["GET"])
def get_flagged_scans():
    """Returns scans currently flagged for human verification."""
    limit = request.args.get("limit", default=50, type=int)
    flagged = storage.get_flagged_records(limit=min(limit, 100))
    return jsonify({"count": len(flagged), "records": flagged}), 200


@stats_bp.route("/scans/<scan_id>/review", methods=["POST"])
def submit_human_review(scan_id: str):
    """Allows human auditor to confirm or override an automated or flagged scan."""
    data = request.get_json(silent=True) or {}
    decision = data.get("human_decision")
    if not decision:
        return jsonify({
            "error": "Missing required field 'human_decision' ('intact' or 'damaged')",
            "code": "MISSING_DECISION",
        }), 400

    decision = str(decision).lower().strip()
    if decision not in ("intact", "damaged"):
        return jsonify({
            "error": f"Invalid decision '{decision}'. Must be 'intact' or 'damaged'.",
            "code": "INVALID_DECISION",
        }), 400

    notes = data.get("notes")
    reviewer = data.get("reviewer_name", "Human Auditor")

    try:
        updated = storage.update_review(
            scan_id=scan_id,
            human_decision=decision,
            notes=notes,
            reviewer_name=reviewer,
        )
        if not updated:
            return jsonify({
                "error": f"Scan with ID '{scan_id}' not found.",
                "code": "SCAN_NOT_FOUND",
            }), 404

        return jsonify({
            "message": "Human audit decision successfully logged",
            "record": updated,
        }), 200
    except Exception as e:
        return jsonify({
            "error": f"Failed to record review: {str(e)}",
            "code": "REVIEW_ERROR",
        }), 500
