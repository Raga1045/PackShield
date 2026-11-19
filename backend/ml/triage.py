"""Human-in-the-Loop (HITL) safe triage and decision fallback module.

Implements automated risk-aware triage according to Trustworthy AI principles:
- High confidence & Low uncertainty: Automated dispatch (Auto-Approved / Auto-Rejected)
- Low confidence or High uncertainty: Safe fallback routed to human auditor
"""

from typing import Any, Dict, Optional


# Configurable safety thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_UNCERTAINTY_THRESHOLD = 0.20


def evaluate_triage(
    label: str,
    confidence: float,
    uncertainty_score: float,
    is_ood: bool = False,
    ood_reason: Optional[str] = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluates triage status and determines if human review is required.

    Args:
        label: Predicted class ('intact', 'damaged', or 'out_of_distribution').
        confidence: Top prediction confidence score [0, 1].
        uncertainty_score: Epistemic/aleatoric uncertainty score [0, 1].
        is_ood: Boolean flag indicating Out-of-Distribution anomaly.
        ood_reason: Optional diagnostic string describing why OOD was flagged.
        confidence_threshold: Minimum confidence for automated routing (e.g. 0.85).
        uncertainty_threshold: Maximum uncertainty allowed for automated routing (e.g. 0.20).

    Returns:
        Dict containing triage status, human review flag, safety level, action, and explanation.
    """
    # 1. Immediate True Out-of-Distribution (OOD) Protection
    if is_ood and (uncertainty_score >= 0.90 or label.upper() == "OUT_OF_DISTRIBUTION"):
        explanation = (
            f"Out-of-Distribution Anomaly: Non-packaging object detected (Epistemic Uncertainty: {uncertainty_score * 100:.1f}%). "
            f"{ood_reason or 'Image lacks packaging material or carton characteristics'}. Automation blocked; routed to human inspector."
        )
        return {
            "triage_status": "FLAGGED_FOR_HUMAN_REVIEW",
            "requires_human_review": True,
            "action": "OUT_OF_DISTRIBUTION_REJECTED",
            "safety_level": "CAUTION",
            "explanation": explanation,
            "thresholds_applied": {
                "min_confidence": confidence_threshold,
                "max_uncertainty": uncertainty_threshold,
            },
        }

    reasons = []
    if uncertainty_score > uncertainty_threshold:
        reasons.append(
            f"Uncertainty ({uncertainty_score * 100:.1f}%) exceeds safety threshold ({uncertainty_threshold * 100:.0f}%)"
        )
    if confidence < confidence_threshold:
        reasons.append(
            f"Confidence ({confidence * 100:.1f}%) below minimum automation threshold ({confidence_threshold * 100:.0f}%)"
        )
    if is_ood and not reasons:
        reasons.append(ood_reason or "Potential Out-of-Distribution or low-margin camera frame detected")

    requires_human_review = len(reasons) > 0

    if requires_human_review:
        triage_status = "FLAGGED_FOR_HUMAN_REVIEW"
        action = "HUMAN_VERIFICATION_REQUIRED"
        safety_level = "CAUTION"
        explanation = "; ".join(reasons) + ". Routed to human inspector for safety verification."
    else:
        requires_human_review = False
        safety_level = "OPTIMAL"
        if label.lower() == "intact":
            triage_status = "AUTO_APPROVED"
            action = "DISPATCH_INTACT"
            explanation = "High confidence and low uncertainty. Package automatically verified intact."
        else:
            triage_status = "AUTO_REJECTED"
            action = "DIVERT_DAMAGED"
            explanation = "High confidence defect detection with low uncertainty. Package automatically quarantined."

    return {
        "triage_status": triage_status,
        "requires_human_review": requires_human_review,
        "action": action,
        "safety_level": safety_level,
        "explanation": explanation,
        "thresholds_applied": {
            "min_confidence": confidence_threshold,
            "max_uncertainty": uncertainty_threshold,
        },
    }
