import React, { useState } from "react";
import "./../styles/modal.css";
import toast from "react-hot-toast";

const DetailModal = ({ image, item, onClose, onReviewed }) => {
  const [currentView, setCurrentView] = useState(item?.gradcam_image ? "gradcam" : "original");
  const [submitting, setSubmitting] = useState(false);
  const backendBase = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";

  const handleAudit = (decision) => {
    if (!item?.id) return;
    setSubmitting(true);
    fetch(`${backendBase}/scans/${item.id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        human_decision: decision,
        notes: "Audited from historical details view.",
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Review submission failed");
        return res.json();
      })
      .then((data) => {
        toast.success(`Scan ${item.id} verified as ${decision.toUpperCase()}.`);
        if (onReviewed) {
          onReviewed(item.id, decision);
        }
        onClose();
      })
      .catch((err) => {
        console.error(err);
        toast.error("Failed to submit audit decision.");
      })
      .finally(() => setSubmitting(false));
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>✕</button>

        {item && (
          <div className="modal-header-info">
            <h3>{item.sku || item.productName || item.filename || "Scan Inspection"}</h3>
            <span className="modal-id-badge">{item.id || "N/A"}</span>
          </div>
        )}

        {item?.gradcam_image && (
          <div className="modal-view-toggle">
            <button
              className={`modal-toggle-btn ${currentView === "gradcam" ? "active" : ""}`}
              onClick={() => setCurrentView("gradcam")}
            >
              Grad-CAM Heatmap
            </button>
            <button
              className={`modal-toggle-btn ${currentView === "original" ? "active" : ""}`}
              onClick={() => setCurrentView("original")}
            >
              Original View
            </button>
          </div>
        )}

        <div className="modal-image-container">
          <img
            src={currentView === "gradcam" && item?.gradcam_image ? item.gradcam_image : (item?.image || image)}
            alt="Product Inspection"
            className="modal-image"
          />
        </div>

        {item && (
          <div className="modal-trustworthy-info">
            <div className="modal-stats-row">
              <div className="stat-pill">
                <span className="stat-pill-label">Status</span>
                <strong className={`status-${(item.label || item.damage || item.condition || "").toLowerCase()}`}>
                  {(item.human_decision || item.label || item.damage || item.condition || "UNKNOWN").toUpperCase()}
                </strong>
              </div>

              <div className="stat-pill">
                <span className="stat-pill-label">Confidence</span>
                <strong>{item.confidence}%</strong>
              </div>

              {item.uncertainty != null && (
                <div className="stat-pill">
                  <span className="stat-pill-label">Uncertainty</span>
                  <strong>{item.uncertainty}%</strong>
                </div>
              )}
            </div>

            {item.requires_human_review && (
              <div className="modal-audit-action">
                <p className="modal-audit-prompt">
                  ⚠ <strong>Flagged for Human Verification:</strong> Please confirm or override the classification.
                </p>
                <div className="modal-action-buttons">
                  <button
                    className="modal-btn-approve"
                    disabled={submitting}
                    onClick={() => handleAudit("intact")}
                  >
                    ✓ Verify Intact
                  </button>
                  <button
                    className="modal-btn-reject"
                    disabled={submitting}
                    onClick={() => handleAudit("damaged")}
                  >
                    ✕ Confirm Damaged
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default DetailModal;
