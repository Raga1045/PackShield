import React, { useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { FaCamera, FaUpload, FaShieldAlt, FaEye, FaExclamationTriangle, FaCheckCircle, FaUserCheck } from "react-icons/fa";
import "./ScanPage.css";
import toast from "react-hot-toast";

const Scan = () => {
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [imageFile, setImageFile] = useState(null);
  const [previewSrc, setPreviewSrc] = useState(null);
  const [loading, setLoading] = useState(false);

  // Inference & Trustworthy AI State
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [uncertainty, setUncertainty] = useState(null);
  const [triageStatus, setTriageStatus] = useState(null);
  const [requiresHumanReview, setRequiresHumanReview] = useState(false);
  const [triageExplanation, setTriageExplanation] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [scanId, setScanId] = useState(null);
  const [viewMode, setViewMode] = useState("gradcam"); // "original" | "gradcam"
  const [reviewedAction, setReviewedAction] = useState(null);

  const backendBase = process.env.REACT_APP_BACKEND_URL || "http://localhost:5000";
  const backendURL = `${backendBase}/predict`;

  const sendToBackend = (file, previewUrl = null) => {
    const formData = new FormData();
    formData.append("image", file);

    setLoading(true);
    setPrediction(null);
    setConfidence(null);
    setUncertainty(null);
    setTriageStatus(null);
    setRequiresHumanReview(false);
    setTriageExplanation(null);
    setGradcamImage(null);
    setScanId(null);
    setReviewedAction(null);
    if (previewUrl) {
      setPreviewSrc(previewUrl);
    }

    fetch(backendURL, {
      method: "POST",
      body: formData,
    })
      .then((res) => {
        if (!res.ok) throw new Error("Inference failed");
        return res.json();
      })
      .then((data) => {
        if (data.label) {
          setPrediction(data.label);
          setConfidence(data.confidence);
          setScanId(data.scan_id);

          // Trustworthy AI Metrics
          const unc = data.uncertainty_score != null ? data.uncertainty_score : (data.uncertainty?.uncertainty_score ?? 0);
          setUncertainty(unc);

          const triage = data.triage || {};
          const isFlagged = Boolean(data.requires_human_review || triage.requires_human_review);
          setRequiresHumanReview(isFlagged);
          setTriageStatus(data.triage_status || triage.triage_status || "AUTOMATED");
          setTriageExplanation(triage.explanation || (isFlagged ? "Elevated uncertainty requires auditor verification." : "Automated verification passed."));

          if (data.gradcam_image) {
            setGradcamImage(data.gradcam_image);
            setViewMode("gradcam");
          }

          if (data.label.toUpperCase() === "OUT_OF_DISTRIBUTION") {
            toast.error("Out-of-Distribution: Non-packaging object detected!", { icon: "⚠️" });
          } else if (isFlagged) {
            toast.error("Safety Fallback: Scan flagged for human verification!", { icon: "🛡️" });
          } else if (data.label.toLowerCase() === "damaged") {
            toast.success("Damaged item automatically detected and logged.");
          } else {
            toast.success("Package verified intact.");
          }
        }
      })
      .catch((err) => {
        console.error(err);
        alert("Failed to predict. Please ensure backend is running.");
      })
      .finally(() => {
        setLoading(false);
        setImageFile(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      });
  };

  const handleStartCamera = () => {
    setCameraOn(true);
  };

  const handleCapture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => {
          const file = new File([blob], "webcam.jpg", { type: "image/jpeg" });
          sendToBackend(file, imageSrc);
        });
    }
  }, [webcamRef]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImageFile(file);
      const objectUrl = URL.createObjectURL(file);
      sendToBackend(file, objectUrl);
    }
  };

  const handleHumanReview = (decision) => {
    if (!scanId) return;
    fetch(`${backendBase}/scans/${scanId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        human_decision: decision,
        notes: `Verified by operator from live scanner interface.`,
      }),
    })
      .then((res) => res.json())
      .then(() => {
        setReviewedAction(decision);
        setRequiresHumanReview(false);
        setPrediction(decision);
        toast.success(`Audit complete: Manually verified as ${decision.toUpperCase()}.`);
      })
      .catch((err) => {
        console.error(err);
        toast.error("Failed to submit review.");
      });
  };

  return (
    <div className="scanner-container">
      <div className="scanner-header-row">
        <div>
          <h1>Package Scanner & Trustworthy AI Verification</h1>
          <p>
            Scan a package via webcam or upload an image. The system computes predictions, Grad-CAM visual heatmaps, and epistemic uncertainty metrics.
          </p>
        </div>
        <div className="trust-badge-header">
          <FaShieldAlt className="trust-shield-icon" />
          <span>Trustworthy AI Active</span>
        </div>
      </div>

      <div className="scanner-panels">
        {/* Webcam Panel */}
        <div className="panel webcam-panel">
          <div className="panel-header webcam-header">
            <FaCamera className="icon" /> Live Camera Stream
          </div>

          <div className="camera-box">
            {cameraOn ? (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                className="webcam-feed"
              />
            ) : (
              <span className="camera-note">
                ⚠ Make sure to allow camera access!
              </span>
            )}
          </div>

          {cameraOn ? (
            <button className="open-camera-btn" onClick={handleCapture}>
              Capture & Predict
            </button>
          ) : (
            <button className="open-camera-btn" onClick={handleStartCamera}>
              Open Camera
            </button>
          )}
        </div>

        {/* Upload Image Panel */}
        <div className="panel scanned-panel">
          <div className="panel-header scanned-header">
            <FaUpload className="icon" /> Upload Package Image
          </div>
          <div className="upload-box">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="file-input"
              ref={fileInputRef}
            />
            {imageFile && (
              <p className="file-name">Selected: {imageFile.name}</p>
            )}
          </div>
        </div>
      </div>

      <div className="result-section">
        {loading && (
          <div className="result-card loading">
            <div className="spinner"></div>
            <p className="loading-text">
              Analyzing image, computing Grad-CAM & Uncertainty...
            </p>
          </div>
        )}

        {prediction && !loading && (
          <div className={`trustworthy-result-container ${prediction.toLowerCase()}`}>
            {/* Visual Explainability (Grad-CAM) */}
            <div className="explainability-card">
              <div className="card-top-header">
                <h3>
                  <FaEye style={{ marginRight: 8, verticalAlign: "middle" }} />
                  XAI Explainability: Grad-CAM
                </h3>
                <div className="tab-pill-group">
                  {gradcamImage && (
                    <button
                      className={`tab-pill ${viewMode === "gradcam" ? "active" : ""}`}
                      onClick={() => setViewMode("gradcam")}
                    >
                      Grad-CAM Heatmap
                    </button>
                  )}
                  {previewSrc && (
                    <button
                      className={`tab-pill ${viewMode === "original" ? "active" : ""}`}
                      onClick={() => setViewMode("original")}
                    >
                      Original
                    </button>
                  )}
                </div>
              </div>

              <div className="explainability-viewer">
                {viewMode === "gradcam" && gradcamImage ? (
                  <img
                    src={gradcamImage}
                    alt="Grad-CAM Activation Map"
                    className="viewer-image"
                  />
                ) : previewSrc ? (
                  <img
                    src={previewSrc}
                    alt="Original Input"
                    className="viewer-image"
                  />
                ) : (
                  <div className="placeholder-view">Image processed</div>
                )}
              </div>
              <p className="viewer-caption">
                {viewMode === "gradcam"
                  ? "Grad-CAM highlights convolutional activations that drove this prediction."
                  : "Original unprocessed package image."}
              </p>
            </div>

            {/* Decision & Trustworthy Metrics */}
            <div className="metrics-card">
              <div className="card-top-header">
                <h3>
                  <FaShieldAlt style={{ marginRight: 8, verticalAlign: "middle" }} />
                  Prediction & Safety Triage
                </h3>
                {scanId && <span className="scan-id-tag">{scanId}</span>}
              </div>

              <div className="metrics-grid">
                <div className="metric-item">
                  <span className="metric-label">Predicted Condition</span>
                  <div className="status-tag-large">
                    {prediction.toUpperCase()}
                  </div>
                </div>

                <div className="metric-item">
                  <span className="metric-label">Model Confidence</span>
                  <div className="metric-value">
                    {confidence != null ? `${(confidence * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="metric-bar-bg">
                    <div
                      className="metric-bar-fill confidence-fill"
                      style={{ width: `${Math.min(100, (confidence || 0) * 100)}%` }}
                    />
                  </div>
                </div>

                <div className="metric-item">
                  <span className="metric-label">Epistemic Uncertainty</span>
                  <div className="metric-value">
                    {uncertainty != null ? `${(uncertainty * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div className="metric-bar-bg">
                    <div
                      className={`metric-bar-fill uncertainty-fill ${
                        uncertainty > 0.20 ? "high" : uncertainty > 0.10 ? "medium" : "low"
                      }`}
                      style={{ width: `${Math.min(100, (uncertainty || 0) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* Safe Triage Banner */}
              <div
                className={`triage-banner ${
                  requiresHumanReview
                    ? "caution"
                    : prediction.toLowerCase() === "damaged"
                    ? "damaged"
                    : "approved"
                }`}
              >
                <div className="triage-banner-header">
                  {requiresHumanReview ? (
                    <>
                      <FaExclamationTriangle className="triage-icon" />
                      <strong>HUMAN-IN-THE-LOOP SAFETY FALLBACK</strong>
                    </>
                  ) : (
                    <>
                      <FaCheckCircle className="triage-icon" />
                      <strong>AUTOMATED TRIAGE: {triageStatus}</strong>
                    </>
                  )}
                </div>
                <p className="triage-reason">{triageExplanation}</p>

                {/* Interactive Human Auditor Action */}
                {requiresHumanReview && !reviewedAction && (
                  <div className="human-action-box">
                    <p className="action-prompt">
                      <FaUserCheck style={{ marginRight: 6 }} />
                      Auditor Override / Confirmation:
                    </p>
                    <div className="action-buttons">
                      <button
                        className="btn-audit-approve"
                        onClick={() => handleHumanReview("intact")}
                      >
                        ✓ Mark as Intact
                      </button>
                      <button
                        className="btn-audit-reject"
                        onClick={() => handleHumanReview("damaged")}
                      >
                        ✕ Confirm Damaged
                      </button>
                    </div>
                  </div>
                )}

                {reviewedAction && (
                  <div className="audited-confirmation">
                    ✓ Human audit logged: Package confirmed as <strong>{reviewedAction.toUpperCase()}</strong>.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="alert-note">
        <strong>Trustworthy AI Audit:</strong> All packaging scans, heatmaps, and uncertainty scores are logged for traceability. View all historical records in the{" "}
        <a href="/details" className="alert-link">
          Details Page
        </a>
        .
      </div>
    </div>
  );
};

export default Scan;
