# DefektGuard: Trustworthy AI Implementation Report

## 1. Project Overview & Problem Formulation
DefektGuard is an end-to-end, computer-vision-based packaging defect detection and inventory triage system. In retail and warehouse supply chains, detecting package damage (tears, punctures, crushed corners) is critical. However, standard deep learning models act as black boxes and exhibit overconfidence when exposed to out-of-distribution (OOD) packaging materials or corrupted camera frames.

To elevate DefektGuard into an industry-grade, compliant system aligned with academic Trustworthy AI criteria, the architecture has been upgraded across three foundational pillars:
1. **Explainability & Interpretability (XAI)**
2. **Robustness & Uncertainty Quantification**
3. **Human-in-the-Loop (HITL) Safe Fallback & Triage**

---

## 2. Pillar 1: Explainability & Interpretability (Grad-CAM)
### Theoretical Formulation
Standard defect classification does not reveal whether the decision was made based on actual structural damage or spurious background features (e.g., lighting variations, conveyer belt markings).

We implement **Grad-CAM (Gradient-weighted Class Activation Mapping)** on the terminal residual convolutional block of the ResNet-34 model (`stage4_block3_out`):
1. Compute gradients of the target class score $y^c$ with respect to feature activation maps $A^k$:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{ij}^k}$$
2. Take a rectified linear combination of feature maps weighted by $\alpha_k^c$:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
3. Resize the 2D activation map to input dimensions $(256 \times 256)$ and project a JET colormap overlay ($\alpha = 0.45$) onto the input image.

### System Integration
- Backend Module: [`backend/ml/gradcam.py`](file:///c:/Users/ragah/Downloads/DefktGuard/backend/ml/gradcam.py)
- API Payload: Returned as a Base64 JPEG data URI (`gradcam_image`) with zero local disk footprint.
- Frontend UI: The scanner interface ([`Scan.jsx`](file:///c:/Users/ragah/Downloads/DefktGuard/frontend/src/pages/Scan.jsx)) provides an interactive toggle between the original image and the Grad-CAM explainability heatmap.

---

## 3. Pillar 2: Robustness & Uncertainty Quantification
### Theoretical Formulation
Deep neural networks are known to make overconfident mistakes on ambiguous or degraded inputs. We integrate two complementary uncertainty measures:

1. **Normalized Shannon Information Entropy (Aleatoric / Predictive Ambiguity)**:
   $$H(p) = -\frac{1}{\log_2(C)} \sum_{i=1}^C p_i \log_2(p_i)$$
   Bounded in $[0, 1]$, where $H(p) = 0$ indicates total model certainty, and $H(p) = 1.0$ indicates maximal uncertainty (uniform probability distribution).

2. **Monte Carlo Dropout (Epistemic / Model Uncertainty)**:
   By keeping dropout active at inference time across $N$ stochastic forward passes ($\hat{y}^{(1)}, \dots, \hat{y}^{(N)}$):
   $$\bar{p} = \frac{1}{N} \sum_{n=1}^N \hat{y}^{(n)}, \quad \sigma^2 = \frac{1}{N} \sum_{n=1}^N (\hat{y}^{(n)} - \bar{p})^2$$
   Variance $\sigma^2$ directly reflects model uncertainty due to lack of representative training data.

3. **Combined Calibrated Uncertainty Metric**:
   $$U = 0.6 \cdot H(p) + 0.4 \cdot U_{\text{epistemic}}$$

### System Integration
- Backend Module: [`backend/ml/uncertainty.py`](file:///c:/Users/ragah/Downloads/DefktGuard/backend/ml/uncertainty.py)
- Calibration & Robustness Evaluation: [`backend/benchmark_trustworthy.py`](file:///c:/Users/ragah/Downloads/DefktGuard/backend/benchmark_trustworthy.py) computes **Expected Calibration Error (ECE)** and validates that uncertainty monotonically increases under image degradation (noise, blur, low light).

---

## 4. Pillar 3: Human-in-the-Loop (HITL) Safe Fallback
### Decision Policy & Triage Thresholds
To prevent costly false positives (discarding intact merchandise) and false negatives (shipping damaged items), inference decisions are automatically triaged:

```
IF Confidence >= 85% AND Uncertainty <= 20%:
    -> AUTOMATED ACTION:
       - If predicted Intact: AUTO_APPROVED (Dispatch)
       - If predicted Damaged: AUTO_REJECTED (Quarantine)
ELSE:
    -> SAFE FALLBACK:
       - Status: FLAGGED_FOR_HUMAN_REVIEW
       - Action: HUMAN_VERIFICATION_REQUIRED
       - Reason: Low confidence or high epistemic uncertainty detected
```

### Audit Traceability & Operator Actions
- Scans flagged for review trigger an amber alert in the Scanner UI and appear in the Dashboard and Details pages.
- Warehouse auditors can review the Grad-CAM heatmap side-by-side with the box and execute a one-click **"Verify Intact"** or **"Confirm Damaged"** audit action via `POST /scans/<scan_id>/review`.
- All decisions, timestamps, and operator overrides are permanently persisted in audit records.

---

## 5. Verification & Testing Summary
- **Backend Unit & Integration Tests**: 23/23 tests passing (`test_model.py`, `test_metrics.py`, `test_api.py`, `test_trustworthy.py`).
- **Frontend Production Build**: Clean build with zero compile errors (`npm run build`).
- **Reproducible Benchmark Script**: `python backend/benchmark_trustworthy.py --samples 15` validates ECE and synthetic corruption interception.

---

## 6. Regulatory & Academic Standards Alignment
- **EU AI Act**: Fulfills requirements for High-Risk AI systems under Article 13 (Transparency and provision of information to deployers) and Article 14 (Human oversight).
- **NIST AI Risk Management Framework (AI RMF 1.0)**: Aligns with the core functions: **Govern, Map, Measure, and Manage** for safety-critical computer vision deployment.
