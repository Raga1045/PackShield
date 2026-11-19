# PackShield – Safe and Explainable AI for Automated Package Defect Inspection

PackShield is an AI-powered package inspection system designed to identify damaged and intact packages using deep learning. It combines explainable AI, uncertainty estimation, out-of-distribution (OOD) detection, and Human-in-the-Loop (HITL) verification to make automated inspection more transparent, reliable, and accountable.

The system classifies package images into two categories: **Intact** and **Damaged**. Along with its prediction, it provides visual explanations and uncertainty information to help determine whether a prediction can be accepted automatically or should be reviewed by a human auditor.

---

## Problem Statement

Automated package inspection systems can make incorrect predictions, particularly when they encounter unclear images, unfamiliar inputs, or damaged packages with subtle defects.

A prediction alone does not explain why a package was classified in a particular way. Additionally, systems that do not account for uncertainty may confidently produce incorrect results.

PackShield addresses these challenges by integrating:

- Deep learning-based package defect classification.
- Explainable AI using Grad-CAM.
- Uncertainty estimation using Shannon Predictive Entropy and Monte Carlo Dropout.
- Out-of-Distribution (OOD) detection.
- Human-in-the-Loop verification for uncertain predictions.
- Audit records for inspection history and accountability.

## Key Features

- **Automated Defect Detection:** Classifies package images as Intact or Damaged using ResNet-34.
- **Explainable Predictions:** Uses Grad-CAM heatmaps to highlight image regions that influence the model's prediction.
- **Uncertainty Estimation:** Uses Shannon Predictive Entropy and Monte Carlo Dropout to assess prediction uncertainty.
- **OOD Detection:** Identifies inputs that may fall outside the model's expected data distribution.
- **Human-in-the-Loop Verification:** Routes uncertain or unfamiliar cases for human review.
- **Auditor Override:** Allows a human auditor to review and override an automated prediction.
- **Inspection Dashboard:** Displays predictions, confidence information, visual explanations, and inspection history.
- **Audit Logging:** Maintains inspection records and auditor decisions.

---

## System Architecture

PackShield follows a four-tier architecture.

<img width="645" height="986" alt="WhatsApp Image 2026-11-20 at 12 38 25 AM" src="https://github.com/user-attachments/assets/2c4ab0d6-181c-40f0-bfc3-76d6549bbb2e" />


### Tier 1: Presentation and HITL Interface

**Technologies:** React 18, Vite, WebRTC

This layer provides the user-facing interface for package inspection.

Responsibilities include:

- Uploading package images and accessing the camera interface.
- Displaying inspection results and prediction confidence.
- Visualizing Grad-CAM heatmaps.
- Displaying telemetry information.
- Providing human auditor review and override functionality.
- Displaying inspection history.

### Tier 2: Security and API Gateway

**Technology:** Python Flask REST API

This layer handles communication between the frontend and the AI model.

Responsibilities include:

- Receiving image data from the frontend.
- Validating uploaded images.
- Applying image preprocessing.
- Managing API requests and responses.
- Forwarding validated inputs to the AI inference pipeline.

### Tier 3: Trustworthy AI Core

**Technologies:** Python, PyTorch, ResNet-34, Grad-CAM

This is the main intelligence layer of PackShield.

It performs package classification and evaluates the trustworthiness of model predictions.

**Components:**

1. **ResNet-34:** A convolutional neural network used to classify package images into Intact and Damaged categories.
2. **Grad-CAM:** Generates visual heatmaps to explain which image regions influenced the model's prediction.
3. **Shannon Predictive Entropy:** Estimates prediction uncertainty by measuring the uncertainty in the model's output distribution.
4. **Monte Carlo Dropout:** Uses multiple stochastic forward passes to estimate predictive uncertainty.
5. **OOD Detection:** Helps identify inputs that differ from the expected package-image distribution.
6. **Trust Decision:** Uses prediction reliability and uncertainty information to determine whether a case can proceed automatically or requires human review.

### Tier 4: Data Storage and Audit

**Technologies:** JSON, MongoDB

This layer maintains inspection information and supports traceability.

Responsibilities include:

- Storing inspection records.
- Maintaining prediction results and timestamps.
- Recording auditor decisions and overrides.
- Maintaining audit logs.
- Supporting inspection history and result retrieval.

---

## System Workflow

The overall inspection process follows these steps:

1. **Image Input:** The user uploads a package image or accesses the camera interface.
2. **Image Validation:** The Flask API receives and validates the image.
3. **Preprocessing:** The image is prepared for model inference.
4. **Classification:** ResNet-34 predicts whether the package is Intact or Damaged.
5. **Explainability:** Grad-CAM generates a heatmap highlighting important image regions.
6. **Uncertainty Estimation:** Shannon Predictive Entropy and Monte Carlo Dropout help assess prediction uncertainty.
7. **OOD Assessment:** The system checks whether the input may be outside the expected data distribution.
8. **Trust Decision:** The system determines whether the prediction can be handled automatically or requires human review.
9. **Human Verification:** Uncertain or OOD cases are sent to the auditor for inspection and possible override.
10. **Audit and Storage:** Inspection results and auditor decisions are recorded.
11. **Result Display:** The frontend displays the prediction, explanation, and relevant inspection information.

---

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React 18, Vite |
| Backend | Python, Flask |
| Deep Learning | PyTorch |
| Classification Model | ResNet-34 |
| Explainable AI | Grad-CAM |
| Uncertainty Estimation | Shannon Predictive Entropy, Monte Carlo Dropout |
| Communication | Flask REST API |
| Camera Interface | WebRTC |
| Storage | JSON, MongoDB |
| Human Review | Human-in-the-Loop (HITL) |

---

## Model Evaluation

The classification model was evaluated on a **holdout subset of 360 images**.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 96.67% |
| Precision | 97.19% |
| Recall | 96.11% |
| Evaluation Samples | 360 |

These results represent the model's performance on the evaluated holdout dataset. They should not be interpreted as results from a real-world warehouse deployment.

---

## Explainable and Trustworthy AI

PackShield incorporates multiple techniques to improve the transparency and reliability of automated predictions.

| Technique | Purpose |
|---|---|
| ResNet-34 | Classifies package images as Intact or Damaged. |
| Grad-CAM | Highlights image regions influencing the prediction. |
| Shannon Predictive Entropy | Measures uncertainty in the predicted class distribution. |
| Monte Carlo Dropout | Estimates predictive uncertainty through repeated stochastic inference. |
| OOD Detection | Helps identify unfamiliar inputs. |
| Human-in-the-Loop | Enables human verification of uncertain cases. |
| Audit Logging | Maintains records of inspection results and human decisions. |

The combination of these techniques aims to make the inspection process more explainable, reliable, and accountable.

---

## Project Structure

The following is the intended high-level project organization. Update the directory names if your repository uses a different structure.

```text
PackShield/
│
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── model/
│
├── README.md
└── .gitignore
```

---

## Installation and Setup

### Prerequisites

Make sure the following tools are installed:

- Git
- Node.js and npm
- Python 3
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/PackShield.git
cd PackShield
```
---
### 2. Backend Setup (Flask API)
Open a terminal and run:
```bash
# Navigate to backend
cd backend
# Create virtual environment
python -m venv venv
# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
# Start the Flask backend server (http://127.0.0.1:5000)
python app.py
```
---
### 3. Frontend Setup (React App)
Open a **new** terminal window and run:
```bash
# Navigate to frontend
cd PackShield/frontend
# Install dependencies
npm install
# Start the React application (http://localhost:3000)
npm start
```
---

## Future Enhancements

Potential improvements to PackShield include:

- Expanding the dataset with more package types and defect categories.
- Improving OOD detection and uncertainty calibration.
- Evaluating performance under different lighting and environmental conditions.
- Integrating with real industrial conveyor-belt inspection systems.
- Enhancing audit analytics and inspection reporting.
- Exploring additional explainability techniques.

---

## Team Members

| Name | Roll Number|
|---|---|
| Akshitha Muttangi | 2024BCS-038 |
| Raga Hasini Kalluri | 2024BCS-030 |


---

## Conclusion

PackShield demonstrates how deep learning can be combined with explainable AI, uncertainty estimation, OOD detection, and human oversight to build a more trustworthy package inspection system.

Rather than relying only on automated predictions, PackShield incorporates transparency, uncertainty awareness, and human verification to support safer and more accountable AI-assisted inspection.

**PackShield – Making Automated Package Inspection Explainable, Reliable, and Human-Centred.**
