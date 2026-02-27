# 🏥 VitalsGuard Clinical Engine (API & ML Backend)

VitalsGuard is a predictive health analytics backend built with **FastAPI** and **Python**. It serves as the intelligent inference engine behind the VitalsGuard clinical dashboard, processing patient vitals and wearable data through pre-trained Machine Learning models to predict metabolic, cerebrovascular, and athletic injury risks.



## ✨ The "Ensemble Engine" Architecture

Instead of relying solely on "black box" machine learning algorithms, VitalsGuard utilizes a hybrid clinical architecture:
1. **Statistical Base (XGBoost/Random Forest):** Evaluates core biometrics against historical datasets to generate a raw baseline probability.
2. **Physiological Heuristics (Python Rule Engine):** Applies clinical "common sense" multipliers. For example, the engine dynamically calculates an "Activity Buffer" where a high daily step count physically mitigates the vascular danger of a high BMI.

This ensures predictions are not only statistically accurate but physiologically realistic and clinically actionable.

---

## 🚀 Tech Stack

* **API Framework:** FastAPI, Uvicorn, Pydantic (Strict Type Validation)
* **Machine Learning:** Scikit-Learn, XGBoost
* **Data Processing:** Pandas, NumPy, Imbalanced-Learn (SMOTE)
* **Model Serialization:** Joblib
* **Deployment:** Render (PaaS)

---

## 📊 Data Science Methodology (CRISP-DM)

The `notebooks/` directory contains the complete lifecycle of the machine learning models, structured around the CRISP-DM framework:

### 1. Business Understanding
The current healthcare system is largely reactive. VitalsGuard flips this paradigm by treating strokes, metabolic syndrome, and athletic injuries as preventable events preceded by measurable physiological signals (e.g., uncontrolled hypertension, high CNS load). The business goal is to provide early-warning clinical decision support.

### 2. Data Understanding & Preparation
* **Handling Extreme Imbalance:** The Stroke dataset features less than 5% positive cases. We implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to synthetically balance the training data, ensuring the model learns the minority class rather than defaulting to naive majority predictions.
* **Feature Engineering:** Mapped categorical lifestyle data into binary flags and engineered continuous variables for model ingestion.

### 3. Modeling
Evaluated multiple tree-based and linear algorithms (Random Forest, XGBoost, Logistic Regression). **XGBoost** was selected for its superior handling of non-linear clinical relationships. We rigorously calculated and applied exact `scale_pos_weight` ratios to correct residual distribution skews.

### 4. Evaluation (Clinical Threshold Tuning)
Out-of-the-box ML algorithms optimize for total Accuracy, which is dangerous in healthcare (resulting in high False Negatives). We manually tuned the classification probability thresholds (e.g., lowering the positive decision boundary to `0.30`) to aggressively prioritize **Recall (Sensitivity)**. This ensures latent medical risks are mathematically flagged for clinical review.

### 5. Deployment
The exported `.pkl` models are wrapped in a high-concurrency FastAPI application, exposing RESTful endpoints for real-time inference by the Next.js frontend.

---

## 📂 Project Structure

```text
├── main.py                  # FastAPI application and physiological heuristic multipliers
├── requirements.txt         # Python dependencies
├── models/                  # Serialized ML models
│   ├── stroke_model.pkl
│   ├── metabolic_model.pkl
│   └── recovery_model.pkl
├── notebooks/               # Jupyter Notebooks containing the CRISP-DM workflow
│   ├── _Stroke_Prediction_Model.ipynb
│   ├── _Metabolic_Risk_Model.ipynb
│   └── _Athletic_Injury_Model.ipynb
└── README.md
