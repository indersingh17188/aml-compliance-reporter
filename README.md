# 🏦 Explainable AML Risk Report Generator

A lightweight end-to-end machine learning demo for **Anti-Money Laundering (AML) risk scoring** using **XGBoost**, **SHAP explainability**, and **Streamlit**.

This project demonstrates how a suspicious transaction monitoring prototype can combine:

- classical machine learning,
- AML-inspired feature engineering,
- class imbalance handling,
- threshold-based risk scoring,
- SHAP-based local explanations,
- and an interactive Streamlit dashboard.

> This is a portfolio/demo project and not a production AML compliance system.

---

## 🚀 Demo Overview

The application allows a user to select a sample transaction and generate:

- AML risk probability
- Low / Medium / High risk category
- analyst recommendation
- top SHAP risk drivers
- SHAP bar visualization
- downloadable AML analyst-style report

---

## 📦 Current Version

This is the first stable version of the project:

**Version:** `v1-xgboost-shap-streamlit`

### ✅ Implemented

- XGBoost classifier for AML risk prediction
- AML-specific feature engineering
- extreme class imbalance handling using `scale_pos_weight`
- evaluation using ROC-AUC and PR-AUC
- threshold analysis for operational risk scoring
- SHAP explanation for individual predictions
- human-readable feature explanations
- Streamlit dashboard
- downloadable text report

### 🔜 Planned Next

- RAG-based AML guidance retrieval
- local knowledge base using markdown files
- analyst report enriched with retrieved guidance
- optional lightweight open-source LLM rewriting
- Docker and Hugging Face Spaces deployment

---

## 📁 Project Structure

```text
aml-risk-report-generator/
│
├── app.py
├── train_model.py
├── train_model_v2.py
├── test_explainability.py
├── requirements.txt
│
├── data/
│   ├── sample_app_inputs_v2.csv
│   └── [dataset CSV file]
│
├── models/
│   ├── xgb_aml_pipeline_v2.pkl
│   └── feature_info_v2.pkl
│
└── utils/
    ├── __init__.py
    ├── feature_engineering.py
    └── explainability.py
```

---

## 💾 Dataset

This project uses a synthetic AML transaction dataset from IBM AML-Data.

The dataset contains transaction-level information such as:

- timestamp
- originating bank
- receiving bank
- amount paid
- amount received
- payment currency
- receiving currency
- payment format
- laundering label

The target column used for supervised classification is:

```text
Is_Laundering
```

---

## 🤖 Model Development

### 📊 Baseline Model

The first model used basic transaction fields with XGBoost and class imbalance handling.

Baseline result:

```text
ROC-AUC: 0.9664
PR-AUC : 0.0480
```

The baseline achieved high recall but produced too many false positives due to the extreme class imbalance.

---

## 🛠️ Improved Feature Engineering

The second model introduced AML-inspired engineered features:

### ⏰ Time Features

- transaction hour
- transaction day
- transaction month
- transaction day of week
- weekend flag
- night transaction flag

### 💰 Amount Features

- amount difference
- absolute amount difference
- paid-to-received amount ratio
- log amount paid
- log amount received

### 🌍 Currency & Bank Movement Features

- same currency flag
- cross-currency flag
- same bank transfer flag
- cross-bank transfer flag

### 🔒 Leakage Reduction

Account identifiers and raw timestamps were excluded from the model feature set to reduce memorization and improve generalization.

---

## 📈 Improved Model Results

After feature engineering:

```text
ROC-AUC: 0.9712
PR-AUC : 0.2273
```

Feature engineering improved PR-AUC from:

```text
0.0480 → 0.2273
```

This is approximately a **4.7× improvement** in PR-AUC, which is more meaningful than ROC-AUC for highly imbalanced AML data.

---

## 🎯 Threshold Analysis

Because AML datasets are highly imbalanced, the default `0.50` classification threshold is not always operationally useful.

The model was evaluated at multiple thresholds:

```text
Threshold    Precision    Recall       F1           Flagged
0.50         0.0068       0.9652       0.0134       147626
0.70         0.0143       0.8551       0.0281        61959
0.80         0.0182       0.8135       0.0355        46381
0.90         0.0316       0.6570       0.0603        21522
0.95         0.0988       0.3942       0.1580         4131
0.98         0.6140       0.1614       0.2555          272
0.99         0.9312       0.1440       0.2494          160
```

For the Streamlit demo, the following risk bands are used:

```text
Low Risk    : score < 0.70
Medium Risk : 0.70 ≤ score < 0.98
High Risk   : score ≥ 0.98
```

The high-risk threshold was intentionally set high to reduce false positives and create more credible AML alerts for analyst review.

---

## 🔍 Explainability

SHAP is used to explain individual transaction predictions.

For each selected transaction, the app shows:

- top contributing features,
- whether each feature increases or decreases risk,
- SHAP impact value,
- analyst-friendly explanation.

Example:

```text
Payment format: ACH decreases risk.
Large received amount pattern decreases risk.
Amount received decreases risk.
```

The Streamlit app also visualizes the top SHAP contributors using a horizontal bar chart.

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd aml-risk-report-generator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If using macOS and XGBoost fails because of OpenMP, run:

```bash
brew install libomp
pip uninstall xgboost -y
pip install xgboost
```

### 4. Train the model

```bash
python train_model_v2.py --data_path data/HI-Small_Trans.csv
```

For a faster test run:

```bash
python train_model_v2.py --data_path data/HI-Small_Trans.csv --sample_size 50000
```

### 5. Test SHAP explanations

```bash
python test_explainability.py
```

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📚 Requirements

A minimal requirements file:

```text
pandas
numpy
scikit-learn
xgboost
joblib
shap
matplotlib
streamlit
```

---

## ⭐ Key Technical Highlights

This project demonstrates:

- end-to-end ML pipeline development,
- modular Python code structure,
- XGBoost classification,
- feature engineering for financial transaction monitoring,
- handling extreme class imbalance,
- precision-recall based evaluation,
- threshold tuning,
- SHAP explainability,
- Streamlit deployment,
- analyst-style report generation.

---

## 💼 Portfolio Talking Point

A concise way to describe the project:

> I built an explainable AML risk scoring demo using XGBoost, SHAP and Streamlit. The model predicts suspicious transaction risk, uses AML-inspired engineered features, handles extreme class imbalance, and explains each prediction using SHAP. I also added threshold-based risk levels and an analyst-style downloadable report. The next version extends the system with RAG-based AML guidance retrieval and LLM-generated reports.

---

## ⚠️ Disclaimer

This project is for educational and portfolio demonstration purposes only. It is not intended for real-world AML decision-making, regulatory compliance, or production transaction monitoring.
