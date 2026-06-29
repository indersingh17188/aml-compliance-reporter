<div align="center">

![Explainable AML Risk Report Generator](screenshots/hero_banner.png)

# 🏦 Explainable AML Compliance Reporter

**Explainable AML risk scoring with XGBoost, SHAP, FAISS RAG, and Streamlit**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)]()
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)]()
[![FAISS](https://img.shields.io/badge/RAG-FAISS-purple.svg)]()
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed-lightgrey.svg)]()
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)]()
[![Status](https://img.shields.io/badge/Status-v2%20RAG%20Working-success.svg)]()

</div>

---

## 🚀 Overview

This project is an end-to-end AI engineering demo for **Anti-Money Laundering (AML) transaction risk assessment**.

It combines:

- 🤖 **XGBoost** for suspicious transaction risk prediction
- 🛠️ **AML-inspired feature engineering**
- 🔍 **SHAP** for explainable model decisions
- 📚 **FAISS-based RAG** over a curated AML knowledge base
- 📄 **RAG-enriched analyst-style reports**
- 🖥️ **Streamlit** for an interactive dashboard
- 🧪 reusable positive/negative testing samples for demo and validation

> ⚠️ This is a synthetic-data portfolio project and not a production AML compliance system.

---

## 📸 Demo Preview

### 🖥️ Streamlit Dashboard

![Dashboard Preview](screenshots/dashboard_preview.png)

### 🔍 SHAP Explainability

![SHAP Explanation](screenshots/shap_explanation.png)

### ⚙️ System Architecture

![System Architecture](screenshots/architecture.png)

---

## ✨ What the App Does

A user selects a transaction sample and clicks **Predict AML Risk**.

The app then:

1. 🧾 Loads the selected transaction
2. 🤖 Predicts AML risk using XGBoost
3. 🎯 Converts probability into Low / Medium / High risk
4. 🔍 Explains the prediction using SHAP
5. 📚 Builds an automatic RAG query from SHAP factors and transaction attributes
6. 🔎 Retrieves relevant AML guidance from a local knowledge base using FAISS
7. 📄 Generates a RAG-enriched analyst report
8. ⬇️ Allows report download as text

---

## 🧠 Architecture

```text
Selected Transaction
        ↓
Feature Engineering
        ↓
XGBoost Risk Model
        ↓
Risk Score + Thresholds
        ↓
SHAP Explanation
        ↓
Automatic Retrieval Query
        ↓
FAISS RAG over AML Knowledge Base
        ↓
RAG-Enriched Analyst Report
        ↓
Streamlit Dashboard
```

---

## 📁 Project Structure

```text
aml-compliance-reporter/
│
├── app.py
├── train_model.py
├── train_model_v2.py
├── create_testing_samples.py
├── test_explainability.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── utils/
│   ├── __init__.py
│   ├── feature_engineering.py
│   └── explainability.py
│
├── rag/
│   ├── __init__.py
│   ├── knowledge_loader.py
│   ├── chunker.py
│   ├── embedding_model.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── report_generator.py
│   └── test_retriever.py
│
├── knowledge_base/
│   ├── 01_structuring.md
│   ├── 02_layering.md
│   ├── 03_integration.md
│   ├── ...
│   └── 20_reporting_guidelines.md
│
├── data/
│   ├── .gitkeep
│   ├── HI-Small_Trans.csv                         # ignored
│   ├── sample_app_inputs_v2.csv                   # ignored
│   └── testing_samples/                           # ignored
│
├── models/
│   ├── .gitkeep
│   ├── xgb_aml_pipeline_v2.pkl                    # ignored
│   └── feature_info_v2.pkl                        # ignored
│
├── rag_index/
│   ├── aml_knowledge.faiss                        # ignored
│   └── chunks_metadata.joblib                     # ignored
│
└── screenshots/
    ├── hero_banner.png
    ├── dashboard_preview.png
    ├── shap_explanation.png
    ├── report_preview.png
    └── architecture.png
```

---

## 💾 Dataset

This project uses a synthetic AML transaction dataset based on IBM AML-Data.

The dataset contains transaction-level fields such as:

- timestamp
- originating bank
- receiving bank
- account identifiers
- amount paid
- amount received
- payment currency
- receiving currency
- payment format
- laundering label

The supervised target column is:

```text
Is_Laundering
```

Account identifiers and raw timestamps are excluded from the final model feature set to reduce memorization and leakage.

---

## 🛠️ Feature Engineering

The improved model uses AML-inspired engineered features.

### ⏰ Time Features

- transaction hour
- transaction day
- transaction month
- day of week
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
- same-bank transfer flag
- cross-bank transfer flag

---

## 📊 Model Performance

The first baseline model achieved high ROC-AUC but weak PR-AUC because the dataset is extremely imbalanced.

| Metric | Baseline Model | Feature-Engineered Model |
|---|---:|---:|
| ROC-AUC | 0.9664 | **0.9712** |
| PR-AUC | 0.0480 | **0.2273** |

The engineered model improved PR-AUC from:

```text
0.0480 → 0.2273
```

This is approximately a **4.7× improvement**, which is more meaningful than ROC-AUC for highly imbalanced AML data.

---

## 🎯 Threshold Strategy

The default `0.50` threshold produced too many false positives. Therefore, the model was evaluated across multiple thresholds.

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

The Streamlit app uses:

```text
Low Risk    : score < 0.70
Medium Risk : 0.70 ≤ score < 0.98
High Risk   : score ≥ 0.98
```

The high-risk threshold is intentionally strict to produce fewer but more credible AML alerts.

---

## 🔍 Explainability with SHAP

SHAP is used to explain individual predictions.

For each selected transaction, the app displays:

- top contributing features
- SHAP impact values
- whether each feature increases or decreases risk
- human-readable explanations

Example:

```text
Cross-currency transaction increases risk.
Night-time transaction increases risk.
Large paid amount pattern increases risk.
```

This makes the prediction easier to interpret for analysts and non-technical stakeholders.

---

## 📚 RAG Knowledge Base

The RAG layer uses a curated local AML knowledge base with 20 markdown documents.

Topics include:

- structuring and smurfing
- layering
- integration
- cross-border transfers
- shell companies
- high-risk jurisdictions
- sanctions screening
- PEP transactions
- cash-intensive businesses
- round-tripping
- trade-based money laundering
- unusual transaction timing
- currency conversion patterns
- large-value transactions
- multiple accounts
- beneficial ownership
- payment methods
- enhanced due diligence
- AML red flags
- analyst reporting guidelines

Each document contains:

- overview
- why it matters
- typical indicators
- analyst considerations
- example scenario
- suggested action
- related topics

---

## 🔎 FAISS RAG Pipeline

The RAG module follows this flow:

```text
AML markdown documents
        ↓
Document chunking
        ↓
FastEmbed sentence embeddings
        ↓
FAISS vector index
        ↓
Top-k similarity retrieval
        ↓
RAG-enriched analyst report
```

The app automatically builds a retrieval query using:

- risk level
- SHAP explanation reasons
- transaction attributes such as cross-currency, cross-bank, night transaction, and amount difference

This means the user does not need to manually ask a question. The system retrieves relevant AML guidance automatically.

---

## 🧪 Testing Samples

The project includes a utility script to generate reusable testing samples from the labelled dataset.

```bash
python create_testing_samples.py
```

This creates local testing files such as:

```text
data/testing_samples/
├── positive_high_risk_with_metadata.csv
├── positive_random_with_metadata.csv
├── negative_low_risk_with_metadata.csv
├── negative_high_score_false_positive_candidates_with_metadata.csv
├── negative_random_with_metadata.csv
└── mixed_testing_samples_with_metadata.csv
```

The Streamlit sidebar can use these groups to demo:

- labelled laundering examples
- normal low-risk examples
- false-positive candidates
- mixed testing samples

These files are ignored by Git because they are generated from local data.

---

## ▶️ How to Run Locally

### 1. Create conda environment

```bash
conda create -n aml python=3.11 -y
conda activate aml
```

### 2. Install dependencies

Recommended:

```bash
pip install -r requirements.txt
```

If using conda for FAISS:

```bash
conda install -c conda-forge faiss-cpu -y
pip install -r requirements.txt
```

### 3. Train the XGBoost model

```bash
python train_model_v2.py --data_path data/HI-Small_Trans.csv
```

For a quick test:

```bash
python train_model_v2.py --data_path data/HI-Small_Trans.csv --sample_size 50000
```

### 4. Build the RAG index

```bash
python -m rag.build_index
```

This creates:

```text
rag_index/
├── aml_knowledge.faiss
└── chunks_metadata.joblib
```

### 5. Test RAG retrieval

```bash
python -m rag.test_retriever
```

### 6. Create testing samples

```bash
python create_testing_samples.py
```

### 7. Run Streamlit

```bash
streamlit run app.py
```

---

## 📚 Requirements

Core dependencies:

```text
numpy
pandas
scikit-learn
xgboost
shap
matplotlib
streamlit
joblib
faiss-cpu
fastembed
```

---

## 🧭 Version Roadmap

| Version | Status | Description |
|---|---|---|
| v1.0 | ✅ Done | XGBoost + SHAP + Streamlit |
| v2.0 | ✅ Done | FAISS RAG + AML knowledge base + RAG report |
| v2.1 | 🔜 Next | Improved UI and report formatting |
| v3.0 | 🔜 Planned | Lightweight LLM report rewriting |
| v4.0 | 🔜 Planned | Docker and Hugging Face Spaces deployment |

---

## 💼 Portfolio Talking Point

> I built an explainable AML compliance reporter using XGBoost, SHAP, FAISS RAG, and Streamlit. The system predicts suspicious transaction risk, explains the prediction using SHAP, retrieves relevant AML guidance from a curated local knowledge base, and generates an analyst-style report. I also created reusable positive and negative test samples to demonstrate model behaviour under different scenarios.

---

## 🏷️ Suggested GitHub Topics

```text
xgboost
shap
faiss
fastembed
streamlit
machine-learning
financial-ai
anti-money-laundering
aml
explainable-ai
xai
python
rag
llm
```

---

## ⚠️ Disclaimer

This project is for educational and portfolio demonstration purposes only. It uses synthetic data and is not intended for real-world AML decision-making, regulatory compliance, or production transaction monitoring.
