# ChurnSight — AI-Powered Customer Churn Prediction & Retention Agent

> An end-to-end agentic AI system that predicts which telecom customers are likely to churn, explains *why*, and autonomously generates personalised retention strategies — all in an interactive Streamlit app.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnsight.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Architecture & Pipeline](#architecture--pipeline)
5. [The AI Retention Agent](#the-ai-retention-agent)
6. [Models](#models)
7. [Dataset](#dataset)
8. [Installation](#installation)
9. [Environment Setup](#environment-setup)
10. [Usage](#usage)
11. [App Pages](#app-pages)
12. [Configuration & Customisation](#configuration--customisation)
13. [Dependencies](#dependencies)

---

## Project Overview

ChurnSight is a full-stack AI analytics system built on the IBM Telco Customer Churn dataset. It goes beyond a standard ML project by layering an **Agentic AI retention engine** on top of the prediction model:

1. **Predict** — Classify customers by churn probability using trained ML models.
2. **Explain** — Surface the key features driving churn for each customer.
3. **Act** — A LangGraph-powered AI agent reasons over the customer profile, retrieves best practices from a knowledge base (RAG), and generates a structured, personalised retention strategy.

---

## Features

| Feature | Details |
|---|---|
| 📊 **Dataset Overview** | Summary KPIs, churn distribution donut chart, monthly-charge histogram by churn status |
| 🤖 **Model Training** | One-click training of Logistic Regression and Decision Tree classifiers |
| 📈 **Evaluation** | Side-by-side ROC curves, confusion matrices, and a full performance table (Accuracy, Precision, Recall, F1, ROC-AUC) |
| 🔮 **Batch Prediction** | Run predictions on the built-in dataset or any uploaded CSV |
| 🎯 **Risk Levels** | Automatic bucketing into 🟢 Low / 🟡 Medium / 🟠 High / 🔴 Critical |
| 🔍 **Feature Importance** | Interactive horizontal bar chart for any trained model with a full sortable table |
| 🧠 **AI Retention Agent** | Agentic pipeline (Data Validation → RAG Retrieval → LLM Generation) that outputs a structured retention report per customer |
| 💾 **Export** | Download predictions as CSV and charts as PNG |
| ⚡ **Persistent Artifacts** | Trained models saved to `models/churn_models.pkl` and reloaded automatically on next run |

---

## Project Structure

```
ChurnSight/
├── app.py                                  # Streamlit UI — all 6 pages
├── requirements.txt                        # Python dependencies
├── .env                                    # Local secrets (GROQ_API_KEY)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco dataset
├── models/
│   └── churn_models.pkl                   # Saved model artifacts (after training)
└── src/
    ├── __init__.py
    ├── agent.py                            # LangGraph-style AI Retention Agent
    ├── data_preprocessing.py              # Load, clean, encode, scale, split
    ├── feature_engineering.py            # Feature importance & risk labelling
    ├── model.py                           # Train, evaluate, save, load models
    ├── retention_kb.txt                   # Knowledge base for RAG retrieval
    └── train.py                           # Standalone CLI training script
```

---

## Architecture & Pipeline

```
Raw CSV
   │
   ▼
data_preprocessing.py
   │  ├─ Drop customerID
   │  ├─ Fix TotalCharges (coerce → fill median)
   │  ├─ Label-encode binary columns (Yes/No)
   │  ├─ One-hot encode multi-category columns
   │  └─ StandardScaler on numeric columns
   │
   ▼
model.py  ──  train_models()
   │  ├─ Logistic Regression  (max_iter=1000)
   │  └─ Decision Tree        (max_depth=8)
   │
   ▼
evaluate_all_models()  →  Accuracy / Precision / Recall / F1 / ROC-AUC
   │
   ▼
save_artifacts()  →  models/churn_models.pkl  (joblib)
   │
   ▼
Streamlit app  →  Predict / Explain / Export
   │
   ▼  (High/Critical risk customers)
AI Retention Agent  →  Structured retention report
```

At **inference time**, uploaded CSVs are preprocessed with the *already-fitted* scaler and label encoders (stored inside the pkl), so the feature space always matches training.

---

## The AI Retention Agent

The agent lives in `src/agent.py` and runs a three-step reasoning pipeline for any at-risk customer:

```
Customer Profile + Churn Probability
          │
          ▼
  1. Data Validator Node
     └─ Flags missing / noisy fields (MonthlyCharges, tenure, TotalCharges)
          │
          ▼
  2. RAG Retriever Node
     └─ Embeds a query built from top churn factors
     └─ Retrieves 2 best-matching chunks from retention_kb.txt (FAISS + HuggingFace embeddings)
          │
          ▼
  3. Generator Node
     └─ Sends customer data + retrieved context to Llama-3.3-70B (via Groq API)
     └─ Produces a structured RetentionReport (Pydantic)
          │
          ▼
  RetentionReport
  ├── risk_summary
  ├── contributing_factors  []
  ├── recommended_actions   [{action, reasoning}, ...]
  ├── supporting_sources    []
  └── business_ethical_disclaimers
```

### Key Components

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph-style sequential state graph |
| LLM | `llama-3.3-70b-versatile` via **Groq API** |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace / sentence-transformers) |
| Vector store | FAISS (in-memory, lazy-loaded) |
| Knowledge base | `src/retention_kb.txt` — curated telecom retention best practices |
| Output schema | Pydantic `RetentionReport` model |

---

## Models

| Model | Key Hyperparameters |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `random_state=42` |
| **Decision Tree** | `max_depth=8`, `random_state=42` |

The best model (highest ROC-AUC) is automatically highlighted in the UI and pre-selected in the Predict, Feature Importance, and AI Retention Agent pages.

---

## Dataset

**IBM Telco Customer Churn**  
[Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 customers |
| Columns | 21 (20 features + 1 target) |
| Target | `Churn` — Yes / No |

### Key Features Used

| Column | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has been with the company |
| `MonthlyCharges` | Numeric | Current monthly bill ($) |
| `TotalCharges` | Numeric | Cumulative spend ($) |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `PaymentMethod` | Categorical | Electronic check / Mail check / Bank transfer / Credit card |
| `OnlineSecurity` | Binary | Yes / No |
| `TechSupport` | Binary | Yes / No |
| `Partner` | Binary | Yes / No |
| `Dependents` | Binary | Yes / No |

---

## Installation

### Prerequisites

- Python **3.9 – 3.12**
- `pip`
- A free **Groq API key** → [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/ChurnSight.git
cd ChurnSight

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Setup

The AI Retention Agent requires a **Groq API key**.

### Local Development

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Streamlit Cloud Deployment

Add the key under **App Settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

The app automatically reads from `st.secrets` when deployed to Streamlit Cloud.

---

## Usage

### Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Training via CLI

To train without launching the UI:

```bash
python src/train.py
```

This runs the full pipeline — load → preprocess → train → evaluate → save — and prints a formatted results table to stdout. The artifact is saved to `models/churn_models.pkl`.

---

## App Pages

### 1. Overview
High-level KPIs: total customers, churned count, retention count, and churn rate. Includes a churn distribution donut chart and a monthly-charges histogram overlaid by churn status. Also shows the full feature reference table.

### 2. Train & Evaluate
- Click **Train All Models** to run the full training pipeline.
- View a performance summary table (Accuracy, Precision, Recall, F1, ROC-AUC).
- Explore ROC curves (one line per model) and confusion matrices side by side.
- The best model by ROC-AUC is highlighted automatically.

### 3. Predict
- Select any trained model (best model pre-selected).
- Use the **bundled Telco dataset** or upload your own CSV.
- See aggregate KPIs, a risk-distribution donut, and a churn probability histogram.
- Filter the customer risk table by risk tier and download results as CSV.

### 4. Feature Importance
- Choose a model to inspect.
- Adjust the slider to show the top N features (5–30).
- View an interactive horizontal bar chart and a full sortable table.

### 5. AI Retention Agent *(New)*
- Automatically scores all customers and filters for **High** and **Critical** risk ones.
- Select any at-risk customer from the dropdown.
- Click **Generate Strategy** — the agent validates the customer's data, retrieves relevant retention tactics from the knowledge base, and calls the LLM to produce a full `RetentionReport` including:
  - Risk summary
  - Contributing factors
  - Recommended actions (with reasoning)
  - Supporting sources from the KB
  - Business & ethical disclaimers

### 6. About
Full data dictionary, model output reference, and tech-stack summary.

---

## Configuration & Customisation

| What | Where | How |
|---|---|---|
| Add / remove ML models | `src/model.py` → `train_models()` | Add a new key–value pair; evaluation and UI pick it up automatically |
| Change train/test split | `src/data_preprocessing.py` → `split_data()` | Adjust `test_size` (default `0.2`) |
| Change risk thresholds | `src/feature_engineering.py` → `get_churn_risk_level()` | Edit the probability cut-offs |
| Update the knowledge base | `src/retention_kb.txt` | Add or edit retention best practices; FAISS index rebuilds on next run |
| Change the LLM | `src/agent.py` → `generator_node()` | Swap `model="llama-3.3-70b-versatile"` for any Groq-supported model |
| Change dataset path | `app.py` and `src/train.py` | Update the `DATASET` constant / `load_data()` call |
| Retrain after code changes | UI or CLI | Delete `models/churn_models.pkl` and re-train |

---

## Dependencies

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | ML models, preprocessing, metrics |
| `plotly` | Interactive charts |
| `joblib` | Model serialisation |
| `langgraph`, `langchain-core` | Agentic AI workflow orchestration |
| `langchain-groq` | Groq LLM integration |
| `langchain-community` | FAISS vector store, document loaders |
| `faiss-cpu` | In-memory vector similarity search |
| `sentence-transformers` | HuggingFace embeddings (`all-MiniLM-L6-v2`) |
| `pydantic` | Structured output schema for agent reports |
| `python-dotenv` | Environment variable management |
| `kaleido` | PNG export of Plotly charts |
| `imbalanced-learn` | Available for SMOTE oversampling |
| `matplotlib`, `seaborn` | Available for static plots |
