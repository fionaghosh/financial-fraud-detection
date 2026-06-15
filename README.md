# 🛡️ Financial Transaction Fraud Detection

Real-time, **explainable** fraud risk scoring on 6.3M mobile-money transactions — built with a four-model comparison, SHAP explanations, cost-aware decision thresholds, and a deployed interactive dashboard.

**[🔗 Live Dashboard](https://YOUR-APP-URL.streamlit.app)** · Dataset: [PaySim (Kaggle)](https://www.kaggle.com/datasets/ealaxi/paysim1)

---

## Overview

Financial institutions lose billions to fraud while drowning in false alarms from rigid rule-based systems. This project builds a machine-learning fraud detector that is **accurate, auditable, and cost-aware** — the three things a risk or audit team actually needs.

The headline result: on the held-out test set, the platform's built-in **rule-based system caught only 32% of fraud** (and was wrong on ~99% of its alerts), while the **ML model caught 99.7% of fraud at 99.7% precision** — a clear, quantified business case for analytics-led fraud detection.

> **Note on the data:** PaySim is a *synthetic* benchmark where fraud follows deterministic rules, so the engineered balance-error features make fraud nearly separable. The focus of this project is therefore the surrounding rigour — model comparison, explainability, cost-aware thresholding, calibration, and temporal validation — rather than the headline accuracy.

---

## Key Results

| Approach | Frauds caught | Recall | Precision | F1 |
|---|---|---|---|---|
| Rule-based (amount > 200k) | 666 / 2,053 | 32.4% | 0.7% | 0.013 |
| **ML — XGBoost** | **2,046 / 2,053** | **99.7%** | **99.7%** | **0.997** |

**Model comparison (PR-AUC, the right metric for 0.3% fraud):**

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **XGBoost** | **0.9976** | 0.9989 |
| Random Forest | 0.9974 | 0.9988 |
| LightGBM | 0.9974 | 0.9990 |
| Logistic Regression (baseline) | 0.6496 | 0.9911 |

---

## What's in this project

**1. Insight-driven preprocessing.** Fraud in PaySim occurs *only* in TRANSFER and CASH_OUT transactions, so the dataset is filtered to where fraud actually lives (6.3M → 2.8M rows), focusing the model and removing noise.

**2. Feature engineering.** Balance-error features capture accounting inconsistencies left by fraudulent transfers (`errorBalanceOrig = newBalance + amount − oldBalance`). SHAP later confirmed `errorBalanceOrig` as by far the strongest predictor — a feature the model learns to trust the same way a human investigator would.

**3. Four-model comparison.** Logistic Regression (interpretable baseline) → Random Forest → XGBoost → LightGBM, all with class-imbalance handling, evaluated on **PR-AUC** rather than misleading overall accuracy. *(LightGBM initially failed under an aggressive `scale_pos_weight`; switching to `is_unbalance` with leaf constraints restored it — a documented debugging story.)*

**4. Explainability (SHAP).** `TreeExplainer` produces global feature importance plus per-transaction explanations, so every flag is **auditable and justifiable** — essential for regulatory and audit contexts.

**5. Cost-aware threshold tuning.** Instead of the default 0.50 cutoff, the decision threshold is tuned to minimise expected cost, weighting a missed fraud far more heavily than a false alarm — framing the model around real operational trade-offs.

**6. Calibration.** Reliability curve and Brier score confirm the model outputs trustworthy probabilities (Brier 0.00002), not just rankings.

**7. Temporal validation.** Re-trained on earlier transactions and tested on later ones ("train the past, predict the future") to check for time leakage — performance held, confirming the model generalises across time.

**8. Deployed dashboard.** An interactive Streamlit app where anyone can score a transaction (from examples or manual entry) and see the risk score *plus* the SHAP feature contributions explaining it.

---

## Tech stack

`Python` · `pandas` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Streamlit` · `matplotlib`

---

## Running locally

```bash
# 1. Download the dataset from Kaggle (ealaxi/paysim1) into this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. (If retraining) run the pipeline scripts to regenerate the .pkl artifacts
# 4. Launch the app
streamlit run streamlit_app.py
```

The app loads three artifacts: `best_model.pkl`, `feature_names.pkl`, and `decision_config.pkl`.

---

## Repository structure

```
streamlit_app.py        # the deployed dashboard
requirements.txt        # pinned dependencies
best_model.pkl          # trained XGBoost model
feature_names.pkl       # ordered feature list
decision_config.pkl     # cost-optimal decision threshold
README.md
```

---

## Author

**Fiona Ghosh** — Master of Data Science, Deakin University
[Portfolio](#) · [LinkedIn](https://linkedin.com/in/fiona-ghosh) · [GitHub](https://github.com/fionaghosh)
