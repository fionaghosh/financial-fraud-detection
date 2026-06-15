"""
Financial Transaction Fraud Detection — Streamlit App
-----------------------------------------------------
Loads the trained XGBoost model and serves real-time fraud risk scoring
with SHAP explanations and a cost-aware decision threshold.

Run locally:   streamlit run streamlit_app.py
Artifacts required in the same folder:
    best_model.pkl        (trained XGBoost)
    feature_names.pkl     (ordered feature list)
    decision_config.pkl   (cost-optimal threshold)   [optional; defaults to 0.50]
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection — Risk Scorer",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------------
# LOAD ARTIFACTS (cached so they load once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    try:
        cfg = joblib.load("decision_config.pkl")
        threshold = float(cfg["threshold"])
    except Exception:
        threshold = 0.50
    explainer = shap.TreeExplainer(model)
    return model, feature_names, threshold, explainer

model, FEATURES, THRESHOLD, EXPLAINER = load_artifacts()

# ----------------------------------------------------------------------
# SAMPLE TRANSACTIONS (real rows pulled from PaySim)
# ----------------------------------------------------------------------
SAMPLES = {
    "— Choose an example —": None,
    "🚨 Known fraud (account drained)": {
        "step": 1, "amount": 181.0, "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "is_transfer": True,
    },
    "✅ Normal transfer": {
        "step": 1, "amount": 62610.80, "oldbalanceOrg": 79114.0, "newbalanceOrig": 16503.20,
        "oldbalanceDest": 517.0, "newbalanceDest": 8383.29, "is_transfer": True,
    },
}

# ----------------------------------------------------------------------
# FEATURE BUILDER — recomputes engineered features from raw inputs
# so the model always sees consistent values.
# ----------------------------------------------------------------------
def build_features(step, amount, oldOrg, newOrig, oldDest, newDest, is_transfer):
    errorBalanceOrig = newOrig + amount - oldOrg
    errorBalanceDest = oldDest + amount - newDest
    origDrainedToZero = int(oldOrg > 0 and newOrig == 0)
    destStartedEmpty = int(oldDest == 0)
    type_TRANSFER = int(is_transfer)

    row = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldOrg,
        "newbalanceOrig": newOrig,
        "oldbalanceDest": oldDest,
        "newbalanceDest": newDest,
        "errorBalanceOrig": errorBalanceOrig,
        "errorBalanceDest": errorBalanceDest,
        "origDrainedToZero": origDrainedToZero,
        "destStartedEmpty": destStartedEmpty,
        "type_TRANSFER": type_TRANSFER,
    }
    # Order columns exactly as the model expects
    return pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)

# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------
st.title("🛡️ Financial Transaction Fraud Detection")
st.markdown(
    "Real-time fraud risk scoring with **explainable** predictions. "
    "Built on the PaySim mobile-money dataset (6.3M transactions). "
    "The model flags transactions and shows *why* — supporting auditable, "
    "cost-aware financial-crime review."
)
st.caption(
    f"Decision threshold: **{THRESHOLD:.2f}** "
    "(tuned so the cost of a missed fraud outweighs a false alarm). "
    "Note: PaySim is synthetic data, so separability is unusually high — "
    "the focus here is methodology and explainability, not the headline score."
)
st.divider()

# ----------------------------------------------------------------------
# LAYOUT: inputs on the left, results on the right
# ----------------------------------------------------------------------
left, right = st.columns([1, 1.1])

with left:
    st.subheader("Transaction details")

    choice = st.selectbox("Start from an example (optional):", list(SAMPLES.keys()))
    preset = SAMPLES[choice] if SAMPLES[choice] else {}

    txn_type = st.radio(
        "Transaction type", ["TRANSFER", "CASH_OUT"],
        index=0 if preset.get("is_transfer", True) else 1,
        horizontal=True,
    )
    is_transfer = (txn_type == "TRANSFER")

    amount = st.number_input("Amount", min_value=0.0,
                             value=float(preset.get("amount", 1000.0)), step=100.0)
    c1, c2 = st.columns(2)
    with c1:
        oldOrg = st.number_input("Sender balance (before)", min_value=0.0,
                                 value=float(preset.get("oldbalanceOrg", 5000.0)), step=100.0)
        oldDest = st.number_input("Receiver balance (before)", min_value=0.0,
                                  value=float(preset.get("oldbalanceDest", 0.0)), step=100.0)
    with c2:
        newOrig = st.number_input("Sender balance (after)", min_value=0.0,
                                  value=float(preset.get("newbalanceOrig", 4000.0)), step=100.0)
        newDest = st.number_input("Receiver balance (after)", min_value=0.0,
                                  value=float(preset.get("newbalanceDest", 0.0)), step=100.0)
    step = st.slider("Time step (hour)", 1, 743, int(preset.get("step", 1)))

    run = st.button("🔍 Assess transaction", type="primary", use_container_width=True)

with right:
    st.subheader("Risk assessment")

    if run:
        X = build_features(step, amount, oldOrg, newOrig, oldDest, newDest, is_transfer)
        prob = float(model.predict_proba(X)[:, 1][0])
        is_fraud = prob >= THRESHOLD

        # Risk verdict
        if is_fraud:
            st.error(f"### 🚨 FLAGGED AS FRAUD\nRisk score: **{prob:.1%}**")
        else:
            st.success(f"### ✅ Looks legitimate\nRisk score: **{prob:.1%}**")

        st.progress(min(prob, 1.0))

        # ----- SHAP explanation -----
        st.markdown("#### Why? — feature contributions")
        shap_vals = EXPLAINER.shap_values(X)
        contrib = pd.DataFrame({
            "feature": FEATURES,
            "value": X.iloc[0].values,
            "shap": shap_vals[0],
        })
        contrib["abs"] = contrib["shap"].abs()
        contrib = contrib.sort_values("abs", ascending=False).head(6)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        colors = ["#d62728" if s > 0 else "#2ca02c" for s in contrib["shap"]]
        ax.barh(contrib["feature"][::-1], contrib["shap"][::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("← lowers risk      pushes toward fraud →")
        ax.set_title("Top drivers of this prediction")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Red bars push the transaction toward *fraud*; green bars toward *legitimate*. "
            "`errorBalanceOrig` (balance-arithmetic inconsistency) is typically the strongest signal."
        )
    else:
        st.info("Enter a transaction (or pick an example) and click **Assess transaction**.")

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Model: XGBoost · Validated with a 4-model comparison, SHAP explainability, "
    "cost-aware thresholding, calibration, and a temporal train/test split. "
    "Dataset: PaySim (synthetic). Built by Fiona Ghosh."
)
