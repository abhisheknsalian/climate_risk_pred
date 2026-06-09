import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Flood Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("flood_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🌊 Flood Risk Prediction")

st.sidebar.markdown("""
### Master's Research Project

**Title:**

Investigating the Trade-off Between Predictive Performance and Explainability in AI-Based Flood Probability Prediction

---

### Best Model

✅ Explainable Boosting Machine (EBM)

### Model Performance

- EBM: 0.8458
- Linear Regression: 0.8449
- XGBoost: 0.7610
- Random Forest: 0.6551

---

### Purpose

Predict flood probability using environmental indicators while maintaining transparency and explainability.
""")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("🌊 Flood Risk Prediction System")

st.markdown("""
This application predicts **Flood Probability** using environmental and climate-related indicators.

The prediction is generated using an **Explainable Boosting Machine (EBM)** model, which achieved the highest predictive performance while maintaining transparency.
""")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.header("📋 Environmental Indicators")

input_data = {}

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):

    if i % 2 == 0:
        with col1:
            input_data[feature] = st.slider(
                feature,
                min_value=0,
                max_value=10,
                value=5
            )
    else:
        with col2:
            input_data[feature] = st.slider(
                feature,
                min_value=0,
                max_value=10,
                value=5
            )

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict Flood Probability"):

    input_df = pd.DataFrame([input_data])

    prediction = float(model.predict(input_df)[0])

# Keep prediction in valid probability range
    prediction = max(0.0, min(prediction, 1.0))

    st.header("📈 Prediction Result")

    st.metric(
        label="Flood Probability",
        value=f"{prediction:.2%}"
    )

    st.progress(float(prediction))

    # Risk Classification

    if prediction < 0.33:

        st.success(
            "🟢 LOW RISK\n\nFlood probability is relatively low."
        )

    elif prediction < 0.66:

        st.warning(
            "🟡 MEDIUM RISK\n\nModerate flood probability detected."
        )

    else:

        st.error(
            "🔴 HIGH RISK\n\nHigh flood probability detected."
        )

    st.subheader("Input Summary")

    st.dataframe(
        input_df,
        use_container_width=True
    )

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.header("📊 Model Performance Comparison")

results = pd.DataFrame({
    "Model": [
        "Explainable Boosting Machine",
        "Linear Regression",
        "XGBoost",
        "Random Forest"
    ],
    "Model Type": [
        "Interpretable",
        "Interpretable",
        "Black-Box",
        "Black-Box"
    ],
    "R² Score": [
        0.8458,
        0.8449,
        0.7610,
        0.6551
    ]
})

st.dataframe(
    results,
    use_container_width=True
)

# -----------------------------
# RESEARCH FINDINGS
# -----------------------------
st.header("🔬 Key Research Findings")

st.markdown("""
### Research Question 1
**How do interpretable and black-box machine learning models differ in predictive performance?**

Interpretable models consistently outperformed black-box models.

---

### Research Question 2
**Which environmental indicators most strongly influence flood probability prediction?**

Major contributing factors include:

- Deteriorating Infrastructure
- Monsoon Intensity
- Dams Quality
- Siltation
- River Management
- Topography Drainage
- Climate Change

---

### Research Question 3
**Can interpretable machine learning models achieve performance comparable to or better than complex ensemble models?**

Yes.

The Explainable Boosting Machine achieved the highest predictive performance while remaining fully interpretable.

---

### Research Question 4
**What are the implications of transparency for responsible AI?**

Transparent models improve trust, accountability, and decision-making in disaster-management systems.
""")

# -----------------------------
