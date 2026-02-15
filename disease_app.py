import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Disease Detection", page_icon="🩸", layout="wide")

# ── Load Model Bundle ────────────────────────────────────────────────────────
MODEL_PATH = "blood_model.pkl"


@st.cache_resource
def load_bundle():
    """Load the model bundle (model + label_encoder + feature_ranges) from disk.

    Supports two formats:
    - New bundle dict with keys: model, label_encoder, feature_names, feature_ranges
    - Legacy: raw sklearn model (RandomizedSearchCV / Pipeline)
    """
    try:
        obj = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file **{MODEL_PATH}** not found. Make sure it is in the same directory as this app.")
        st.stop()

    if isinstance(obj, dict) and "model" in obj:
        return obj  # new bundle format

    # Legacy format – wrap into a bundle so the rest of the app stays uniform
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(["Anemia", "Diabetes", "Healthy",
                            "Heart Disease", "Thalassemia", "Thrombocytosis"])
    return {
        "model": obj,
        "label_encoder": le,
        "feature_names": [
            "Glucose", "Cholesterol", "Hemoglobin", "Platelets",
            "White Blood Cells", "Red Blood Cells", "Hematocrit",
            "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
            "Mean Corpuscular Hemoglobin Concentration", "Insulin", "BMI",
            "Systolic Blood Pressure", "Diastolic Blood Pressure",
            "Triglycerides", "HbA1c", "LDL Cholesterol", "HDL Cholesterol",
            "ALT", "AST", "Heart Rate", "Creatinine", "Troponin",
            "C-reactive Protein",
        ],
        "feature_ranges": {
            "Glucose": (50.0, 300.0), "Cholesterol": (100.0, 400.0),
            "Hemoglobin": (5.0, 20.0), "Platelets": (50000.0, 500000.0),
            "White Blood Cells": (2000.0, 20000.0), "Red Blood Cells": (2.0, 8.0),
            "Hematocrit": (20.0, 60.0), "Mean Corpuscular Volume": (60.0, 110.0),
            "Mean Corpuscular Hemoglobin": (20.0, 40.0),
            "Mean Corpuscular Hemoglobin Concentration": (28.0, 38.0),
            "Insulin": (2.0, 300.0), "BMI": (10.0, 50.0),
            "Systolic Blood Pressure": (80.0, 200.0),
            "Diastolic Blood Pressure": (40.0, 130.0),
            "Triglycerides": (30.0, 500.0), "HbA1c": (3.0, 15.0),
            "LDL Cholesterol": (30.0, 250.0), "HDL Cholesterol": (15.0, 100.0),
            "ALT": (5.0, 200.0), "AST": (5.0, 200.0),
            "Heart Rate": (40.0, 150.0), "Creatinine": (0.3, 5.0),
            "Troponin": (0.0, 2.0), "C-reactive Protein": (0.0, 50.0),
        },
    }


bundle = load_bundle()
model = bundle["model"]
label_encoder = bundle["label_encoder"]
FEATURE_NAMES = bundle["feature_names"]

# feature_ranges from bundle are (min, max); add sensible defaults for sliders
_saved_ranges = bundle["feature_ranges"]
_DEFAULTS = {
    "Glucose": 100.0, "Cholesterol": 200.0, "Hemoglobin": 14.0,
    "Platelets": 250000.0, "White Blood Cells": 7000.0, "Red Blood Cells": 5.0,
    "Hematocrit": 42.0, "Mean Corpuscular Volume": 85.0,
    "Mean Corpuscular Hemoglobin": 29.0,
    "Mean Corpuscular Hemoglobin Concentration": 33.0,
    "Insulin": 15.0, "BMI": 25.0, "Systolic Blood Pressure": 120.0,
    "Diastolic Blood Pressure": 80.0, "Triglycerides": 150.0, "HbA1c": 5.5,
    "LDL Cholesterol": 100.0, "HDL Cholesterol": 55.0, "ALT": 30.0,
    "AST": 30.0, "Heart Rate": 75.0, "Creatinine": 1.0, "Troponin": 0.02,
    "C-reactive Protein": 3.0,
}
FEATURE_RANGES = {
    feat: (lo, hi, _DEFAULTS.get(feat, (lo + hi) / 2))
    for feat, (lo, hi) in _saved_ranges.items()
}

# Build disease label mapping from saved LabelEncoder
DISEASE_LABELS = {i: name for i, name in enumerate(label_encoder.classes_)}

DISEASE_COLORS = {
    "Anemia": "#e74c3c",
    "Diabetes": "#f39c12",
    "Healthy": "#2ecc71",
    "Heart Disease": "#9b59b6",
    "Thalassemia": "#3498db",
    "Thrombocytosis": "#e67e22",
}

DISEASE_DESCRIPTIONS = {
    "Anemia": "A condition where you lack enough healthy red blood cells to carry adequate oxygen to your tissues.",
    "Diabetes": "A group of metabolic diseases characterized by high blood sugar levels over a prolonged period.",
    "Healthy": "No significant disease detected based on the provided blood sample values.",
    "Heart Disease": "A range of conditions that affect the heart, including coronary artery disease and heart failure.",
    "Thalassemia": "An inherited blood disorder causing the body to have less hemoglobin than normal.",
    "Thrombocytosis": "A condition of high platelet count in the blood, which can cause clotting issues.",
}


def normalize_input(values: dict) -> np.ndarray:
    """Normalize raw feature values to [0, 1] using saved feature_ranges."""
    row = []
    for feat in FEATURE_NAMES:
        lo, hi = _saved_ranges[feat]
        row.append((values[feat] - lo) / (hi - lo))
    return np.array([row])


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🩸 Blood Sample Disease Detection")
st.markdown(
    "Enter your blood test values below and click **Predict** to detect potential diseases. "
    "The model was trained on a blood samples dataset using a Stacking Classifier pipeline."
)

# ── Input ─────────────────────────────────────────────────────────────────────
st.subheader("✏️ Enter Blood Parameters")
col1, col2, col3 = st.columns(3)
columns = [col1, col2, col3]

features = {}
for i, feat in enumerate(FEATURE_NAMES):
    _, _, default = FEATURE_RANGES[feat]
    with columns[i % 3]:
        features[feat] = st.number_input(feat, value=default, format="%.4f")


# ── Prediction ────────────────────────────────────────────────────────────────
st.divider()
if st.button("🔍 Predict Disease", type="primary", use_container_width=True):
        input_array = normalize_input(features)
        prediction = model.predict(input_array)[0]
        disease = label_encoder.inverse_transform([prediction])[0]
        color = DISEASE_COLORS.get(disease, "#555")

        st.divider()

        # Result Card
        col_result, col_info = st.columns([1, 2])

        with col_result:
            st.markdown(
                f"""
                <div style="background:{color}22; border-left: 5px solid {color};
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color:{color}; margin:0;">🏥 {disease}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_info:
            st.info(DISEASE_DESCRIPTIONS.get(disease, ""))

        # Show input summary
        with st.expander("📋 View Input Summary"):
            summary_df = pd.DataFrame(
                {"Feature": FEATURE_NAMES, "Value": [features[f] for f in FEATURE_NAMES]}
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ This tool is for educational purposes only and should not replace professional medical advice.")
