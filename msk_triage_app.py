import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 1. LOAD MODEL
# -----------------------------

MODEL_PATH = "msk_triage_rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Feature definitions (must match training)
triage_text_features = ["subjective_injury_description"]

triage_numeric_features = [
    "age",
    "training_hours_per_week",
    "previous_injuries_count",
    "previous_same_site_injury",
    "time_since_onset_days",
    "pain_at_rest",
    "pain_on_activity",
    "range_of_motion_limit",
]

triage_categorical_features = [
    "sex", "sport", "level",
    "injury_site", "injury_type",
    "injury_severity", "onset", "swelling",
    "treatment_modality",
]

TREATMENT_MODALITIES = ["physio", "chiropractor", "osteopath", "sports_massage"]

required_cols = triage_text_features + triage_numeric_features + triage_categorical_features


def recommend_treatment(input_dict, model=model, treatment_modalities=TREATMENT_MODALITIES):
    """
    Simulates different treatment modalities for the same injury presentation
    and ranks them by predicted success probability.
    """
    base = pd.Series(input_dict)

    missing = [c for c in required_cols if c not in base.index]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    simulated_rows = []
    for m in treatment_modalities:
        r = base.copy()
        r["treatment_modality"] = m
        simulated_rows.append(r.to_dict())

    sim_df = pd.DataFrame(simulated_rows)[required_cols]

    probs = model.predict_proba(sim_df)[:, 1]

    df_results = pd.DataFrame({
        "treatment_modality": treatment_modalities,
        "predicted_success_prob": probs
    }).sort_values("predicted_success_prob", ascending=False).reset_index(drop=True)

    return df_results


# -----------------------------
# 2. STREAMLIT PAGE CONFIG & STYLING
# -----------------------------

st.set_page_config(
    page_title="MSK Triage Recommender",
    page_icon="💪",
    layout="centered",
)

# Light custom CSS to make it look less “raw Streamlit”
st.markdown("""
    <style>
        .main { padding-top: 1rem; }
        .reportview-container .markdown-text-container {
            font-family: "Helvetica", sans-serif;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1.2rem;
            margin-bottom: 0.3rem;
            color: #15416e;
        }
        .card {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            border: 1px solid #e0e0e0;
            background-color: #fafafa;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💪 Soft Tissue / MSK Triage Recommender")
st.caption("Prototype tool to explore likely treatment success pathways based on synthetic injury data.")

st.markdown(
    "<div class='card'>"
    "<b>Disclaimer:</b> This is a prototype model trained on synthetic data. It is not a medical device and "
    "must not be used for real clinical decision-making."
    "</div>",
    unsafe_allow_html=True
)

# -----------------------------
# 3. INPUT FORM
# -----------------------------

with st.form("triage_form"):
    st.markdown("<div class='section-title'>1. Subjective injury description</div>", unsafe_allow_html=True)
    subjective_injury_description = st.text_area(
        "",
        value="Felt a sharp pain in the back of my thigh while sprinting. Pain is worse with fast running and high-speed drills.",
        height=90
    )

    st.markdown("<div class='section-title'>2. Patient & training profile</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=16, max_value=80, value=28)
        sex = st.selectbox("Sex", options=["male", "female"])

    with col2:
        sport = st.selectbox("Sport", options=["football", "running", "tennis", "gym", "rugby", "basketball"])
        level = st.selectbox("Sport level", options=["recreational", "semi_pro", "elite"])

    with col3:
        training_hours_per_week = st.number_input(
            "Training hours per week", min_value=0.0, max_value=30.0, value=6.0, step=0.5
        )
        previous_injuries_count = st.number_input(
            "Previous injuries (any site)", min_value=0, max_value=20, value=1
        )

    st.markdown("<div class='section-title'>3. Injury details</div>", unsafe_allow_html=True)
    col4, col5 = st.columns(2)

    with col4:
        injury_site = st.selectbox(
            "Injury site",
            options=["hamstring", "quadriceps", "calf", "ankle", "knee", "lower_back", "shoulder", "achilles"]
        )
        injury_type = st.selectbox(
            "Injury type",
            options=["muscle_strain", "tendonitis", "ligament_sprain", "overuse"]
        )
        injury_severity = st.selectbox("Injury severity (1–3)", options=[1, 2, 3])

    with col5:
        onset = st.selectbox("Onset", options=["acute", "gradual"])
        swelling = st.selectbox("Swelling", options=["none", "mild", "moderate", "severe"])
        previous_same_site_injury = st.selectbox(
            "Previous same-site injury?",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.markdown("<div class='section-title'>4. Pain & function</div>", unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)

    with col6:
        time_since_onset_days = st.number_input(
            "Time since onset (days)", min_value=0, max_value=365, value=7
        )

    with col7:
        pain_at_rest = st.number_input(
            "Pain at rest (0–10)", min_value=0, max_value=10, value=3
        )

    with col8:
        pain_on_activity = st.number_input(
            "Pain on activity (0–10)", min_value=0, max_value=10, value=7
        )

    range_of_motion_limit = st.slider(
        "Range of motion limitation (0 = no restriction, 10 = very restricted)",
        min_value=0, max_value=10, value=5
    )

    submit_button = st.form_submit_button("Get treatment recommendations")

# -----------------------------
# 4. RUN MODEL & DISPLAY RESULTS
# -----------------------------

if submit_button:
    # Build the input dict for the model
    input_dict = {
        "subjective_injury_description": subjective_injury_description,
        "age": float(age),
        "training_hours_per_week": float(training_hours_per_week),
        "previous_injuries_count": int(previous_injuries_count),
        "previous_same_site_injury": int(previous_same_site_injury),
        "time_since_onset_days": int(time_since_onset_days),
        "pain_at_rest": int(pain_at_rest),
        "pain_on_activity": int(pain_on_activity),
        "range_of_motion_limit": int(range_of_motion_limit),
        "sex": sex,
        "sport": sport,
        "level": level,
        "injury_site": injury_site,
        "injury_type": injury_type,
        "injury_severity": int(injury_severity),
        "onset": onset,
        "swelling": swelling,
        # placeholder – this will be overwritten during simulation
        "treatment_modality": "physio",
    }

    with st.spinner("Running triage model and simulating treatment options..."):
        df_results = recommend_treatment(input_dict)

    st.markdown("### Recommended treatment options")
    df_display = df_results.copy()
    df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
    df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)
    st.table(df_display)

    best_row = df_display.iloc[0]
    st.markdown(
        f"<div class='card'>"
        f"<b>Suggested plan:</b> {best_row['treatment_modality']}<br>"
        f"Estimated success probability: <b>{best_row['Predicted success (%)']}%</b> "
        f"based on similar synthetic cases."
        f"</div>",
        unsafe_allow_html=True
    )

    # Simple “clinical feel” message (still synthetic!)
    if pain_on_activity >= 8 or (injury_severity == 3 and range_of_motion_limit >= 7):
        st.warning(
            "High pain levels and/or severe functional limitation detected. "
            "In a real clinical setting, this would be a flag for thorough assessment "
            "and possibly imaging or onward referral."
        )
