import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. LOAD MODEL
# -----------------------------
MODEL_PATH = "msk_triage_rf_model.pkl"
model = joblib.load(MODEL_PATH)

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
# 2. PAGE CONFIG + GLOBAL STYLE
# -----------------------------
st.set_page_config(
    page_title="MSK Triage Recommender",
    page_icon="💪",
    layout="wide",
)

# Simple styling
st.markdown("""
    <style>
        .main { padding-top: 1.5rem; }
        body, input, textarea, select {
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
        }
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 1rem;
            background: linear-gradient(135deg, #1f6feb, #3b8cff);
            color: white;
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: 2rem;
            margin-bottom: 0.4rem;
        }
        .hero-sub {
            font-size: 0.98rem;
            opacity: 0.9;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            font-size: 0.75rem;
            margin-bottom: 0.5rem;
        }
        .card {
            padding: 1.3rem 1.5rem;
            border-radius: 0.9rem;
            border: 1px solid #e0e0e0;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 0.8rem;
            margin-bottom: 0.3rem;
            color: #12355b;
        }
        .step-number {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #e3edff;
            color: #1f6feb;
            font-weight: 600;
            margin-right: 0.5rem;
            font-size: 0.85rem;
        }
        .small-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }
        .result-badge {
            display: inline-block;
            padding: 0.12rem 0.55rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.75rem;
            margin-left: 0.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. LANDING / HERO SECTION
# -----------------------------
left_hero, right_hero = st.columns([2.2, 1.8])

with left_hero:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">Prototype • Soft tissue & MSK</div>
            <h1>MSK Triage Recommender</h1>
            <p class="hero-sub">
                Explore how different treatment options might influence recovery for common
                soft tissue and musculoskeletal injuries. Built on synthetic data to help you
                design future digital pathways for real patients and athletes.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with right_hero:
    st.markdown(
        """
        <div class="card">
          <div class="small-label">How this works</div>
          <p style="font-size:0.9rem; margin-bottom:0.6rem;">
            1. Enter an example injury presentation<br>
            2. The model simulates different treatment modalities<br>
            3. You see a ranked list of options by predicted success.
          </p>
          <p style="font-size:0.8rem; color:#64748b;">
            This is a prototype trained on synthetic data only – it is <b>not</b> intended
            for real clinical use or patient-facing deployment.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# 4. "HOW IT WORKS" STRIP
# -----------------------------
st.markdown("### How the prototype works")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="card">
          <div><span class="step-number">1</span><b>Describe the injury</b></div>
          <p style="font-size:0.85rem; margin-top:0.4rem;">
            Capture key details: mechanism, pain behaviour, time since onset and
            functional impact.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_b:
    st.markdown(
        """
        <div class="card">
          <div><span class="step-number">2</span><b>Add context</b></div>
          <p style="font-size:0.85rem; margin-top:0.4rem;">
            Include training load, sport, previous injuries and severity to give
            better context around risk and capacity.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_c:
    st.markdown(
        """
        <div class="card">
          <div><span class="step-number">3</span><b>View suggested pathways</b></div>
          <p style="font-size:0.85rem; margin-top:0.4rem;">
            The model predicts the likelihood of a successful outcome under
            different treatment modalities and ranks them.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------
# 5. ASSESSMENT FORM
# -----------------------------
st.markdown("## Start a new triage simulation")

with st.form("triage_form"):
    st.markdown("<div class='section-title'>1. Subjective injury description</div>", unsafe_allow_html=True)
    subjective_injury_description = st.text_area(
        "",
        value="Felt a sharp pain in the back of my thigh while sprinting. Pain is worse with fast running and high-speed drills.",
        height=100
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

    with col5:
        injury_severity = st.selectbox("Injury severity (1–3)", options=[1, 2, 3])
        onset = st.selectbox("Onset", options=["acute", "gradual"])
        swelling = st.selectbox("Swelling", options=["none", "mild", "moderate", "severe"])

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
# 6. RESULTS AREA
# -----------------------------
if submit_button:
    input_dict = {
        "subjective_injury_description": subjective_injury_description,
        "age": float(age),
        "training_hours_per_week": float(training_hours_per_week),
        "previous_injuries_count": int(previous_injuries_count),
        "previous_same_site_injury": 1 if "Yes" in str(previous_injuries_count) else int(0),
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
        "treatment_modality": "physio",  # placeholder; overwritten in simulation
    }

    with st.spinner("Running triage model and simulating treatment options..."):
        df_results = recommend_treatment(input_dict)

    st.markdown("## Suggested treatment pathways")

    tab1, tab2 = st.tabs(["Ranked options", "What this means"])

    with tab1:
        df_display = df_results.copy()
        df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
        df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)
        st.table(df_display)

        best_row = df_display.iloc[0]
        st.markdown(
            f"<div class='card'>"
            f"<div class='small-label'>Top suggestion</div>"
            f"<p style='font-size:0.95rem; margin-bottom:0.3rem;'>"
            f"<b>{best_row['treatment_modality'].title()}</b>"
            f"<span class='result-badge'>~{best_row['Predicted success (%)']}% estimated success</span>"
            f"</p>"
            f"<p style='font-size:0.85rem; color:#475569;'>"
            f"This reflects the patterns learned from synthetic cases with a similar injury profile, "
            f"pain picture and training context. In a real deployment, this would be combined with "
            f"clinical reasoning – not replace it."
            f"</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        if pain_on_activity >= 8 or (int(injury_severity) == 3 and range_of_motion_limit >= 7):
            st.warning(
                "High pain levels and/or severe functional limitation detected. "
                "In a real clinical workflow this would trigger a more detailed assessment, "
                "and possibly imaging or onward referral."
            )

    with tab2:
        st.markdown(
            """
            - The model was trained on **synthetic soft tissue / MSK cases** only.
            - It uses your inputs (injury description, severity, sport, training load etc.) to simulate how
              the same case might respond under different treatment modalities.
            - The probabilities shown are **relative** within this synthetic dataset – they are not real-world
              outcome probabilities and should not be used to guide clinical care.
            - The purpose of this prototype is to explore how AI-driven triage and pathway suggestion could
              be integrated into digital MSK products in the future.
            """
        )
