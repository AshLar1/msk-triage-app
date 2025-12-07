import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =========================================
# 1. LOAD MODEL
# =========================================
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


# =========================================
# 2. PAGE CONFIG & GLOBAL STYLES
# =========================================
st.set_page_config(
    page_title="MSK Triage Recommender",
    page_icon="💙",
    layout="wide",
)

st.markdown(
    """
    <style>
        /* Global */
        html, body, [class*="css"] {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .main {
            padding-top: 1.5rem;
        }

        /* Top nav */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0 1.0rem 0;
        }
        .top-nav-left {
            font-weight: 700;
            font-size: 1.05rem;
            color: #0f172a;
        }
        .top-nav-right a {
            margin-left: 1.0rem;
            font-size: 0.9rem;
            color: #0f172a;
            text-decoration: none;
            opacity: 0.8;
        }
        .top-nav-right a:hover {
            opacity: 1.0;
        }

        /* Hero */
        .hero {
            border-radius: 1.2rem;
            padding: 2.0rem 2.4rem;
            background: radial-gradient(circle at top left, #4f46e5 0, #1d4ed8 45%, #020617 100%);
            color: white;
            margin-bottom: 1.8rem;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.7rem;
            border-radius: 999px;
            background: rgba(15,23,42,0.45);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-sub {
            font-size: 0.98rem;
            max-width: 520px;
            opacity: 0.92;
        }
        .hero-cta-row {
            margin-top: 1.3rem;
            display: flex;
            gap: 0.7rem;
            align-items: center;
        }
        .hero-cta-primary {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.55rem 1.2rem;
            border-radius: 999px;
            background: #f97316;
            color: white;
            font-size: 0.9rem;
            font-weight: 600;
            text-decoration: none;
        }
        .hero-cta-primary:hover {
            background: #ea580c;
        }
        .hero-cta-secondary {
            font-size: 0.9rem;
            opacity: 0.9;
        }

        /* Cards / sections */
        .card {
            padding: 1.3rem 1.5rem;
            border-radius: 1rem;
            border: 1px solid #e2e8f0;
            background-color: #ffffff;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04);
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
            color: #0f172a;
        }
        .section-sub {
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 0.9rem;
        }
        .pill-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin-bottom: 0.2rem;
        }

        /* Form headings */
        .form-section-title {
            font-size: 1.0rem;
            font-weight: 600;
            margin-top: 0.4rem;
            margin-bottom: 0.1rem;
            color: #0f172a;
        }

        /* Results */
        .result-highlight {
            margin-top: 0.6rem;
            padding: 0.9rem 1.0rem;
            border-radius: 0.9rem;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            font-size: 0.9rem;
        }
        .result-badge {
            display: inline-block;
            padding: 0.1rem 0.6rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.75rem;
            margin-left: 0.3rem;
        }

        /* Footer */
        .footer {
            margin-top: 2.0rem;
            padding-top: 1.0rem;
            font-size: 0.8rem;
            color: #94a3b8;
            border-top: 1px solid #e2e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# 3. TOP NAV + HERO
# =========================================
nav_col1, nav_col2, nav_col3 = st.columns([0.1, 0.8, 0.1])
with nav_col2:
    st.markdown(
        """
        <div class="top-nav">
            <div class="top-nav-left">
                MSK Triage Prototype
            </div>
            <div class="top-nav-right">
                <a href="#how-it-works">How it works</a>
                <a href="#try-prototype">Try the prototype</a>
                <a href="#about-model">About the model</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hero_l, hero_r, hero_pad = st.columns([0.6, 0.35, 0.05])
with hero_l:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">AI-enabled soft tissue triage • Prototype</div>
            <div class="hero-title">Explore recovery pathways for MSK injuries.</div>
            <p class="hero-sub">
                This demo uses a machine learning model trained on synthetic sports and MSK cases
                to estimate which treatment plans are most likely to lead to a successful outcome.
                It’s designed to help you imagine what digital triage could look like in your own service.
            </p>
            <div class="hero-cta-row">
                <a class="hero-cta-primary" href="#try-prototype">Try the prototype</a>
                <span class="hero-cta-secondary">No real patient data. Not for clinical use.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_r:
    st.markdown(
        """
        <div class="card">
          <div class="pill-label">Prototype at a glance</div>
          <p style="font-size:0.9rem; margin-bottom:0.6rem;">
            • Synthetic dataset of 1,000 soft tissue / MSK cases<br>
            • Combines structured injury data with free-text descriptions<br>
            • Estimates likelihood of a <b>successful return to sport / activity</b>
          </p>
          <p style="font-size:0.8rem; color:#64748b;">
            The goal is not to replace clinical judgement, but to explore how AI might
            support prioritisation, self-management advice and routing in the future.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<a name='how-it-works'></a>", unsafe_allow_html=True)

# =========================================
# 4. HOW IT WORKS SECTION
# =========================================
st.markdown("### How it works")
st.markdown(
    """
    <p class="section-sub">
        The prototype follows the same broad pattern as many AI triage tools: capture a rich
        history, match to similar prior cases, and surface a suggested route that can be
        reviewed by a clinician or service designer.
    </p>
    """,
    unsafe_allow_html=True,
)

hw_col1, hw_col2, hw_col3 = st.columns(3)
with hw_col1:
    st.markdown(
        """
        <div class="card">
            <div class="pill-label">Step 1</div>
            <div class="section-title" style="font-size:1.0rem;">Capture the story</div>
            <p style="font-size:0.85rem; color:#475569;">
                The user describes their injury in their own words, alongside sport, training
                load and basic history of previous injuries.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hw_col2:
    st.markdown(
        """
        <div class="card">
            <div class="pill-label">Step 2</div>
            <div class="section-title" style="font-size:1.0rem;">Model the probabilities</div>
            <p style="font-size:0.85rem; color:#475569;">
                The model compares the case against synthetic examples and estimates the
                likelihood of a successful outcome under different treatment modalities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hw_col3:
    st.markdown(
        """
        <div class="card">
            <div class="pill-label">Step 3</div>
            <div class="section-title" style="font-size:1.0rem;">Surface options</div>
            <p style="font-size:0.85rem; color:#475569;">
                You see a ranked list of options, so you can explore how different plans
                might impact recovery in a safe, sandboxed environment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("<a name='try-prototype'></a>", unsafe_allow_html=True)

# =========================================
# 5. TRIAGE FORM SECTION
# =========================================
st.markdown("## Try the prototype triage tool")
st.markdown(
    """
    <p class="section-sub">
        Use the example below or adjust the fields to match a scenario you recognise from your
        own caseload. The outputs are based on synthetic training data and are for exploration only.
    </p>
    """,
    unsafe_allow_html=True,
)

form_col_l, form_col_r = st.columns([0.63, 0.37])

with form_col_l:
    with st.form("triage_form"):
        st.markdown("<div class='form-section-title'>1. Subjective injury description</div>", unsafe_allow_html=True)
        subjective_injury_description = st.text_area(
            "",
            value="Felt a sharp pain in the back of my thigh while sprinting. Pain is worse with fast running and high-speed drills.",
            height=110
        )

        st.markdown("<div class='form-section-title'>2. Patient & training profile</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=16, max_value=80, value=28)
            sex = st.selectbox("Sex", options=["male", "female"])
        with c2:
            sport = st.selectbox("Sport", options=["football", "running", "tennis", "gym", "rugby", "basketball"])
            level = st.selectbox("Sport level", options=["recreational", "semi_pro", "elite"])
        with c3:
            training_hours_per_week = st.number_input(
                "Training hours / week", min_value=0.0, max_value=30.0, value=6.0, step=0.5
            )
            previous_injuries_count = st.number_input(
                "Previous injuries (any site)", min_value=0, max_value=20, value=1
            )

        st.markdown("<div class='form-section-title'>3. Injury details</div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            injury_site = st.selectbox(
                "Injury site",
                options=["hamstring", "quadriceps", "calf", "ankle", "knee", "lower_back", "shoulder", "achilles"]
            )
            injury_type = st.selectbox(
                "Injury type",
                options=["muscle_strain", "tendonitis", "ligament_sprain", "overuse"]
            )
        with c5:
            injury_severity = st.selectbox("Injury severity (1–3)", options=[1, 2, 3])
            onset = st.selectbox("Onset", options=["acute", "gradual"])
            swelling = st.selectbox("Swelling", options=["none", "mild", "moderate", "severe"])
            previous_same_site_injury = st.selectbox(
                "Previous same-site injury?",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )

        st.markdown("<div class='form-section-title'>4. Pain & function</div>", unsafe_allow_html=True)
        c6, c7, c8 = st.columns(3)
        with c6:
            time_since_onset_days = st.number_input(
                "Time since onset (days)", min_value=0, max_value=365, value=7
            )
        with c7:
            pain_at_rest = st.number_input(
                "Pain at rest (0–10)", min_value=0, max_value=10, value=3
            )
        with c8:
            pain_on_activity = st.number_input(
                "Pain on activity (0–10)", min_value=0, max_value=10, value=7
            )

        range_of_motion_limit = st.slider(
            "Range of motion limitation (0 = no restriction, 10 = very restricted)",
            min_value=0, max_value=10, value=5
        )

        submit_btn = st.form_submit_button("Run triage simulation")

with form_col_r:
    st.markdown(
        """
        <div class="card">
          <div class="pill-label">What you’ll see</div>
          <p style="font-size:0.88rem; color:#475569;">
            • Ranked list of treatment modalities (e.g. physio, osteopath).<br>
            • Relative likelihood of a successful outcome in this synthetic dataset.<br>
            • A short explanation of how to interpret the suggestions.
          </p>
          <p style="font-size:0.8rem; color:#94a3b8;">
            In a real deployment, this kind of model would sit alongside clinical triage,
            safety netting and local pathways – not replace them.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# 6. RUN MODEL & SHOW RESULTS
# =========================================
if submit_btn:
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
        "treatment_modality": "physio",  # placeholder, overwritten in simulation
    }

    with st.spinner("Running model and simulating treatment options..."):
        df_results = recommend_treatment(input_dict)

    st.markdown("### Suggested treatment pathways")

    df_display = df_results.copy()
    df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
    df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)

    res_col1, res_col2 = st.columns([0.55, 0.45])

    with res_col1:
        st.table(df_display)

    with res_col2:
        best_row = df_display.iloc[0]
        st.markdown(
            f"""
            <div class="card">
                <div class="pill-label">Top suggestion</div>
                <div class="section-title" style="font-size:1.0rem; margin-bottom:0.2rem;">
                    {best_row['treatment_modality'].title()}
                    <span class="result-badge">~{best_row['Predicted success (%)']}% (synthetic)</span>
                </div>
                <p style="font-size:0.86rem; color:#475569;">
                    Within this synthetic dataset, similar injuries and training profiles were most likely
                    to have a successful outcome when managed primarily with
                    <b>{best_row['treatment_modality'].replace('_',' ')}</b>.
                    In reality, this kind of signal would be combined with your own service design,
                    local pathways and clinical reasoning.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pain_on_activity >= 8 or (int(injury_severity) == 3 and range_of_motion_limit >= 7):
            st.warning(
                "High pain levels and/or marked functional restriction detected. "
                "In a live system, this scenario would likely be routed to a high-priority or "
                "face-to-face pathway with clear safety-netting."
            )

st.markdown("<a name='about-model'></a>", unsafe_allow_html=True)

# =========================================
# 7. ABOUT THE MODEL / FOOTER
# =========================================
st.markdown("## About this prototype")
st.markdown(
    """
    - Trained on **1,000 synthetic MSK and soft-tissue injury cases** (no real patients).
    - Inputs include: sport, level, training load, injury site/type, severity, pain scores and a
      free-text description of the injury.
    - The model is a Random Forest classifier wrapped in a preprocessing pipeline with TF–IDF
      for text and one-hot encoding for categorical fields.
    - The output is the probability of a <b>successful outcome</b> for each treatment modality,
      within this synthetic dataset only.
    """
)

st.markdown(
    """
    <div class="footer">
        This is a concept demo for exploring AI-enabled MSK triage. It is <b>not</b> a medical
        device and must not be used for real clinical decisions or patient advice.
    </div>
    """,
    unsafe_allow_html=True,
)
