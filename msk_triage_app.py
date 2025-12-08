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
# 1. MODEL LOADING & TRIAGE LOGIC
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
    """
    Takes a single case (dict), simulates each treatment modality,
    and returns a DataFrame with predicted success probabilities.
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


# =========================================
# 2. PAGE CONFIG & GLOBAL STYLES
# =========================================
st.set_page_config(
    page_title="MSK Triage Prototype",
    page_icon="💙",
    layout="wide",
)

# Global CSS – Visiba-inspired aesthetic
st.markdown(
    """
    <style>
        body {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: #f7f9fb;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0rem;
            max-width: 1100px;
        }

        /* NAV BAR */
        .navbar {
            width: 100%;
            padding: 1.0rem 0 1.0rem 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-left {
            font-size: 1.4rem;
            font-weight: 600;
            color: #12355b;
        }
        .nav-right span {
            margin-left: 1.1rem;
            font-size: 0.9rem;
            color: #12355b;
            cursor: pointer;
        }
        .nav-right span:hover {
            text-decoration: underline;
        }
        .nav-pill {
            background:#0059ff;
            color:white;
            padding:0.45rem 1.1rem;
            border-radius:20px;
            font-weight:500;
        }

        /* HERO */
        .hero {
            background: #d7e8ef;
            padding: 3.5rem 3rem;
            border-radius: 1rem;
            margin-bottom: 3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 2.5rem;
        }
        .hero-title {
            font-size: 3rem;
            font-family: Georgia, 'Times New Roman', serif;
            color: #12355b;
            line-height: 1.15;
        }
        .hero-sub {
            font-size: 1.05rem;
            margin-top: 1rem;
            color: #243b53;
            max-width: 430px;
        }
        .hero-btn {
            margin-top: 1.8rem;
            display: inline-block;
            padding: 0.8rem 1.7rem;
            background: #0059ff;
            color: white !important;
            border-radius: 30px;
            font-size: 0.98rem;
            font-weight: 600;
            text-decoration: none !important;
        }
        .hero-btn:hover {
            background: #0042c4;
        }

        /* SECTION HEADINGS */
        .section-title {
            font-size: 2rem;
            font-family: Georgia, 'Times New Roman', serif;
            color: #12355b;
            margin-bottom: 0.4rem;
            text-align: center;
        }
        .section-subtitle {
            text-align: center;
            font-size: 0.96rem;
            color: #64748b;
            max-width: 780px;
            margin: 0 auto 1.6rem auto;
        }

        /* FEATURE CARDS */
        .feature-grid {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            margin-top: 2rem;
        }
        .feature-card {
            flex: 1;
            background: white;
            padding: 1.8rem;
            border-radius: 0.9rem;
            box-shadow: 0 6px 18px rgba(15,23,42,0.08);
        }
        .feature-card-title {
            font-size: 1.2rem;
            color: #12355b;
            font-family: Georgia, serif;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .feature-card p {
            font-size: 0.9rem;
            color: #4b5563;
            text-align: center;
        }

        /* FORM CARD */
        .form-card {
            background: white;
            padding: 2rem 2.2rem;
            border-radius: 1rem;
            box-shadow: 0 6px 20px rgba(15,23,42,0.08);
            margin-top: 2.2rem;
            margin-bottom: 2rem;
        }
        .form-section-title {
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #12355b;
        }

        /* RESULTS CARD */
        .results-card {
            background: white;
            padding: 1.8rem 2rem;
            border-radius: 1rem;
            box-shadow: 0 6px 20px rgba(15,23,42,0.08);
            margin-top: 1rem;
            margin-bottom: 2.5rem;
        }
        .results-highlight {
            margin-top: 0.9rem;
            padding: 0.8rem 1rem;
            border-radius: 0.8rem;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            font-size: 0.92rem;
            color: #1e293b;
        }

        /* FOOTER */
        .footer {
            margin-top: 1.5rem;
            padding-top: 0.8rem;
            border-top: 1px solid #e2e8f0;
            font-size: 0.8rem;
            color: #94a3b8;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================
# 3. NAVBAR
# =========================================
st.markdown(
    """
    <div class="navbar">
        <div class="nav-left">MSK Triage</div>
        <div class="nav-right">
            <span onclick="window.location.hash='#what-is';">What it is</span>
            <span onclick="window.location.hash='#impact';">Impact</span>
            <span class="nav-pill" onclick="window.location.hash='#triage-form';">
                Try prototype
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================
# 4. HERO SECTION
# =========================================
hero_col = st.container()
with hero_col:
    st.markdown(
        """
        <div class="hero">
            <div>
                <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.09em; color:#334e68; margin-bottom:0.3rem;">
                    AI-enabled MSK triage · Prototype
                </div>
                <div class="hero-title">
                    Smarter triage.<br>Better MSK care.
                </div>
                <div class="hero-sub">
                    Explore how AI could support soft tissue and musculoskeletal triage by
                    simulating likely outcomes for different treatment approaches – using
                    synthetic data only.
                </div>
                <a class="hero-btn" href="#triage-form">Try the triage prototype</a>
            </div>
            <div>
                <!-- Soft tissue / pain-related illustration -->
                <img src="https://images.pexels.com/photos/4506109/pexels-photo-4506109.jpeg?auto=compress&cs=tinysrgb&w=700"
                     alt="Soft tissue injury and pain assessment"
                     style="border-radius:0.8rem; box-shadow:0 15px 35px rgba(15,23,42,0.35); max-width:360px;">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================
# 5. WHAT IS / IMPACT SECTIONS
# =========================================
st.markdown('<a id="what-is"></a>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What is this MSK triage prototype?</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="section-subtitle">
        This concept model estimates the likelihood of a successful outcome (for example, return to sport)
        under different treatment modalities for soft tissue and musculoskeletal injuries. It is trained only
        on synthetic data and is <b>not</b> intended for clinical use.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<a id="impact"></a>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="feature-grid">
        <div class="feature-card">
            <div style="font-size:2rem; text-align:center;">📋</div>
            <div class="feature-card-title">Reliable summaries</div>
            <p>Capture consistent histories and pain profiles that are easy to review at a glance.</p>
        </div>
        <div class="feature-card">
            <div style="font-size:2rem; text-align:center;">🧠</div>
            <div class="feature-card-title">Suggested pathways</div>
            <p>Estimate which treatment modalities are most likely to succeed for similar cases.</p>
        </div>
        <div class="feature-card">
            <div style="font-size:2rem; text-align:center;">⚡</div>
            <div class="feature-card-title">Service design sandbox</div>
            <p>Use the prototype to explore how AI-enabled triage could support your MSK service.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================
# 6. FORM – TRIAGE PROTOTYPE
# =========================================
st.markdown('<a id="triage-form"></a>', unsafe_allow_html=True)
st.markdown('<div class="section-title" style="margin-top:3rem;">Try the MSK triage prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Use the example case below or adjust the fields to mirror a typical injury from your service.</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    with st.form("triage_form"):
        # Subjective description
        st.markdown('<div class="form-section-title">1. Subjective description</div>', unsafe_allow_html=True)
        subjective_injury_description = st.text_area(
            "",
            value="Felt a sharp pain in the back of my thigh while sprinting. Pain is worse with fast running and high-speed drills.",
            height=90
        )

        # Patient & training profile
        st.markdown('<div class="form-section-title" style="margin-top:1.0rem;">2. Patient & training profile</div>', unsafe_allow_html=True)
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

        # Injury details
        st.markdown('<div class="form-section-title" style="margin-top:1.0rem;">3. Injury details</div>', unsafe_allow_html=True)
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

        # Pain & function
        st.markdown('<div class="form-section-title" style="margin-top:1.0rem;">4. Pain & function</div>', unsafe_allow_html=True)
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

        submitted = st.form_submit_button("Run triage simulation")

    st.markdown('</div>', unsafe_allow_html=True)  # close form-card

# =========================================
# 7. RUN MODEL & SHOW RESULTS
# =========================================
if submitted:
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
        "treatment_modality": "physio",  # placeholder; overwritten in simulation
    }

    with st.spinner("Running model and simulating treatment options..."):
        df_results = recommend_treatment(input_dict)

    # Prepare display
    df_display = df_results.copy()
    df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
    df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)

    st.markdown('<div class="results-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-bottom:0.6rem; color:#12355b;">Suggested treatment pathways</h3>', unsafe_allow_html=True)

    col_table, col_text = st.columns([0.55, 0.45])
    with col_table:
        st.table(df_display)

    with col_text:
        best_row = df_display.iloc[0]
        st.markdown(
            f"""
            <div class="results-highlight">
                <b>Top suggestion:</b> {best_row['treatment_modality'].title()}<br>
                Estimated success likelihood in this synthetic dataset:
                <b>{best_row['Predicted success (%)']}%</b>.
                <br><br>
                In a real deployment, this kind of model would sit alongside clinical
                judgement, local pathways and safety netting – not replace them.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pain_on_activity >= 8 or (int(injury_severity) == 3 and range_of_motion_limit >= 7):
            st.warning(
                "High pain and/or marked functional restriction detected. "
                "In a live system, this scenario would likely be routed to a high-priority pathway."
            )

    st.markdown('</div>', unsafe_allow_html=True)  # close results-card

# =========================================
# 8. FOOTER
# =========================================
st.markdown(
    """
    <div class="footer">
        MSK Triage Prototype · Built on synthetic data for exploration only. Not a medical device.
    </div>
    """,
    unsafe_allow_html=True,
)
