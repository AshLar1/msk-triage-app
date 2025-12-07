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
    page_title="MSK Triage Prototype",
    page_icon="💙",
    layout="wide",
)

st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .main {
            padding-top: 1.2rem;
        }

        /* Top nav */
        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0 1.0rem 0;
        }
        .nav-logo {
            font-weight: 700;
            font-size: 1.1rem;
            color: #111827;
        }
        .nav-links a {
            margin-left: 1.0rem;
            font-size: 0.88rem;
            color: #111827;
            text-decoration: none;
            opacity: 0.75;
        }
        .nav-links a:hover {
            opacity: 1.0;
        }
        .nav-cta {
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: #2563eb;
            color: #ffffff !important;
            font-weight: 500;
        }

        /* Hero / landing */
        .hero-wrapper {
            background-color: #cfe6f3; /* soft blue, Visiba-like */
            border-radius: 0.8rem;
            padding: 3.2rem 3.0rem;
            margin-bottom: 2.2rem;
        }
        .hero-inner {
            display: flex;
            flex-direction: row;
            align-items: center;
            justify-content: space-between;
            gap: 2.5rem;
        }
        .hero-left {
            max-width: 540px;
        }
        .hero-eyebrow {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #1d3557;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 3rem;
            line-height: 1.05;
            color: #102a43;
            margin-bottom: 0.5rem;
        }
        .hero-sub {
            font-size: 0.98rem;
            color: #1f2933;
            max-width: 460px;
        }
        .hero-cta-row {
            margin-top: 1.4rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        .hero-cta-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.7rem 1.5rem;
            border-radius: 999px;
            background: #2563eb;
            color: white;
            font-size: 0.92rem;
            font-weight: 600;
            text-decoration: none;
        }
        .hero-cta-btn:hover {
            background: #1d4ed8;
        }
        .hero-cta-sub {
            font-size: 0.85rem;
            color: #1f2933;
            opacity: 0.8;
        }

        .hero-illustration {
            flex: 1;
            min-height: 220px;
            border-radius: 1rem;
            background: white;
            border: 1px solid #e5e7eb;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1.5rem;
        }
        .hero-illustration-inner {
            width: 100%;
            max-width: 380px;
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 0.8rem;
        }
        .hero-panel {
            border-radius: 0.7rem;
            border: 1px solid #e5e7eb;
            padding: 0.7rem;
            background: #f9fafb;
        }
        .hero-panel-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.3rem;
        }
        .hero-badge {
            display: inline-block;
            font-size: 0.7rem;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            margin-top: 0.25rem;
        }

        /* Section headings */
        .section-heading {
            font-family: "Georgia", "Times New Roman", serif;
            font-size: 1.9rem;
            color: #102a43;
            text-align: center;
            margin-top: 1.5rem;
            margin-bottom: 0.2rem;
        }
        .section-subtitle {
            text-align: center;
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 1.4rem;
        }

        .card {
            padding: 1.3rem 1.5rem;
            border-radius: 1rem;
            border: 1px solid #e2e8f0;
            background-color: #ffffff;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .benefit-title {
            font-size: 1.05rem;
            font-weight: 600;
            margin-top: 0.6rem;
            margin-bottom: 0.3rem;
            color: #111827;
            font-family: "Georgia", "Times New Roman", serif;
        }
        .benefit-body {
            font-size: 0.9rem;
            color: #4b5563;
        }

        .trust-strip {
            background: #f9fafb;
            padding: 1.6rem 1.2rem;
            border-radius: 0.7rem;
            margin-top: 2rem;
            margin-bottom: 1.8rem;
            border: 1px solid #e5e7eb;
        }

        .trust-logos {
            display: flex;
            justify-content: center;
            gap: 1.8rem;
            flex-wrap: wrap;
            margin-top: 0.8rem;
        }
        .trust-logo-pill {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            border: 1px dashed #cbd5f5;
            font-size: 0.8rem;
            color: #475569;
            background: white;
        }

        .footer {
            margin-top: 2.4rem;
            padding-top: 1.0rem;
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
# 3. TOP NAV + HERO
# =========================================
nav_col1, nav_col2, nav_col3 = st.columns([0.12, 0.76, 0.12])
with nav_col2:
    st.markdown(
        """
        <div class="top-nav">
            <div class="nav-logo">MSK Triage</div>
            <div class="nav-links">
                <a href="#what-is-msk">What it is</a>
                <a href="#impact">Impact</a>
                <a href="#try-prototype" class="nav-cta">Try prototype</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

hero_outer_col1, hero_outer_col2, hero_outer_col3 = st.columns([0.08, 0.84, 0.08])
with hero_outer_col2:
    st.markdown(
        """
        <div class="hero-wrapper">
          <div class="hero-inner">
            <div class="hero-left">
              <div class="hero-eyebrow">AI-enabled MSK triage • Prototype</div>
              <div class="hero-title">
                Smarter triage.<br>Better MSK care.
              </div>
              <p class="hero-sub">
                Explore how AI could support soft tissue and musculoskeletal triage by
                simulating likely outcomes for different treatment approaches – using
                synthetic data only.
              </p>
              <div class="hero-cta-row">
                <a href="#try-prototype" class="hero-cta-btn">Try the triage prototype</a>
                <div class="hero-cta-sub">Takes less than a minute. No real patient data.</div>
              </div>
            </div>

            <div class="hero-illustration">
              <div class="hero-illustration-inner">
                <div class="hero-panel">
                  <div class="hero-panel-title">Injury details</div>
                  <p style="font-size:0.8rem; color:#4b5563;">
                    “Sharp pain in the back of my thigh when sprinting. Worse with acceleration.”
                  </p>
                  <span class="hero-badge">Captured via digital triage</span>
                </div>
                <div class="hero-panel">
                  <div class="hero-panel-title">Suggested pathways</div>
                  <p style="font-size:0.8rem; color:#111827; margin-bottom:0.3rem;">
                    • Physio-led rehab<br>
                    • Load management plan<br>
                    • Return-to-sport review
                  </p>
                  <span class="hero-badge">AI-generated priorities</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================
# 4. "WHAT IS IT?" SECTION
# =========================================
st.markdown("<a name='what-is-msk'></a>", unsafe_allow_html=True)
sec1_col_outer1, sec1_col_outer2, sec1_col_outer3 = st.columns([0.08, 0.84, 0.08])
with sec1_col_outer2:
    left, right = st.columns([0.48, 0.52])

    with left:
        st.image(
            "https://images.pexels.com/photos/6129040/pexels-photo-6129040.jpeg?auto=compress&cs=tinysrgb&w=800",
            caption="Concept image • digital triage in clinic",
            use_column_width=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-heading" style="text-align:left; margin-top:0;">
                What is this MSK triage prototype?
            </div>
            <p style="font-size:0.92rem; color:#475569; margin-bottom:0.8rem;">
                This tool is an AI-enabled concept model that estimates the likelihood of a
                successful outcome (e.g. return to sport) under different treatment modalities
                for soft tissue and MSK injuries.
            </p>
            <p style="font-size:0.9rem; color:#475569; margin-bottom:0.6rem;">
                Patients would normally submit details of their symptoms online or via an app,
                and clinicians could review a concise summary with suggested next steps.
            </p>
            <ul style="font-size:0.9rem; color:#1f2937; padding-left:1.1rem;">
                <li>Reliable, structured summaries of subjective history.</li>
                <li>Estimated likelihood of a successful outcome for different plans.</li>
                <li>Support for routing patients to the most appropriate point of care.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# =========================================
# 5. IMPACT / BENEFITS SECTION
# =========================================
st.markdown("<a name='impact'></a>", unsafe_allow_html=True)
st.markdown('<div class="section-heading">Experience the impact of AI-enabled triage</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Imagine applying this kind of intelligence to your own MSK or sports medicine service.</div>',
    unsafe_allow_html=True,
)

b_outer1, b_outer2, b_outer3 = st.columns([0.08, 0.84, 0.08])
with b_outer2:
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown(
            """
            <div class="card" style="text-align:left;">
                <div style="font-size:2rem;">💼</div>
                <div class="benefit-title">Ease pressure</div>
                <div class="benefit-body">
                    Reduce workloads by capturing high-quality histories up front and
                    giving teams a clearer picture before first contact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            """
            <div class="card" style="text-align:left;">
                <div style="font-size:2rem;">⚡</div>
                <div class="benefit-title">Improve efficiency</div>
                <div class="benefit-body">
                    Free up time for clinicians to focus on complex cases, while routine
                    presentations follow evidence-based, pre-defined pathways.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            """
            <div class="card" style="text-align:left;">
                <div style="font-size:2rem;">💙</div>
                <div class="benefit-title">Increase adoption</div>
                <div class="benefit-body">
                    Offer a clear, user-friendly digital journey that keeps patients engaged,
                    informed and routed to the right place, first time.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Trust / standards strip (placeholder logos)
t_outer1, t_outer2, t_outer3 = st.columns([0.08, 0.84, 0.08])
with t_outer2:
    st.markdown(
        """
        <div class="trust-strip">
            <div style="text-align:center; font-family:'Georgia',serif; font-size:1.3rem; color:#0f172a;">
                We hold ourselves to the highest standards
            </div>
            <div class="trust-logos">
                <div class="trust-logo-pill">MDR-ready concept</div>
                <div class="trust-logo-pill">ISO 27001 (example)</div>
                <div class="trust-logo-pill">Cyber security best practice</div>
                <div class="trust-logo-pill">Clinical safety by design</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# =========================================
# 6. TRIAGE PROTOTYPE FORM
# =========================================
st.markdown("<a name='try-prototype'></a>", unsafe_allow_html=True)
st.markdown('<div class="section-heading">Try the MSK triage prototype</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Use the example below or adjust the fields to mirror a typical case from your service.</div>',
    unsafe_allow_html=True,
)

form_outer1, form_outer2, form_outer3 = st.columns([0.08, 0.84, 0.08])
with form_outer2:
    form_left, form_right = st.columns([0.6, 0.4])

    with form_left:
        with st.form("triage_form"):
            st.markdown("<div style='font-weight:600; margin-bottom:0.3rem;'>Subjective description</div>", unsafe_allow_html=True)
            subjective_injury_description = st.text_area(
                "",
                value="Felt a sharp pain in the back of my thigh while sprinting. Pain is worse with fast running and high-speed drills.",
                height=110
            )

            st.markdown("<div style='font-weight:600; margin-top:0.8rem;'>Patient & training profile</div>", unsafe_allow_html=True)
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

            st.markdown("<div style='font-weight:600; margin-top:0.8rem;'>Injury details</div>", unsafe_allow_html=True)
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

            st.markdown("<div style='font-weight:600; margin-top:0.8rem;'>Pain & function</div>", unsafe_allow_html=True)
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

    with form_right:
        st.markdown(
            """
            <div class="card">
              <div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:0.1rem;">
                What you’ll see
              </div>
              <p style="font-size:0.9rem; color:#475569;">
                • Ranked treatment modalities (e.g. physio, chiropractor).<br>
                • Relative likelihood of a successful outcome in this synthetic dataset.<br>
                • A short narrative you can use when explaining the concept to colleagues.
              </p>
              <p style="font-size:0.8rem; color:#94a3b8;">
                This is a <b>non-clinical</b> demo and should not be used to make decisions
                about real patients.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================
# 7. RUN MODEL & SHOW RESULTS
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
        "treatment_modality": "physio",  # placeholder; overwritten in simulation
    }

    with st.spinner("Running model and simulating treatment options..."):
        df_results = recommend_treatment(input_dict)

    res_outer1, res_outer2, res_outer3 = st.columns([0.08, 0.84, 0.08])
    with res_outer2:
        st.markdown("### Suggested treatment pathways")

        df_display = df_results.copy()
        df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
        df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)

        rc1, rc2 = st.columns([0.5, 0.5])
        with rc1:
            st.table(df_display)

        with rc2:
            best_row = df_display.iloc[0]
            st.markdown(
                f"""
                <div class="result-highlight">
                    <b>Top suggestion:</b> {best_row['treatment_modality'].title()}
                    <span class="result-badge">~{best_row['Predicted success (%)']}% (synthetic)</span>
                    <br><br>
                    Within this synthetic dataset, similar injury profiles were most likely
                    to have a successful outcome when managed primarily with
                    <b>{best_row['treatment_modality'].replace('_',' ')}</b>.
                    In reality, this signal would sit alongside local pathways and clinical judgement.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if pain_on_activity >= 8 or (int(injury_severity) == 3 and range_of_motion_limit >= 7):
                st.warning(
                    "High pain levels and/or marked functional restriction detected. "
                    "In a live system, this case would likely be routed to a higher-priority pathway."
                )

# =========================================
# 8. FOOTER
# =========================================
st.markdown(
    """
    <div class="footer">
        MSK Triage Prototype – built for exploration using synthetic data only. Not a medical device.
    </div>
    """,
    unsafe_allow_html=True,
)
