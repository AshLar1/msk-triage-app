import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained triage model v2
# -----------------------------
MODEL_PATH = "msk_triage_triage_model_v2.pkl"
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


def recommend_treatment_from_input_triage_v2(input_dict, model=model, treatment_modalities=TREATMENT_MODALITIES):
    base_row = pd.Series(input_dict)

    missing = [c for c in required_cols if c not in base_row.index]
    if missing:
        raise ValueError(f"Missing fields in input_dict: {missing}")

    simulated_rows = []
    for m in treatment_modalities:
        r = base_row.copy()
        r["treatment_modality"] = m
        simulated_rows.append(r.to_dict())

    sim_df = pd.DataFrame(simulated_rows)
    sim_df = sim_df[required_cols]

    probs = model.predict_proba(sim_df)[:, 1]

    result_df = pd.DataFrame({
        "treatment_modality": treatment_modalities,
        "predicted_success_prob": probs
    }).sort_values("predicted_success_prob", ascending=False).reset_index(drop=True)

    return result_df


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="MSK Triage Recommender", page_icon="💪", layout="centered")

st.title("💪 Soft Tissue / MSK Triage Recommender")
st.write(
    "Enter the pre-treatment details below. The model will simulate different treatment "
    "options and suggest which plan is most likely to lead to a successful outcome."
)

with st.form("triage_form"):
    subjective_injury_description = st.text_area(
        "Subjective injury description",
        value="Felt a sharp pain in the back of my thigh when sprinting. Pain is worse with fast running.",
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=16, max_value=80, value=28)
        training_hours_per_week = st.number_input("Training hours per week", min_value=0.0, max_value=30.0, value=6.0, step=0.5)
        previous_injuries_count = st.number_input("Previous injuries (total count)", min_value=0, max_value=20, value=1)
        previous_same_site_injury = st.selectbox("Previous same-site injury?", options=[0, 1])

        time_since_onset_days = st.number_input("Time since onset (days)", min_value=0, max_value=365, value=7)

    with col2:
        pain_at_rest = st.number_input("Pain at rest (0–10)", min_value=0, max_value=10, value=3)
        pain_on_activity = st.number_input("Pain on activity (0–10)", min_value=0, max_value=10, value=7)
        range_of_motion_limit = st.number_input("Range of motion limit (0–10)", min_value=0, max_value=10, value=5)

        sex = st.selectbox("Sex", options=["male", "female"])
        sport = st.selectbox("Sport", options=["football", "running", "tennis", "gym", "rugby", "basketball"])
        level = st.selectbox("Level", options=["recreational", "semi_pro", "elite"])

    col3, col4 = st.columns(2)
    with col3:
        injury_site = st.selectbox("Injury site", options=["hamstring","quadriceps","calf","ankle","knee","lower_back","shoulder","achilles"])
        injury_type = st.selectbox("Injury type", options=["muscle_strain","tendonitis","ligament_sprain","overuse"])

    with col4:
        injury_severity = st.selectbox("Injury severity (1–3)", options=[1, 2, 3])
        onset = st.selectbox("Onset", options=["acute", "gradual"])
        swelling = st.selectbox("Swelling", options=["none", "mild", "moderate", "severe"])

    submitted = st.form_submit_button("Get Treatment Recommendation")

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
        # placeholder, overwritten inside recommendation function
        "treatment_modality": "physio",
    }

    with st.spinner("Calculating recommendations..."):
        df_results = recommend_treatment_from_input_triage_v2(input_dict)

    st.subheader("Recommended treatment options")
    df_display = df_results.copy()
    df_display["predicted_success_prob"] = (df_display["predicted_success_prob"] * 100).round(1)
    df_display.rename(columns={"predicted_success_prob": "Predicted success (%)"}, inplace=True)
    st.table(df_display)
    best = df_display.iloc[0]
    st.success(f"Suggested plan: **{best['treatment_modality']}** "
               f"(~{best['Predicted success (%)']}% chance of success based on similar cases).")
