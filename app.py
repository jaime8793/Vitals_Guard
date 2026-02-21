import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Integrated Health Risk Dashboard", page_icon="⚕️", layout="wide")

# --- LOAD THE UNIFIED MODEL ---
@st.cache_resource
def load_unified_model():
    try:
        return joblib.load('unified_risk_model.pkl')
    except FileNotFoundError:
        st.error("🚨 'unified_risk_model.pkl' not found! Please run the new notebook code and place the file in this folder.")
        st.stop()

unified_model = load_unified_model()

st.title("⚕️ Integrated Health Risk Framework")
st.markdown("This dashboard uses a **Unified Machine Learning Model** that naturally calculates the interactions between clinical biomarkers and daily behavioral habits.")
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📝 Patient Data Inputs")

with st.sidebar.expander("🏥 Clinical Metrics (Biomarkers)", expanded=True):
    pregnancies = st.slider("Pregnancies", 0, 15, 1)
    glucose = st.slider("Glucose Level", 50, 250, 110)
    blood_pressure = st.slider("Blood Pressure", 40, 140, 70)
    skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
    insulin = st.slider("Insulin", 0, 500, 79)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    dpf = st.slider("Diabetes Pedigree", 0.0, 2.5, 0.4)
    age = st.slider("Age", 21, 100, 33)

with st.sidebar.expander("🏃 Lifestyle Metrics (Wearable)", expanded=True):
    total_steps = st.slider("Total Steps", 0, 25000, 5000)
    total_distance = st.slider("Total Distance (Miles)", 0.0, 20.0, 3.5)
    calories = st.slider("Calories Burned", 1000, 5000, 2000)
    very_active = st.slider("Very Active Min", 0, 120, 10)
    fairly_active = st.slider("Fairly Active Min", 0, 120, 15)
    lightly_active = st.slider("Lightly Active Min", 0, 300, 100)
    sedentary_minutes = st.slider("Sedentary Min", 0, 1440, 800)

# --- DATA PREPARATION ---
# Calculate engineered features
total_active = very_active + fairly_active + lightly_active
sedentary_hours = sedentary_minutes / 60

# The unified model expects exactly 17 columns in this specific order
features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', # Clinical
    'TotalSteps', 'TotalDistance', 'Calories', 'VeryActiveMinutes', 'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalActiveMinutes', 'SedentaryHours' # Lifestyle
]

X_input = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
    total_steps, total_distance, calories, very_active, fairly_active, lightly_active, sedentary_minutes, total_active, sedentary_hours
]], columns=features)

# --- THE UNIFIED ML PREDICTION ---
predicted_class = unified_model.predict(X_input)[0]
prediction_probs = unified_model.predict_proba(X_input)[0]
classes = list(unified_model.classes_)

# Extract the probability for the "Critical Risk" class specifically for our gauge
prob_critical = prediction_probs[classes.index('Critical Risk')] * 100

# Set colors based on the ML's decision
if predicted_class == "Healthy":
    bg_color, icon = "green", "✅"
    message = "Patient has excellent biomarkers and an active lifestyle. Maintain current regimen."
elif predicted_class == "Elevated (Behavioral)":
    bg_color, icon = "orange", "🟡"
    message = "Biomarkers are normal, but highly sedentary lifestyle puts patient at risk for future metabolic issues. Increase daily steps."
elif predicted_class == "Moderate (Medical)":
    bg_color, icon = "darkorange", "⚠️"
    message = "Patient has good lifestyle habits, but clinical markers show risk. Medical consultation recommended."
else: # Critical Risk
    bg_color, icon = "red", "🚨"
    message = "High clinical biomarkers AND poor lifestyle habits detected. Immediate intervention required."

# --- DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🧠 ML Integrated Diagnosis")
    st.markdown(f"""
    <div style='background-color:{bg_color}; padding:30px; border-radius:15px; color:white; text-align:center;'>
        <h1 style='margin:0; font-size: 3rem;'>{icon}</h1>
        <h2 style='margin:10px 0 0 0;'>{predicted_class}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"**Recommendation:** {message}")

with col2:
    st.subheader("📊 Critical Risk Probability")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob_critical,
        number = {'suffix': "%", 'valueformat': ".1f"},
        title = {'text': "Likelihood of Critical Health Failure"},
        gauge = {'axis': {'range': [0, 100]},
                 'bar': {'color': "darkblue"},
                 'steps': [
                     {'range': [0, 25], 'color': "lightgreen"},
                     {'range': [25, 60], 'color': "orange"},
                     {'range': [60, 100], 'color': "#ff6961"}]}))
    st.plotly_chart(fig, use_container_width=True)