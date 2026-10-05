import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="HealthSure | AI Insurance Estimator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 12px rgba(13, 110, 253, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Cache Model Loading
@st.cache_resource
def load_assets():
    model = joblib.load('insurance_model.pkl')
    features = joblib.load('insurance_features.pkl')
    return model, features

try:
    model, features = load_assets()
except Exception as e:
    st.error(f"⚠️ Failed to load model files: {e}")
    st.stop()

# App Header
st.title("🩺 HealthSure AI — Medical Insurance Cost Predictor")
st.caption("Machine Learning powered underwriting engine for healthcare premium estimation.")
st.divider()

# Sidebar: User Profile Inputs
with st.sidebar:
    st.header("📋 Patient Profile")
    st.write("Configure candidate health and demographic variables.")
    
    age = st.slider("Age", min_value=18, max_value=100, value=28, help="Age of the primary beneficiary")
    sex = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=230, value=172)
    with col_sb2:
        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    
    # Calculate BMI automatically
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
    st.info(f"**Calculated BMI:** `{bmi}` kg/m²")
    
    children = st.selectbox("Dependents / Children", [0, 1, 2, 3, 4, 5])
    smoker = st.selectbox("Tobacco / Smoking Habit", ["No", "Yes"])
    region = st.selectbox("US Region", ["northeast", "northwest", "southeast", "southwest"])

# Main Dashboard Layout
col_main1, col_main2 = st.columns([1.2, 1])

with col_main1:
    st.subheader("📊 Underwriting & Risk Analysis")
    
    # Risk Metrics Display
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Age Group", f"{age} yrs", delta="Base Tier" if age < 35 else "Senior Tier", delta_color="normal")
    with c2:
        bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else ("Overweight" if 25 <= bmi <= 29.9 else "Obese")
        st.metric("BMI Category", bmi_status, delta=f"{bmi} BMI")
    with c3:
        st.metric("Smoker Status", smoker, delta="High Risk" if smoker == "Yes" else "Standard", delta_color="inverse" if smoker == "Yes" else "normal")

    st.markdown("---")
    
    # Prediction Action
    if st.button("🚀 Calculate Estimated Premium", type="primary", use_container_width=True):
        # Construct feature dictionary matching model layout
        input_data = {feat: 0 for feat in features}
        
        input_data['age'] = age
        input_data['bmi'] = bmi
        input_data['children'] = children
        input_data['sex'] = 1 if sex == "Female" else 0
        input_data['smoker'] = 1 if smoker == "Yes" else 0
        
        region_key = f"region_{region}"
        if region_key in input_data:
            input_data[region_key] = 1
            
        input_df = pd.DataFrame([input_data])
        predicted_cost = model.predict(input_df)[0]
        monthly_cost = predicted_cost / 12
        
        # Display Prediction Results
        st.success(f"### Estimated Annual Premium: **${predicted_cost:,.2f}**")
        
        m1, m2 = st.columns(2)
        m1.metric("Estimated Monthly Cost", f"${monthly_cost:,.2f}")
        m2.metric("Daily Equivalent", f"${monthly_cost / 30:,.2f}")
        
        st.markdown("### 💡 Risk Drivers Breakdown")
        if smoker == "Yes":
            st.error("🚬 **Tobacco Usage:** Smoking is the highest cost multiplier in medical insurance underwriting.")
        if bmi >= 30:
            st.warning("⚠️ **BMI Alert:** A BMI of 30 or higher elevates risks for chronic conditions, increasing base charges.")
        if age >= 50:
            st.info("📈 **Age Factor:** Premiums scale non-linearly past age 50 due to increased actuarial risk.")
            
with col_main2:
    st.subheader("📌 Key Insights & Factors")
    
    st.write("""
    This machine learning model utilizes a **Random Forest Regressor** trained on healthcare underwriting datasets to evaluate risk factors and forecast annual medical charges.
    
    **Primary Cost Drivers:**
    * **Smoking Status:** Increases annual premiums by 300%–400% on average.
    * **Body Mass Index (BMI):** Synergizes with smoking status to produce high risk surcharges.
    * **Age:** Predicts baseline healthcare utilization frequency.
    * **Dependents:** Account for additional covered lives under family policies.
    """)
    
    st.divider()
    st.caption("🛡️ *Disclaimer: This application is a machine learning portfolio demonstration and does not constitute official medical insurance policy binding or financial advice.*")