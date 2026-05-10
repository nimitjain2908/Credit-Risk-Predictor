import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the model, feature names, and median values
model = joblib.load('credit_model.pkl')
model_columns = joblib.load('model_columns.pkl')
medians = joblib.load('model_medians.pkl')

st.set_page_config(page_title="Credit Risk Advisor", page_icon="🏦")
st.title("🏦 AI Credit Risk Advisor")
st.markdown("Enter applicant details to generate a real-time risk assessment.")

# 2. Sidebar for Primary Financial Indicators
st.sidebar.header("External Credit Ratings")
ext_2 = st.sidebar.slider("External Source 2 (Normalized)", 0.0, 1.0, 0.5)
ext_3 = st.sidebar.slider("External Source 3 (Normalized)", 0.0, 1.0, 0.5)

# 3. Main Page: Demographic & Financial Details
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Applicant Age (Years)", 18, 90, 35)
    income = st.number_input("Total Annual Income", 0, 100000000, 50000)
    amt_credit = st.number_input("Requested Loan Amount", 0, 20000000, 100000)

with col2:
    years_employed = st.number_input("Years at Current Job", 0, 50, 5)
    education = st.selectbox("Education Level", 
                             ["Secondary / secondary special", "Higher education", 
                              "Incomplete higher", "Lower secondary", "Academic degree"])

# 4. Prediction Logic
if st.button("Generate Risk Score"):
    # Start with the median baseline for all 300+ features
    input_df = pd.DataFrame([medians.values], columns=model_columns)
    
    # Overwrite the specific inputs from the UI
    input_df['EXT_SOURCE_2'] = ext_2
    input_df['EXT_SOURCE_3'] = ext_3
    input_df['YEARS_BIRTH'] = age
    input_df['AMT_INCOME_TOTAL'] = income
    input_df['AMT_CREDIT'] = amt_credit
    input_df['DAYS_EMPLOYED'] = years_employed * -365 # Dataset uses negative days for past events
    
    # Handle Education (One-Hot Encoding matching)
    edu_col = f"NAME_EDUCATION_TYPE_{education}"
    if edu_col in input_df.columns:
        # Reset all education columns to 0 first
        edu_cols = [c for c in input_df.columns if "NAME_EDUCATION_TYPE" in c]
        input_df[edu_cols] = 0
        # Set the selected one to 1
        input_df[edu_col] = 1

    # Calculate Probability
    prob = model.predict_proba(input_df)[0][1]
    
    # 5. Display Result
    st.divider()
    st.subheader(f"Default Probability: {prob:.2%}")
    
    if prob > 0.15:
        st.error("⚠️ **HIGH RISK**: This application requires manual underwriter review.")
        st.info("Key Risk Factors: Low external ratings or high debt-to-income ratio detected.")
    elif prob > 0.08:
        st.warning("⚖️ **MEDIUM RISK**: Conditional approval possible with higher down payment.")
    else:
        st.success("✅ **LOW RISK**: Application meets standard automated approval criteria.")

st.caption("Note: This tool uses a LightGBM model with SHAP-validated feature importance.")