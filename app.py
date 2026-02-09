import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. LOAD THE CHAMPIONS
# We wake up the brain and the scaler we saved in Step 7
model = joblib.load('income_model.pkl')
scaler = joblib.load('income_scaler.pkl')

st.title("💰 Income Prediction App")
st.write("Enter the details below to see if this profile earns more or less than $50k/year.")

# 2. CREATE THE INPUT FIELDS (The UI)
# We need to collect all 14 features in the correct order
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=30)
    workclass = st.selectbox("Workclass", [0,1,2,3,4,5,6,7,8]) # Numeric IDs from LabelEncoder
    fnlwgt = st.number_input("Final Weight (fnlwgt)", value=180000)
    education = st.selectbox("Education Level", [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    edu_num = st.slider("Years of Education", 1, 16, 10)

with col2:
    marital = st.selectbox("Marital Status", [0,1,2,3,4,5,6])
    occupation = st.selectbox("Occupation", [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    relationship = st.selectbox("Relationship", [0,1,2,3,4,5])
    race = st.selectbox("Race", [0,1,2,3,4])
    sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])

cap_gain = st.number_input("Capital Gain", value=0)
cap_loss = st.number_input("Capital Loss", value=0)
hours = st.slider("Hours per Week", 1, 100, 40)
country = st.selectbox("Native Country ID", [0,1,2,3,4,5,6,7,8,9,10]) # Simplified for demo

# 3. THE "TRANSFORMATION" BUTTON
if st.button("Predict Income"):
    # A. Apply the "Shrink Ray" (Log Transform) to Capital Gain/Loss
    cap_gain_log = np.log1p(cap_gain)
    cap_loss_log = np.log1p(cap_loss)

    # B. Arrange features in the exact order the model expects
    features = np.array([[age, workclass, fnlwgt, education, edu_num, marital, 
                          occupation, relationship, race, sex, 
                          cap_gain_log, cap_loss_log, hours, country]])

    # C. Apply the "Uniform Maker" (Scaling)
    features_scaled = scaler.transform(features)

    # D. THE PREDICTION
    prediction = model.predict(features_scaled)
    
    # E. SHOW THE RESULT
    if prediction[0] == 1:
        st.success("Target Result: >50K (High Earner) 🚀")
    else:
        st.info("Target Result: <=50K (Average Earner) 🏠")

st.write("---")
st.caption("Note: For this demo, categorical inputs use ID numbers. In a production app, we would map these to friendly names!")