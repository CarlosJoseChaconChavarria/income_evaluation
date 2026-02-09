import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. LOAD THE CHAMPIONS
model = joblib.load('income_model.pkl')
scaler = joblib.load('income_scaler.pkl')

# --- TRANSLATION DICTIONARIES (The "Secret Decoder Ring") ---
# These match exactly how your LabelEncoder converted the words during training.
workclass_dict = {"Federal-gov": 1, "Local-gov": 2, "Never-worked": 3, "Private": 4, "Self-emp-inc": 5, "Self-emp-not-inc": 6, "State-gov": 7, "Without-pay": 8, "Unknown (?)": 0}
edu_dict = {"Bachelors": 9, "Some-college": 15, "11th": 1, "HS-grad": 11, "Prof-school": 14, "Assoc-acdm": 7, "Assoc-voc": 8, "9th": 6, "7th-8th": 5, "12th": 2, "Masters": 12, "1st-4th": 3, "10th": 0, "Doctorate": 10, "5th-6th": 4, "Preschool": 13}
marital_dict = {"Divorced": 0, "Married-civ-spouse": 2, "Never-married": 4, "Separated": 5, "Widowed": 6, "Married-spouse-absent": 3, "Married-AF-spouse": 1}
occ_dict = {"Tech-support": 13, "Craft-repair": 3, "Other-service": 8, "Sales": 12, "Exec-managerial": 4, "Prof-specialty": 10, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 1, "Farming-fishing": 5, "Transport-moving": 14, "Priv-house-serv": 9, "Protective-serv": 11, "Armed-Forces": 2, "Unknown (?)": 0}
rel_dict = {"Wife": 5, "Own-child": 3, "Husband": 0, "Not-in-family": 1, "Other-relative": 2, "Unmarried": 4}
race_dict = {"White": 4, "Asian-Pac-Islander": 1, "Amer-Indian-Eskimo": 0, "Other": 3, "Black": 2}
country_dict = {"United-States": 39, "Mexico": 26, "Philippines": 30, "Germany": 11, "Canada": 2, "Puerto-Rico": 33, "El-Salvador": 8, "India": 19, "Cuba": 5, "England": 9, "Jamaica": 23, "South": 35, "China": 3, "Italy": 22, "Dominican-Republic": 6, "Vietnam": 40, "Guatemala": 13, "Japan": 24, "Poland": 31, "Columbia": 4, "Taiwan": 36, "Haiti": 14, "Iran": 20, "Portugal": 32, "Nicaragua": 27, "Peru": 29, "Greece": 12, "France": 10, "Ecuador": 7, "Ireland": 21, "Hong": 17, "Cambodia": 1, "Trinadad&Tobago": 38, "Laos": 25, "Thailand": 37, "Yugoslavia": 41, "Outlying-US(Guam-USVI-etc)": 28, "Hungary": 18, "Honduras": 16, "Scotland": 34, "Holand-Netherlands": 15, "Unknown (?)": 0}

st.set_page_config(page_title="Income Predictor", page_icon="💰")

st.title("💰 High-Income Predictor")
st.write("Determine the likelihood of an individual earning over **$50,000 USD** per year based on Census data.")

# 2. CREATE THE INPUT FIELDS (The UI)
st.subheader("Personal & Professional Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 17, 90, 30)
    # Users see words, the variable stores the numeric ID
    workclass = workclass_dict[st.selectbox("Workclass", options=list(workclass_dict.keys()))]
    fnlwgt = st.number_input("Final Weight (Census Representation)", value=180000)
    education = edu_dict[st.selectbox("Highest Degree", options=list(edu_dict.keys()))]
    edu_num = st.slider("Years of Schooling", 1, 16, 10)

with col2:
    marital = marital_dict[st.selectbox("Marital Status", options=list(marital_dict.keys()))]
    occupation = occ_dict[st.selectbox("Occupation", options=list(occ_dict.keys()))]
    relationship = rel_dict[st.selectbox("Relationship Role", options=list(rel_dict.keys()))]
    race = race_dict[st.selectbox("Race", options=list(race_dict.keys()))]
    sex = st.radio("Sex", ["Female", "Male"])
    sex_val = 1 if sex == "Male" else 0

st.subheader("Financials & Location")
c1, c2, c3 = st.columns(3)
with c1:
    cap_gain = st.number_input("Capital Gain ($)", value=0)
with c2:
    cap_loss = st.number_input("Capital Loss ($)", value=0)
with c3:
    hours = st.slider("Hours Worked/Week", 1, 100, 40)

country = country_dict[st.selectbox("Country of Origin", options=list(country_dict.keys()))]

# 3. THE PREDICTION LOGIC
if st.button("Analyze Profile"):
    # A. The "Shrink Ray" (Log Transform)
    cap_gain_log = np.log1p(cap_gain)
    cap_loss_log = np.log1p(cap_loss)

    # B. Construct the feature array in the EXACT order the model was trained
    # [age, workclass, fnlwgt, education, edu_num, marital, occupation, relationship, race, sex, gain, loss, hours, country]
    user_data = np.array([[age, workclass, fnlwgt, education, edu_num, marital, 
                           occupation, relationship, race, sex_val, 
                           cap_gain_log, cap_loss_log, hours, country]])

    # C. The "Uniform Maker" (Scale the data)
    user_scaled = scaler.transform(user_data)

    # D. Predict
    prediction = model.predict(user_scaled)
    probability = model.predict_proba(user_scaled)[0][1] # Probability of being >50K

    # E. Display Results
    st.markdown("---")
    if prediction[0] == 1:
        st.success(f"### Prediction: >$50K (High Earner) 🚀")
        st.write(f"Confidence Score: {probability:.2%}")
    else:
        st.info(f"### Prediction: <=$50K (Average Earner) 🏠")
        st.write(f"Confidence Score: {(1-probability):.2%}")