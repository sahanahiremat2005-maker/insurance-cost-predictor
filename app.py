import streamlit as st
import joblib
import os

st.set_page_config(page_title="Insurance Predictor", page_icon="💰")

# UI
st.markdown("## 💰 Insurance Cost Prediction")
st.markdown("---")

# If model missing → train
if not os.path.exists("model.pkl"):
    import train

model = joblib.load("model.pkl")

# Inputs
age = st.slider("Age", 18, 65, 30)
bmi = st.slider("BMI", 15.0, 40.0, 25.0)
children = st.slider("Children", 0, 5, 1)

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast","northwest","southeast","southwest"])

# Encoding
sex_val = 1 if sex == "male" else 0
smoker_val = 1 if smoker == "yes" else 0
region_map = {"northeast":0,"northwest":1,"southeast":2,"southwest":3}
region_val = region_map[region]

# Features
bmi_category = 0 if bmi<18.5 else 1 if bmi<25 else 2 if bmi<30 else 3
risk_score = age*0.2 + bmi*0.3 + smoker_val*5
bmi_smoker = bmi * smoker_val

if st.button("💸 Predict Cost"):
    pred = model.predict([[age,bmi,children,sex_val,smoker_val,region_val,
                           bmi_category,risk_score,bmi_smoker]])[0]

    # Risk
    risk = "Low" if pred<10000 else "Medium" if pred<20000 else "High"

    st.success(f"💰 Cost: ₹{pred:,.2f}")
    st.info(f"⚠️ Risk: {risk}")
