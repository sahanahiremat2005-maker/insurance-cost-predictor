import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="💰",
    layout="wide"
)

# Load model
model = joblib.load("models/model.pkl")

# Custom style
st.markdown("""
<style>
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background: #eafaf1;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Insurance Cost Prediction")
st.caption("Predict insurance charges instantly using Machine Learning")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 30)
    bmi = st.slider("BMI", 15.0, 50.0, 25.0)
    children = st.selectbox("Children", [0,1,2,3,4,5])

with col2:
    sex = st.selectbox("Gender", ["male", "female"])
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

# Predict button
if st.button("Predict Insurance Cost"):
    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    prediction = model.predict(data)[0]

    st.markdown(
        f'<div class="result-box">Estimated Cost: ${prediction:,.2f}</div>',
        unsafe_allow_html=True
    )
