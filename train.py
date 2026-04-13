import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("insurance.csv")

# Encoding
df["sex"] = df["sex"].map({"male": 1, "female": 0})
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
df["region"] = df["region"].astype("category").cat.codes

# Feature Engineering
df["bmi_category"] = pd.cut(df["bmi"], bins=[0,18.5,25,30,100], labels=[0,1,2,3]).astype(int)
df["risk_score"] = df["age"]*0.2 + df["bmi"]*0.3 + df["smoker"]*5
df["bmi_smoker"] = df["bmi"] * df["smoker"]

X = df[["age","bmi","children","sex","smoker","region","bmi_category","risk_score","bmi_smoker"]]
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = XGBRegressor(n_estimators=150)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model trained using REAL dataset!")
