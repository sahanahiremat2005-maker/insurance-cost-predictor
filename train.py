import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("data/insurance.csv")

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Columns
num_cols = ["age", "bmi", "children"]
cat_cols = ["sex", "smoker", "region"]

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
}

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_score = -1
best_model = None
best_name = ""

# Train and compare
for name, algo in models.items():
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", algo)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    score = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"{name}")
    print(f"R2 Score: {score:.4f}")
    print(f"MAE: {mae:.2f}")
    print("-" * 30)

    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name = name

# Save best model
joblib.dump(best_model, "models/model.pkl")

print(f"Best Model: {best_name}")
print("Model saved to models/model.pkl")
