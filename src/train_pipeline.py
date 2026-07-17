import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Remove customerID
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Features
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================================
# Column Lists
# ==========================================================

numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    c for c in X.columns
    if c not in numeric_features
]

# ==========================================================
# Numeric Pipeline
# ==========================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ==========================================================
# Categorical Pipeline
# ==========================================================

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ==========================================================
# Preprocessor
# ==========================================================

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==========================================================
# Full Pipeline
# ==========================================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ==========================================================
# Train
# ==========================================================

pipeline.fit(X_train, y_train)

# ==========================================================
# Predict
# ==========================================================

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ==========================================================
# Save Pipeline
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    "models/churn_pipeline.pkl"
)

print("\nPipeline saved successfully!") 