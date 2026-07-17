import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==========================================================
# Load Data
# ==========================================================

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("Training Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# ==========================================================
# Models
# ==========================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )
}

results = {}

best_model = None
best_score = 0

# ==========================================================
# Train Models
# ==========================================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if f1 > best_score:
        best_score = f1
        best_model = model

# ==========================================================
# Save Best Model
# ==========================================================

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/churn_model.pkl")

print("\nBest Model Saved Successfully!")

# ==========================================================
# Results Summary
# ==========================================================

print("\n==============================")
print("Model Comparison")
print("==============================")

for model_name, metrics in results.items():
    print(f"\n{model_name}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}") 