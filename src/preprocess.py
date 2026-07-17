import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

# ==========================================================
# Remove customerID
# ==========================================================

df.drop("customerID", axis=1, inplace=True)

# ==========================================================
# Convert TotalCharges to numeric
# ==========================================================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check missing values before filling
print("\nMissing values before filling:")
print(df.isnull().sum())

# Fill missing values in TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# Verify all missing values are removed
print("\nMissing values after filling:")
print(df.isnull().sum())  

# ==========================================================
# Encode Target Variable
# ==========================================================

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ==========================================================
# Encode Categorical Columns
# ==========================================================

label_encoders = {}

categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save encoders

os.makedirs("models", exist_ok=True)
joblib.dump(label_encoders, "models/label_encoders.pkl")

# ==========================================================
# Feature / Target Split
# ==========================================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# Feature Scaling
# ==========================================================

scaler = StandardScaler()

numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X_train[numerical_cols] = scaler.fit_transform(
    X_train[numerical_cols]
)

X_test[numerical_cols] = scaler.transform(
    X_test[numerical_cols]
)

# Save scaler

joblib.dump(scaler, "models/scaler.pkl")

# ==========================================================
# Save Processed Data
# ==========================================================

os.makedirs("data/processed", exist_ok=True)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nPreprocessing Completed Successfully!")

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
