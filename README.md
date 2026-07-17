# 🚀 Cloud-Based Customer Churn Prediction Pipeline

An end-to-end Machine Learning project that predicts customer churn using a production-ready Scikit-learn Pipeline, FastAPI REST API, Streamlit dashboard, Docker, and cloud deployment.

---

## 📌 Project Overview

Customer churn is one of the biggest challenges for subscription-based businesses. This project predicts whether a customer is likely to leave the company based on demographic information, account details, and subscribed services.

The project demonstrates an end-to-end ML workflow:

- Data preprocessing
- Feature engineering
- Machine Learning Pipeline
- Model training & evaluation
- REST API development
- Interactive web application
- Docker containerization
- Cloud deployment

---

## 🏗️ Project Architecture

```text
                 Telco Customer Dataset
                           │
                           ▼
                 Data Preprocessing
                           │
                           ▼
              Scikit-learn Pipeline
   (OneHotEncoder + StandardScaler +
      Logistic Regression Classifier)
                           │
                           ▼
               churn_pipeline.pkl
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      FastAPI REST API           Streamlit Dashboard
             │                           │
             └─────────────┬─────────────┘
                           ▼
                      End Users
```

---

# ✨ Features

- Production-ready Scikit-learn Pipeline
- Automatic preprocessing during inference
- Real-time churn prediction
- Probability score output
- FastAPI REST API
- Interactive Streamlit dashboard
- Docker support
- Ready for cloud deployment
- Swagger API documentation

---

# 🛠️ Tech Stack

### Programming

- Python 3.11

### Machine Learning

- Scikit-learn
- Logistic Regression
- ColumnTransformer
- Pipeline

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- Streamlit

### Deployment

- Docker
- Git
- GitHub
- Render

---

# 📂 Project Structure

```text
Customer-Churn-Pipeline
│
├── app
│   ├── main.py
│   ├── predictor.py
│   └── schemas.py
│
├── data
│   └── raw
│       └── Telco-Customer-Churn.csv
│
├── models
│   └── churn_pipeline.pkl
│
├── src
│   └── train_pipeline.py
│
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 📊 Dataset

Dataset:

**IBM Telco Customer Churn Dataset**

Features include:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges
- Payment Method
- Churn

Target:

```
Churn
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<nirUpma52>/Customer-Churn-Pipeline.git

cd Customer-Churn-Pipeline
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run FastAPI

```bash
uvicorn app.main:app --reload
```

Visit

```
http://127.0.0.1:8000/docs
```

---

# ▶️ Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

# 🐳 Docker

Build image

```bash
docker build -t churn-pipeline .
```

Run container

```bash
docker run -d -p 8000:8000 --name churn-api churn-pipeline
```

Swagger

```
http://localhost:8000/docs
```

---

# 🔥 API Example

POST

```
/predict
```

Request

```json
{
  "gender":"Female",
  "SeniorCitizen":0,
  "Partner":"Yes",
  "Dependents":"No",
  "tenure":12,
  "PhoneService":"Yes",
  "MultipleLines":"No",
  "InternetService":"Fiber optic",
  "OnlineSecurity":"No",
  "OnlineBackup":"Yes",
  "DeviceProtection":"No",
  "TechSupport":"No",
  "StreamingTV":"Yes",
  "StreamingMovies":"Yes",
  "Contract":"Month-to-month",
  "PaperlessBilling":"Yes",
  "PaymentMethod":"Electronic check",
  "MonthlyCharges":79.85,
  "TotalCharges":958.2
}
```

Response

```json
{
    "prediction":"No",
    "probability":0.184
}
```

---

# 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 79.99% | 64.47% | 54.81% | 59.25% |
| Random Forest | 78.78% | 62.98% | 48.66% | 54.90% |
| XGBoost | 77.86% | 59.57% | 51.60% | 55.30% |

**Selected Model:** Logistic Regression

---

# 🚀 Future Improvements

- Hyperparameter tuning
- SHAP Explainability
- MLflow experiment tracking
- CI/CD with GitHub Actions
- Automated model retraining
- Kubernetes deployment
- Monitoring & logging

---

# 📸 Screenshots



### FastAPI Swagger


images/swagger.png


### Streamlit Dashboard

```
images/streamlit.png
```

---

# 👤 Author

**Nirupma Kumari**

GitHub:
https://github.com/<nirUpma52>

LinkedIn:
https://linkedin.com/in/nirupma-kumari

---

# 📄 License

This project is licensed under the MIT License.
