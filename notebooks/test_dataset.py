import pandas as pd

df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

print(df.dtypes)