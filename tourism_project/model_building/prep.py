"""
CI-safe data preparation script. Reads the already-registered, cleaned
dataset from the repo (tourism_project/data/tourism.csv) and produces
train/test splits for model training. Does not depend on Google Drive
or Colab, so it can run unattended in GitHub Actions.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"

def prepare_data():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Data loaded. Shape: {df.shape}")

    # Defensive cleaning (safe even if the CSV was already cleaned)
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])

    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed:')]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped index-artifact columns: {unnamed_cols}")

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    if 'ProdTaken' not in df.columns:
        raise ValueError("'ProdTaken' column not found in dataset.")

    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("Xtrain.csv", index=False)
    X_test.to_csv("Xtest.csv", index=False)
    y_train.to_csv("ytrain.csv", index=False)
    y_test.to_csv("ytest.csv", index=False)

    print(f"Train/test splits saved. X_train: {X_train.shape}, X_test: {X_test.shape}")

if __name__ == "__main__":
    prepare_data()
