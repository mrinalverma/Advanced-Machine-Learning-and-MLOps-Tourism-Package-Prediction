import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(input_path="data/travel.csv", output_dir="artifacts"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    
    print("🧹 Cleaning data...")
    # Drop CustomerID as it's not a predictive feature
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    # Standardize Gender categories (sometimes 'Fe Male' appears in this dataset)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    # Split features and target
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    
    # Split into train and test sets (Stratified to maintain class balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Save train and test splits to artifacts folder
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Data Preparation complete.")
    print(f"Train set saved to {train_path} ({train_df.shape})")
    print(f"Test set saved to {test_path} ({test_df.shape})")

if __name__ == "__main__":
    prepare_data()