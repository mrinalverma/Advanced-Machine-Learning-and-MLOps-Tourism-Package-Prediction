import os
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Create the artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Load the dataset directly from the repository data folder
    file_path = "data/tourism.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Dataset not found at {file_path}. Please ensure it is uploaded.")
        
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded successfully. Shape: {df.shape}")
    
    # 2. Perform data cleaning and remove unnecessary columns
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        print("Dropped 'CustomerID' (not a predictive feature).")
        
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    # Handle missing values to ensure clean train/test splits
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 3. Split the cleaned dataset into training and testing sets
    X = df.drop(columns=['ProdTaken'])
    y = df['ProdTaken']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Save them locally
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv("artifacts/train.csv", index=False)
    test_df.to_csv("artifacts/test.csv", index=False)
    print(f"✅ Train set saved to artifacts/train.csv ({train_df.shape})")
    print(f"✅ Test set saved to artifacts/test.csv ({test_df.shape})")

if __name__ == "__main__":
    prepare_data()