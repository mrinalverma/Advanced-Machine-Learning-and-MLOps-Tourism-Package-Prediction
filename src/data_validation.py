import os
import pandas as pd
import gdown

EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched',
    'NumberOfFollowups', 'DurationOfPitch'
]

def validate_data(file_path="data/travel.csv"):
    # 1. Create the data folder automatically if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 2. Download the CSV from your Google Drive link if it's not already in the folder
    if not os.path.exists(file_path):
        print(f"⬇️ Downloading dataset to {file_path}...")
        file_id = '1oljnFCQQhG_5hexZkXn7xNDqDfNiyCGf'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, file_path, quiet=False)
    
    # Safety check
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Failed to download. Dataset not found at {file_path}")
    
    # 3. Load Data
    df = pd.read_csv(file_path)
    print(f"\n✅ Dataset loaded successfully. Shape: {df.shape}")
    
    # 4. Check for missing columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing expected columns: {missing_cols}")
    else:
        print("✅ All expected columns are present.")
        
    print("\n--- Data Summary ---")
    print(df.info())

if __name__ == "__main__":
    validate_data()