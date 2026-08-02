import os
import pandas as pd

EXPECTED_COLUMNS = [
    'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier', 'Occupation',
    'Gender', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'MaritalStatus',
    'NumberOfTrips', 'Passport', 'OwnCar', 'NumberOfChildrenVisiting',
    'Designation', 'MonthlyIncome', 'PitchSatisfactionScore', 'ProductPitched',
    'NumberOfFollowups', 'DurationOfPitch'
]

def validate_data(file_path="data/tourism.csv"):
    # Check if file exists in the repo
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Dataset not found at {file_path}. Please ensure tourism.csv is in the data folder.")
    
    # Load Data
    df = pd.read_csv(file_path)
    print(f"\n✅ Dataset loaded successfully. Shape: {df.shape}")
    
    # Check for missing columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing expected columns: {missing_cols}")
    else:
        print("✅ All expected columns are present.")
        
    print("\n--- Data Summary ---")
    print(df.info())

if __name__ == "__main__":
    validate_data()