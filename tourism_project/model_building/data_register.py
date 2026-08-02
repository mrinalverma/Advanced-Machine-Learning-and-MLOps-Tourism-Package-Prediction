"""
Registers (validates) the tourism dataset already committed to the repo.
Reads tourism_project/data/tourism.csv, checks the expected columns are
present, and prints a short summary. The dataset lives inside the GitHub
repo, so no external dataset store is needed.
"""
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

def register_dataset():
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape}")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Registered dataset is missing expected columns: {missing}")

    print("All expected columns present.")
    print("\nColumn dtypes:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())

if __name__ == "__main__":
    register_dataset()
