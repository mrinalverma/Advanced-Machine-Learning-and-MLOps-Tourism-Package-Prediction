import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

def train_and_tune_model():
    print("--- Model Building with Experimentation Tracking (XGBoost) ---")

    # Define paths for the split data files
    X_train_path = "Xtrain.csv"
    y_train_path = "ytrain.csv"
    X_test_path = "Xtest.csv"
    y_test_path = "ytest.csv"

    # 1. Load the train and test data
    if not os.path.exists(X_train_path) or not os.path.exists(y_train_path) or \
       not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Train or test data not found. Ensure data_preparation.py saved the splits.")

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).squeeze() # .squeeze() to convert single-column DataFrame to Series
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    print(f"Data splits loaded successfully. X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Identify numerical and categorical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    # Preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])

    # 2. Define a model (XGBoost) and parameters
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [5, 7, 10],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }

    # 3. Tune the model with the defined parameters
    print("Tuning XGBoost parameters using GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # 4. Log all the tuned parameters
    print("\nTuning Complete! Logged Best Parameters:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  - {param_name}: {param_value}")

    # 5. Evaluate the model performance
    y_pred = best_model.predict(X_test)
    print("\nModel Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save the best model so the pipeline can commit it
    os.makedirs("artifacts", exist_ok=True) # Ensure artifacts directory exists
    model_output = "artifacts/model.pkl"
    joblib.dump(best_model, model_output)
    print(f"\nBest model saved to {model_output}")

if __name__ == "__main__":
    train_and_tune_model()
