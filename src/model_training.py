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
    
    # 1. Load the train and test data from the workflow artifact
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train or test data not found. Run data_preparation.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']

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
    print("🔍 Tuning XGBoost parameters using GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_

    # 4. Log all the tuned parameters
    print("\n✅ Tuning Complete! Logged Best Parameters:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  - {param_name}: {param_value}")
    
    # 5. Evaluate the model performance
    y_pred = best_model.predict(X_test)
    print("\n📊 Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Save the best model so the pipeline can commit it
    model_output = "artifacts/model.pkl"
    joblib.dump(best_model, model_output)
    print(f"\n💾 Best model saved to {model_output}")

if __name__ == "__main__":
    train_and_tune_model()