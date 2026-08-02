import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_and_tune_model(data_dir="artifacts", model_output="artifacts/model.pkl"):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['ProdTaken'])
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(columns=['ProdTaken'])
    y_test = test_df['ProdTaken']
    
    # Define categorical and numerical columns
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=['object']).columns.tolist()

    print("⚙️ Building Preprocessing Pipeline...")
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    # Create the master pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Define parameters for tuning
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 15, None],
        'classifier__min_samples_split': [2, 5]
    }
    
    print("🔍 Tuning model parameters using GridSearchCV...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Log tuned parameters
    best_model = grid_search.best_estimator_
    print("\n✅ Tuning Complete. Logging Best Parameters:")
    for param_name, param_value in grid_search.best_params_.items():
        print(f"  - {param_name}: {param_value}")
        
    print(f"  - Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    print("\n📊 Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save best model
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(best_model, model_output)
    print(f"💾 Best model committed to {model_output}")

if __name__ == "__main__":
    train_and_tune_model()