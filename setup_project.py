import os

# Define required directories
directories = [
    "data",
    "src",
    "artifacts",
    ".github/workflows"
]

# Define initial content for placeholder files
files = {
    "src/__init__.py": "# Initialize src package",
    "src/data_validation.py": "# Data validation script placeholder",
    "src/data_preparation.py": "# Data preparation script placeholder",
    "src/model_training.py": "# Model training script placeholder",
    "app.py": "# Streamlit app placeholder",
    "requirements.txt": "pandas\nnumpy\nscikit-learn\nxgboost\njoblib\nstreamlit",
    ".github/workflows/pipeline.yml": "# GitHub Actions pipeline YAML placeholder",
    "README.md": "# Visit with Us - MLOps Project\n"
}

def create_project_structure():
    print("🚀 Setting up project structure for 'Visit with Us' MLOps...\n")
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")
        
    # Create files
    for file_path, content in files.items():
        parent_dir = os.path.dirname(file_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 Created file:      {file_path}")
        else:
            print(f"⚠️  File exists, skipping: {file_path}")
            
    print("\n✅ Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()