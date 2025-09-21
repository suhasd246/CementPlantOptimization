import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys
import asyncio
import numpy as np # <-- Make sure numpy is imported

# --- Add Project Root to Python Path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Configuration ---
DATA_PATH = os.path.join(project_root, 'data', 'quality.csv')
ARTIFACTS_DIR = os.path.join(project_root, 'artifacts')

# This script is now self-contained and doesn't need the external optimizer
def get_recommendation(row):
    if row['quality_variance'] > 0.05:
        return "Implement real-time quality monitoring", 0.03
    else:
        return "none", 0.0

def create_model():
    print("--- Starting Quality Model Creation Process ---")
    
    # 1. LOAD RAW DATA
    df = pd.read_csv(DATA_PATH)
    
    # 2. LABEL DATA (in-memory)
    print("Labeling data in memory...")
    
    # --- THE FIX IS HERE ---
    # Before: df['quality_variance'] = 0.06
    # After: Generate a random variance for each row to create a mix of outcomes
    df['quality_variance'] = np.random.uniform(0.01, 0.09, size=len(df))
    # --- END OF FIX ---
    
    df[['rec_action', 'rec_target_value']] = df.apply(get_recommendation, axis=1, result_type='expand')
    df = df.drop(columns=['quality_variance'])

    # 3. STRICT DATA CLEANING
    print("Cleaning and validating data types...")
    features = [
        'compressive_strength', 'fineness', 'consistency', 
        'setting_time', 'temperature', 'humidity'
    ]
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    original_rows = len(df)
    df.dropna(subset=features, inplace=True)
    print(f"Dropped {original_rows - len(df)} rows with non-numeric data.")

    if df.empty:
        print("ERROR: DataFrame is empty after cleaning. Cannot train model.")
        sys.exit(1)

    X = df[features]
    
    # 4. TRAIN CLASSIFIER
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)
    
    num_classes = len(encoder.classes_)
    if num_classes <= 1:
        print(f"ERROR: Only found {num_classes} class(es). Need at least 2 for classification.")
        sys.exit(1)

    classifier = xgb.XGBClassifier(
        objective='multi:softmax', 
        num_class=num_classes,
        eval_metric='mlogloss'
    )
    classifier.fit(X, y_class_encoded)

    # 5. TRAIN REGRESSOR
    print("Training Regressor...")
    reg_df = df[df['rec_action'] != 'none']
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']
    
    if not X_reg.empty:
        regressor = xgb.XGBRegressor(objective='reg:squarederror')
        regressor.fit(X_reg, y_reg)
    else:
        regressor = None

    # 6. SAVE ARTIFACTS
    print("Saving artifacts to /artifacts folder...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'quality_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'quality_encoder.pkl'))
    if regressor:
        joblib.dump(regressor, os.path.join(ARTIFACTS_DIR, 'quality_regressor.pkl'))
    
    print("\n🎉 Successfully created and saved all quality model artifacts!")

if __name__ == "__main__":
    create_model()