import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import ast
import numpy as np

# --- Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(project_root, 'data')
ARTIFACTS_DIR = os.path.join(project_root, 'artifacts')

def train_raw_materials_model():
    """
    Trains all models for the Raw Materials process.
    """
    print(f"\n--- Training Raw Materials Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_raw_materials.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Feature Engineering ---
    print("Preprocessing data for Raw Materials...")
    
    # Flatten the particle size distribution list
    psd_df = df['particle_size_distribution'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    psd_df.columns = [f'psd_{i+1}' for i in range(psd_df.shape[1])]
    
    # Add new summary features
    psd_df['psd_mean'] = psd_df.mean(axis=1)
    psd_df['psd_std_dev'] = psd_df.std(axis=1)
    
    df = pd.concat([df, psd_df], axis=1).drop('particle_size_distribution', axis=1)
    
    features = [
        'limestone_quality', 'clay_content', 'iron_ore_grade', 'moisture_content',
        'temperature', 'flow_rate', 
        'psd_1', 'psd_2', 'psd_3', 'psd_4', 'psd_5', 'psd_mean', 'psd_std_dev'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER (Diagnosis) ---
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the TARGET VALUE REGRESSOR (Action) ---
    print("Training Target Value Regressor...")
    reg_df = df[df['rec_action'] != 'maintain_parameters'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    target_regressor = None
    if not X_reg.empty:
        target_regressor = xgb.XGBRegressor(objective='reg:squarederror')
        target_regressor.fit(X_reg, y_reg)
    else:
        print("Warning: No samples for Raw Materials target regressor.")

    # --- 4. Train KPI REGRESSORS (Outcomes) ---
    print("Training KPI Regressors...")
    X_kpi = X # Use all data
    
    # Energy Regressor
    y_energy = df['energy_savings']
    energy_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    energy_regressor.fit(X_kpi, y_energy)
    
    # Quality Regressor
    y_quality = df['quality_improvement']
    quality_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    quality_regressor.fit(X_kpi, y_quality)
    
    # Sustainability Regressor
    y_sustain = df['sustainability_score']
    sustain_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    sustain_regressor.fit(X_kpi, y_sustain)

    # --- 5. Save Artifacts ---
    print("Saving Raw Materials artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'raw_materials_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'raw_materials_encoder.pkl'))
    if target_regressor:
        joblib.dump(target_regressor, os.path.join(ARTIFACTS_DIR, 'raw_materials_target_regressor.pkl'))
    
    joblib.dump(energy_regressor, os.path.join(ARTIFACTS_DIR, 'raw_materials_energy_regressor.pkl'))
    joblib.dump(quality_regressor, os.path.join(ARTIFACTS_DIR, 'raw_materials_quality_regressor.pkl'))
    joblib.dump(sustain_regressor, os.path.join(ARTIFACTS_DIR, 'raw_materials_sustainability_regressor.pkl'))
    print("✅ All Raw Materials models saved.")


def train_grinding_model():
    """
    Trains all models for the Grinding process.
    """
    print(f"\n--- Training Grinding Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_grinding.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Feature Engineering ---
    print("Preprocessing data for Grinding...")
    if 'specific_energy' not in df.columns:
         df['specific_energy'] = df['energy_consumption'] / (df['feed_rate'] + 1e-6)

    features = [
        'mill_power', 'feed_rate', 'product_fineness', 'energy_consumption',
        'temperature', 'vibration_level', 'noise_level', 'specific_energy' # The "golden feature"
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER (Diagnosis) ---
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the TARGET VALUE REGRESSOR (Action) ---
    print("Training Target Value Regressor...")
    reg_df = df[df['rec_action'] != 'maintain_parameters'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    target_regressor = None
    if not X_reg.empty:
        target_regressor = xgb.XGBRegressor(objective='reg:squarederror')
        target_regressor.fit(X_reg, y_reg)
    else:
        print("Warning: No samples for Grinding target regressor.")

    # --- 4. Train KPI REGRESSORS (Outcomes) ---
    print("Training KPI Regressors...")
    X_kpi = X
    
    y_energy = df['energy_savings']
    energy_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    energy_regressor.fit(X_kpi, y_energy)
    
    y_quality = df['quality_improvement']
    quality_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    quality_regressor.fit(X_kpi, y_quality)
    
    y_sustain = df['sustainability_score']
    sustain_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    sustain_regressor.fit(X_kpi, y_sustain)

    # --- 5. Save Artifacts ---
    print("Saving Grinding artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'grinding_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'grinding_encoder.pkl'))
    if target_regressor:
        joblib.dump(target_regressor, os.path.join(ARTIFACTS_DIR, 'grinding_target_regressor.pkl'))
    
    joblib.dump(energy_regressor, os.path.join(ARTIFACTS_DIR, 'grinding_energy_regressor.pkl'))
    joblib.dump(quality_regressor, os.path.join(ARTIFACTS_DIR, 'grinding_quality_regressor.pkl'))
    joblib.dump(sustain_regressor, os.path.join(ARTIFACTS_DIR, 'grinding_sustainability_regressor.pkl'))
    print("✅ All Grinding models saved.")


def train_clinkerization_model():
    """
    Trains all models for the Clinkerization process.
    """
    print(f"\n--- Training Clinkerization Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_clinkerization.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Feature Engineering ---
    print("Preprocessing data for Clinkerization...")
    features = [
        'kiln_temperature', 'residence_time', 'fuel_consumption', 
        'alternative_fuel_ratio', 'clinker_quality', 
        'exhaust_gas_temperature', 'oxygen_content'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER (Diagnosis) ---
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the TARGET VALUE REGRESSOR (Action) ---
    print("Training Target Value Regressor...")
    reg_df = df[df['rec_action'] != 'maintain_parameters'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    target_regressor = None
    if not X_reg.empty:
        target_regressor = xgb.XGBRegressor(objective='reg:squarederror')
        target_regressor.fit(X_reg, y_reg)
    else:
        print("Warning: No samples for Clinkerization target regressor.")

    # --- 4. Train KPI REGRESSORS (Outcomes) ---
    print("Training KPI Regressors...")
    X_kpi = X
    
    y_energy = df['energy_savings']
    energy_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    energy_regressor.fit(X_kpi, y_energy)
    
    y_quality = df['quality_improvement']
    quality_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    quality_regressor.fit(X_kpi, y_quality)
    
    y_sustain = df['sustainability_score']
    sustain_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    sustain_regressor.fit(X_kpi, y_sustain)

    # --- 5. Save Artifacts ---
    print("Saving Clinkerization artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'clinkerization_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'clinkerization_encoder.pkl'))
    if target_regressor:
        joblib.dump(target_regressor, os.path.join(ARTIFACTS_DIR, 'clinkerization_target_regressor.pkl'))
    
    joblib.dump(energy_regressor, os.path.join(ARTIFACTS_DIR, 'clinkerization_energy_regressor.pkl'))
    joblib.dump(quality_regressor, os.path.join(ARTIFACTS_DIR, 'clinkerization_quality_regressor.pkl'))
    joblib.dump(sustain_regressor, os.path.join(ARTIFACTS_DIR, 'clinkerization_sustainability_regressor.pkl'))
    print("✅ All Clinkerization models saved.")


def train_quality_model():
    """
    Trains all models for the Quality process.
    """
    print(f"\n--- Training Quality Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_quality.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Feature Engineering ---
    print("Preprocessing data for Quality...")
    
    # Add the new 'gypsum_added' feature
    if 'gypsum_added' not in df.columns:
        df['gypsum_added'] = df['setting_time'].apply(lambda x: (x - np.random.uniform(25.0, 90.0)) / 5.0)
        df['gypsum_added'] = df['gypsum_added'].clip(2.5, 5.5)

    features = [
        'compressive_strength', 'fineness', 'consistency', 
        'setting_time', 'temperature', 'humidity', 
        'gypsum_added' # The new control lever
    ]
    # Dropping 'product_type' as it's non-numeric and requires more complex encoding
    X = df[features]

    # --- 2. Train the CLASSIFIER (Diagnosis) ---
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the TARGET VALUE REGRESSOR (Action) ---
    print("Training Target Value Regressor...")
    reg_df = df[df['rec_action'] != 'maintain_parameters'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    target_regressor = None
    if not X_reg.empty:
        target_regressor = xgb.XGBRegressor(objective='reg:squarederror')
        target_regressor.fit(X_reg, y_reg)
    else:
        print("Warning: No samples for Quality target regressor.")

    # --- 4. Train KPI REGRESSORS (Outcomes) ---
    print("Training KPI Regressors...")
    X_kpi = X
    
    y_energy = df['energy_savings']
    energy_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    energy_regressor.fit(X_kpi, y_energy)
    
    y_quality = df['quality_improvement']
    quality_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    quality_regressor.fit(X_kpi, y_quality)
    
    y_sustain = df['sustainability_score']
    sustain_regressor = xgb.XGBRegressor(objective='reg:squarederror')
    sustain_regressor.fit(X_kpi, y_sustain)

    # --- 5. Save Artifacts ---
    print("Saving Quality artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'quality_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'quality_encoder.pkl'))
    if target_regressor:
        joblib.dump(target_regressor, os.path.join(ARTIFACTS_DIR, 'quality_target_regressor.pkl'))
    
    joblib.dump(energy_regressor, os.path.join(ARTIFACTS_DIR, 'quality_energy_regressor.pkl'))
    joblib.dump(quality_regressor, os.path.join(ARTIFACTS_DIR, 'quality_quality_regressor.pkl'))
    joblib.dump(sustain_regressor, os.path.join(ARTIFACTS_DIR, 'quality_sustainability_regressor.pkl'))
    print("✅ All Quality models saved.")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting model training process...")
    # Create artifacts directory if it doesn't exist
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    train_raw_materials_model()
    train_grinding_model()
    train_clinkerization_model()
    train_quality_model()
    
    print("\n🎉🎉🎉 All models have been successfully trained and saved! 🎉🎉🎉")