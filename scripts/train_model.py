import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import ast

# --- Configuration ---
# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(project_root, 'data')
ARTIFACTS_DIR = os.path.join(project_root, 'artifacts')


def train_raw_materials_model():
    """
    Loads labeled data, trains an XGBoost classifier AND a regressor, 
    and saves the artifacts.
    """
    print(f"\n--- Training Raw Materials Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_raw_materials.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Feature Engineering & Preprocessing ---
    print("Preprocessing data for Raw Materials...")
    psd_df = df['particle_size_distribution'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    psd_df.columns = [f'psd_{i+1}' for i in range(psd_df.shape[1])]
    df = pd.concat([df, psd_df], axis=1).drop('particle_size_distribution', axis=1)
    
    features = [
        'limestone_quality', 'clay_content', 'iron_ore_grade', 'moisture_content',
        'temperature', 'flow_rate', 'psd_1', 'psd_2', 'psd_3', 'psd_4', 'psd_5'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER for the 'action' ---
    print("Training Classifier...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the REGRESSOR for the 'target_value' ---
    print("Training Regressor...")
    # We only train the regressor on data where a recommendation and a target value exist
    reg_df = df[df['rec_action'] != 'none'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    if not X_reg.empty:
        regressor = xgb.XGBRegressor(objective='reg:squarederror')
        regressor.fit(X_reg, y_reg)
    else:
        regressor = None
        print("Warning: No samples available for regressor training. Skipping.")

    # --- 4. Save Artifacts ---
    print("Saving artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'raw_materials_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'raw_materials_encoder.pkl'))
    if regressor:
        joblib.dump(regressor, os.path.join(ARTIFACTS_DIR, 'raw_materials_regressor.pkl'))
        print("✅ Classifier, Encoder, and Regressor saved for Raw Materials.")


def train_grinding_model():
    """
    Loads labeled grinding data, trains an XGBoost classifier and regressor,
    and saves the artifacts.
    """
    print(f"\n--- Training Grinding Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_grinding.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Preprocessing ---
    print("Preprocessing data for Grinding...")
    features = [
        'mill_power', 'feed_rate', 'product_fineness', 'energy_consumption',
        'temperature', 'vibration_level', 'noise_level'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER for the 'action' ---
    print("Training Classifier for Grinding...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the REGRESSOR for the 'target_value' ---
    print("Training Regressor for Grinding...")
    reg_df = df[df['rec_action'] != 'none'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    if not X_reg.empty:
        regressor = xgb.XGBRegressor(objective='reg:squarederror')
        regressor.fit(X_reg, y_reg)
    else:
        regressor = None
        print("Warning: No samples available for grinding regressor. Skipping.")

    # --- 4. Save Artifacts ---
    print("Saving grinding artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'grinding_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'grinding_encoder.pkl'))
    if regressor:
        joblib.dump(regressor, os.path.join(ARTIFACTS_DIR, 'grinding_regressor.pkl'))
        print("✅ Classifier, Encoder, and Regressor saved for Grinding.")


def train_clinkerization_model():
    """
    Loads labeled clinkerization data, trains an XGBoost classifier and regressor,
    and saves the artifacts.
    """
    print(f"\n--- Training Clinkerization Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_clinkerization.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Preprocessing ---
    print("Preprocessing data for Clinkerization...")
    features = [
        'kiln_temperature', 'residence_time', 'fuel_consumption', 
        'alternative_fuel_ratio', 'clinker_quality', 
        'exhaust_gas_temperature', 'oxygen_content'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER for the 'action' ---
    print("Training Classifier for Clinkerization...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the REGRESSOR for the 'target_value' ---
    print("Training Regressor for Clinkerization...")
    reg_df = df[df['rec_action'] != 'none'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    if not X_reg.empty:
        regressor = xgb.XGBRegressor(objective='reg:squarederror')
        regressor.fit(X_reg, y_reg)
    else:
        regressor = None
        print("Warning: No samples available for clinkerization regressor. Skipping.")

    # --- 4. Save Artifacts ---
    print("Saving clinkerization artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'clinkerization_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'clinkerization_encoder.pkl'))
    if regressor:
        joblib.dump(regressor, os.path.join(ARTIFACTS_DIR, 'clinkerization_regressor.pkl'))
        print("✅ Classifier, Encoder, and Regressor saved for Clinkerization.")

def train_quality_model():
    """
    Loads labeled quality data, trains an XGBoost classifier and regressor,
    and saves the artifacts.
    """
    print(f"\n--- Training Quality Models ---")
    labeled_data_path = os.path.join(DATA_PATH, 'labeled_quality.csv')
    df = pd.read_csv(labeled_data_path)

    # --- 1. Preprocessing ---
    print("Preprocessing data for Quality...")
    # NOTE: 'product_type' is a categorical feature. A more advanced model would
    # one-hot encode it. For simplicity, we are dropping it.
    features = [
        'compressive_strength', 'fineness', 'consistency', 
        'setting_time', 'temperature', 'humidity'
    ]
    X = df[features]

    # --- 2. Train the CLASSIFIER for the 'action' ---
    print("Training Classifier for Quality...")
    y_class = df['rec_action']
    encoder = LabelEncoder()
    y_class_encoded = encoder.fit_transform(y_class)

    classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    classifier.fit(X, y_class_encoded)
    
    # --- 3. Train the REGRESSOR for the 'target_value' ---
    print("Training Regressor for Quality...")
    reg_df = df[df['rec_action'] != 'none'].copy()
    X_reg = reg_df[features]
    y_reg = reg_df['rec_target_value']

    if not X_reg.empty:
        regressor = xgb.XGBRegressor(objective='reg:squarederror')
        regressor.fit(X_reg, y_reg)
    else:
        regressor = None
        print("Warning: No samples available for quality regressor. Skipping.")

    # --- 4. Save Artifacts ---
    print("Saving quality artifacts...")
    joblib.dump(classifier, os.path.join(ARTIFACTS_DIR, 'quality_model.pkl'))
    joblib.dump(encoder, os.path.join(ARTIFACTS_DIR, 'quality_encoder.pkl'))
    if regressor:
        joblib.dump(regressor, os.path.join(ARTIFACTS_DIR, 'quality_regressor.pkl'))
        print("✅ Classifier, Encoder, and Regressor saved for Quality.")


# --- Main Execution ---
if __name__ == "__main__":
    # train_raw_materials_model()
    # train_grinding_model()
    # train_clinkerization_model()
    train_quality_model()