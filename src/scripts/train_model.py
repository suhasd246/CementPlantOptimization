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
DATA_PATH = os.path.join(project_root, 'data', 'labeled_raw_materials.csv')
MODEL_OUTPUT_DIR = os.path.join(project_root, 'artifacts') # Directory to save trained models
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'raw_materials_model.pkl')
ENCODER_PATH = os.path.join(MODEL_OUTPUT_DIR, 'raw_materials_encoder.pkl')


def train_raw_materials_model():
    """
    Loads labeled data, trains an XGBoost classifier, and saves the model
    and label encoder.
    """
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    # --- 1. Feature Engineering & Preprocessing ---
    print("Preprocessing data...")
    
    # The model can't use the text-based recommendation columns as input features
    # These are our targets or just metadata
    cols_to_drop = ['rec_parameter', 'rec_action', 'rec_target_value', 'rec_priority', 'timestamp']
    
    # Handle the 'particle_size_distribution' which is a string of a list
    # We'll expand it into separate features: psd_1, psd_2, etc.
    psd_df = df['particle_size_distribution'].apply(lambda x: pd.Series(ast.literal_eval(x)))
    psd_df.columns = [f'psd_{i+1}' for i in range(psd_df.shape[1])]
    
    # Combine the new psd features and drop the original column
    df = pd.concat([df, psd_df], axis=1).drop('particle_size_distribution', axis=1)

    # Define our features (X) and target (y)
    X = df.drop(columns=cols_to_drop)
    y = df['rec_action']

    # Encode the target variable (e.g., 'Increase pre-drying time' -> 0, 'none' -> 1)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Features used for training: {X.columns.tolist()}")
    print(f"Target classes: {encoder.classes_}")

    # --- 2. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 3. Model Training ---
    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(encoder.classes_),
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # --- 4. Model Evaluation ---
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Data: {accuracy:.4f}")
    
    # Show a detailed report
    print("\nClassification Report:")
    # We need to decode the numeric predictions back to text labels for the report
    y_test_labels = encoder.inverse_transform(y_test)
    y_pred_labels = encoder.inverse_transform(y_pred)
    print(classification_report(y_test_labels, y_pred_labels))

    # --- 5. Save the Model and Encoder ---
    print("Saving model and encoder...")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True) # Create models directory if it doesn't exist
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Encoder saved to: {ENCODER_PATH}")

def train_grinding_model():
    """
    Loads labeled grinding data, trains an XGBoost classifier, and saves the model.
    """
    data_path = os.path.join(project_root, 'data', 'labeled_grinding.csv')
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'grinding_model.pkl')
    encoder_path = os.path.join(MODEL_OUTPUT_DIR, 'grinding_encoder.pkl')

    print(f"\n--- Training Grinding Model ---")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 1. Preprocessing ---
    # Define features (X) and target (y)
    cols_to_drop = ['rec_action', 'timestamp']
    X = df.drop(columns=cols_to_drop)
    y = df['rec_action']

    # Encode the target variable
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Features used for training: {X.columns.tolist()}")
    print(f"Target classes: {encoder.classes_}")

    # --- 2. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 3. Model Training ---
    print("\nTraining XGBoost model for grinding...")
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(encoder.classes_),
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)

    # --- 4. Model Evaluation ---
    print("\nEvaluating grinding model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Grinding Model Accuracy on Test Data: {accuracy:.4f}")

    # --- 5. Save the Model and Encoder ---
    print("Saving grinding model and encoder...")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Encoder saved to: {encoder_path}")


def train_clinkerization_model():
    """
    Loads labeled clinkerization data, trains an XGBoost classifier, and saves it.
    """
    data_path = os.path.join(project_root, 'data', 'labeled_clinkerization.csv')
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'clinkerization_model.pkl')
    encoder_path = os.path.join(MODEL_OUTPUT_DIR, 'clinkerization_encoder.pkl')

    print(f"\n--- Training Clinkerization Model ---")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 1. Preprocessing ---
    cols_to_drop = ['rec_action', 'timestamp']
    X = df.drop(columns=cols_to_drop)
    y = df['rec_action']
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Features used for training: {X.columns.tolist()}")
    print(f"Target classes: {encoder.classes_}")

    # --- 2. Train-Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- 3. Model Training ---
    print("\nTraining XGBoost model for clinkerization...")
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # --- 4. Model Evaluation ---
    print("\nEvaluating clinkerization model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Clinkerization Model Accuracy on Test Data: {accuracy:.4f}")

    # --- 5. Save the Model and Encoder ---
    print("Saving clinkerization model and encoder...")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Encoder saved to: {encoder_path}")


def train_quality_model():
    """
    Loads labeled quality data, trains an XGBoost classifier, and saves it.
    """
    data_path = os.path.join(project_root, 'data', 'labeled_quality.csv')
    model_path = os.path.join(MODEL_OUTPUT_DIR, 'quality_model.pkl')
    encoder_path = os.path.join(MODEL_OUTPUT_DIR, 'quality_encoder.pkl')

    print(f"\n--- Training Quality Model ---")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # --- 1. Preprocessing ---
    # NOTE: 'product_type' is categorical. For a more advanced model, you would
    # use one-hot encoding. For simplicity here, we are dropping it.
    cols_to_drop = ['rec_action', 'timestamp', 'product_type']
    X = df.drop(columns=cols_to_drop)
    y = df['rec_action']
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    print(f"Features used for training: {X.columns.tolist()}")
    print(f"Target classes: {encoder.classes_}")

    # --- 2. Train-Test Split & 3. Model Training ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print("\nTraining XGBoost model for quality...")
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(encoder.classes_), use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    # --- 4. Model Evaluation ---
    print("\nEvaluating quality model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Quality Model Accuracy on Test Data: {accuracy:.4f}")

    # --- 5. Save the Model and Encoder ---
    print("Saving quality model and encoder...")
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Encoder saved to: {encoder_path}")


if __name__ == "__main__":
    # train_raw_materials_model()
    # train_grinding_model()
    # train_clinkerization_model()
    train_quality_model()
