import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import sys

# --- Add Project Root to Python Path ---
# This ensures we can import from src.models, etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- Now we can import the Pydantic models ---
try:
    from src.schemas.data_models import (
        RawMaterialData, GrindingData, ClinkerizationData, QualityData
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the script is run from a location where 'src' is accessible, or the project root is in the Python path.")
    sys.exit(1)

# --- Configuration ---
NUM_SAMPLES = 5000  # Number of data points to generate for each process
OUTPUT_DIR = os.path.join(project_root, 'data') # Save data in a /data directory at the project root

# --- Helper Function for Particle Size Distribution ---
def generate_psd():
    """Generates a normalized 5-point particle size distribution."""
    dist = np.random.rand(5)
    return (dist / np.sum(dist)).tolist()

# --- Data Generation Functions ---

def generate_raw_material_data(n: int) -> pd.DataFrame:
    """Generates synthetic data for raw material processing."""
    data = []
    current_time = datetime.now()
    for i in range(n):
        record = RawMaterialData(
            timestamp=current_time - timedelta(minutes=15 * i),
            limestone_quality=np.random.uniform(0.85, 0.98),
            clay_content=np.random.uniform(0.05, 0.20),
            iron_ore_grade=np.random.uniform(0.60, 0.80),
            moisture_content=np.random.uniform(0.05, 0.25), # Wider range to create optimization opportunities
            particle_size_distribution=generate_psd(),
            temperature=np.random.uniform(15.0, 35.0),
            flow_rate=np.random.uniform(300.0, 450.0)
        )
        data.append(record.model_dump())
    return pd.DataFrame(data)

def generate_grinding_data(n: int) -> pd.DataFrame:
    """Generates synthetic data for the grinding process."""
    data = []
    current_time = datetime.now()
    for i in range(n):
        mill_power = np.random.uniform(4000, 5500)
        # Energy consumption should be plausibly related to mill power
        energy_consumption = mill_power * np.random.uniform(0.70, 0.95) 
        record = GrindingData(
            timestamp=current_time - timedelta(minutes=15 * i),
            mill_power=mill_power,
            feed_rate=np.random.uniform(180, 250),
            product_fineness=np.random.uniform(0.85, 0.98), # Often called Blaine
            energy_consumption=energy_consumption,
            temperature=np.random.uniform(90.0, 115.0),
            vibration_level=np.random.uniform(1.0, 5.0),
            noise_level=np.random.uniform(90.0, 105.0)
        )
        data.append(record.model_dump())
    return pd.DataFrame(data)

def generate_clinkerization_data(n: int) -> pd.DataFrame:
    """Generates synthetic data for the clinkerization process."""
    data = []
    current_time = datetime.now()
    for i in range(n):
        record = ClinkerizationData(
            timestamp=current_time - timedelta(minutes=15 * i),
            kiln_temperature=np.random.uniform(1380, 1520), # Wider range for optimization
            residence_time=np.random.uniform(20, 35),
            fuel_consumption=np.random.uniform(3000, 4500), # In MJ/ton or similar unit
            alternative_fuel_ratio=np.random.uniform(0.10, 0.40),
            clinker_quality=np.random.uniform(0.90, 0.99), # e.g., free lime content
            exhaust_gas_temperature=np.random.uniform(300, 450),
            oxygen_content=np.random.uniform(1.5, 4.0)
        )
        data.append(record.model_dump())
    return pd.DataFrame(data)

def generate_quality_data(n: int) -> pd.DataFrame:
    """Generates synthetic data for final product quality."""
    data = []
    current_time = datetime.now()
    product_types = ['OPC-43', 'OPC-53', 'PPC', 'PSC']
    for i in range(n):
        record = QualityData(
            timestamp=current_time - timedelta(hours=i), # Quality tests are less frequent
            product_type=random.choice(product_types),
            compressive_strength=np.random.uniform(40.0, 60.0), # In MPa
            fineness=np.random.uniform(0.90, 0.97),
            consistency=np.random.uniform(0.25, 0.35),
            setting_time=np.random.uniform(30.0, 120.0), # Initial setting time in minutes
            temperature=np.random.uniform(20.0, 30.0),
            humidity=np.random.uniform(40.0, 75.0)
        )
        data.append(record.model_dump())
    return pd.DataFrame(data)


# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating {NUM_SAMPLES} samples for each process...")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: '{OUTPUT_DIR}'")

    # # Generate and save data for each process
    # df_raw_materials = generate_raw_material_data(NUM_SAMPLES)
    # raw_path = os.path.join(OUTPUT_DIR, 'raw_materials.csv')
    # df_raw_materials.to_csv(raw_path, index=False)
    # print(f"✅ Successfully generated and saved raw materials data to {raw_path}")

    # df_grinding = generate_grinding_data(NUM_SAMPLES)
    # grinding_path = os.path.join(OUTPUT_DIR, 'grinding.csv')
    # df_grinding.to_csv(grinding_path, index=False)
    # print(f"✅ Successfully generated and saved grinding data to {grinding_path}")

    # df_clinkerization = generate_clinkerization_data(NUM_SAMPLES)
    # clinker_path = os.path.join(OUTPUT_DIR, 'clinkerization.csv')
    # df_clinkerization.to_csv(clinker_path, index=False)
    # print(f"✅ Successfully generated and saved clinkerization data to {clinker_path}")

    df_quality = generate_quality_data(NUM_SAMPLES)
    quality_path = os.path.join(OUTPUT_DIR, 'quality.csv')
    df_quality.to_csv(quality_path, index=False)
    print(f"✅ Successfully generated and saved quality data to {quality_path}")

    print("\n🎉 All data generation complete!")