import pandas as pd
import numpy as np 
import asyncio
import os
import sys

# --- Add Project Root to Python Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.services.optimization_engine import CementPlantOptimizer
from src.schemas.data_models import RawMaterialData, GrindingData, ClinkerizationData, QualityData 

# --- Configuration ---
DATA_DIR = os.path.join(project_root, 'data')
INPUT_CSV = os.path.join(DATA_DIR, 'raw_materials.csv')
OUTPUT_LABELED_CSV = os.path.join(DATA_DIR, 'labeled_raw_materials.csv')

async def label_data():
    """
    Reads generated data, runs it through the existing optimization engine,
    and saves the inputs along with the generated recommendations (labels).
    """
    df = pd.read_csv(INPUT_CSV)
    optimizer = CementPlantOptimizer()
    
    labeled_results = []

    print(f"Starting to label {len(df)} rows from {INPUT_CSV}...")

    for index, row in df.iterrows():
        # Pydantic model expects a list for this field, but CSV reads it as a string
        row['particle_size_distribution'] = eval(row['particle_size_distribution'])

        # Create the Pydantic model from the row
        input_data = RawMaterialData(**row.to_dict())
        
        # Get the recommendation from your hardcoded engine
        result = await optimizer.optimize_raw_materials(input_data)
        
        if result.recommendations:
            # For each recommendation, create a labeled record
            for rec in result.recommendations:
                # Flatten the data into a single dictionary
                labeled_row = row.to_dict()
                labeled_row['rec_parameter'] = rec.parameter
                labeled_row['rec_action'] = rec.action
                labeled_row['rec_target_value'] = rec.target_value
                labeled_row['rec_priority'] = rec.priority.value
                labeled_results.append(labeled_row)
        else:
            # Optionally, keep rows with no recommendations
            labeled_row = row.to_dict()
            labeled_row['rec_parameter'] = 'none'
            labeled_row['rec_action'] = 'none'
            labeled_row['rec_target_value'] = 0.0
            labeled_row['rec_priority'] = 'none'
            labeled_results.append(labeled_row)

    # Convert to DataFrame and save
    labeled_df = pd.DataFrame(labeled_results)
    labeled_df.to_csv(OUTPUT_LABELED_CSV, index=False)
    print(f"\n✅ Successfully created labeled dataset at {OUTPUT_LABELED_CSV}")


async def label_grinding_data():
    """Labels the grinding data using the hardcoded optimizer logic."""
    input_csv = os.path.join(DATA_DIR, 'grinding.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_grinding.csv')
    
    df = pd.read_csv(input_csv)
    optimizer = CementPlantOptimizer()
    labeled_results = []

    print(f"Starting to label {len(df)} rows from {input_csv}...")

    for index, row in df.iterrows():
        input_data = GrindingData(**row.to_dict())
        result = await optimizer.optimize_grinding(input_data)
        
        if result.recommendations:
            for rec in result.recommendations:
                labeled_row = row.to_dict()
                labeled_row['rec_action'] = rec.action
                labeled_results.append(labeled_row)
        else:
            labeled_row = row.to_dict()
            labeled_row['rec_action'] = 'none'
            labeled_results.append(labeled_row)

    labeled_df = pd.DataFrame(labeled_results)
    labeled_df.to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")


async def label_clinkerization_data():
    """Labels the clinkerization data using the hardcoded optimizer logic."""
    input_csv = os.path.join(DATA_DIR, 'clinkerization.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_clinkerization.csv')
    
    df = pd.read_csv(input_csv)
    optimizer = CementPlantOptimizer()
    labeled_results = []

    print(f"Starting to label {len(df)} rows from {input_csv}...")

    for index, row in df.iterrows():
        input_data = ClinkerizationData(**row.to_dict())
        result = await optimizer.optimize_clinkerization(input_data)
        
        if result.recommendations:
            for rec in result.recommendations:
                labeled_row = row.to_dict()
                labeled_row['rec_action'] = rec.action
                labeled_results.append(labeled_row)
        else:
            labeled_row = row.to_dict()
            labeled_row['rec_action'] = 'none'
            labeled_results.append(labeled_row)

    labeled_df = pd.DataFrame(labeled_results)
    labeled_df.to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")


async def label_quality_data():
    """Labels the quality data using the hardcoded optimizer logic."""
    input_csv = os.path.join(DATA_DIR, 'quality.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_quality.csv')
    
    df = pd.read_csv(input_csv)
    optimizer = CementPlantOptimizer()
    labeled_results = []

    print(f"Starting to label {len(df)} rows from {input_csv}...")

    # The original rule was based on 'quality_variance'. We'll create a synthetic
    # version of it to trigger the rule for labeling purposes.
    # We'll say variance is high if strength is outside a certain range.
    strength_mean = df['compressive_strength'].mean()
    df['quality_variance'] = np.where(
        (df['compressive_strength'] < strength_mean - 5) | 
        (df['compressive_strength'] > strength_mean + 5), 
        0.06, 0.02
    )

    for index, row in df.iterrows():
        # The old method expected a dictionary, so we convert the row
        input_dict = row.to_dict()
        result = await optimizer.optimize_quality(input_dict)
        
        # We don't need the synthetic column in our final labeled data
        del input_dict['quality_variance']

        if result.recommendations:
            for rec in result.recommendations:
                labeled_row = input_dict
                labeled_row['rec_action'] = rec.action
                labeled_results.append(labeled_row)
        else:
            labeled_row = input_dict
            labeled_row['rec_action'] = 'none'
            labeled_results.append(labeled_row)

    labeled_df = pd.DataFrame(labeled_results)
    labeled_df.to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")


if __name__ == "__main__":
    # Run all labeling processes
    # asyncio.run(label_data()) 
    # asyncio.run(label_grinding_data())
    # asyncio.run(label_clinkerization_data())
    asyncio.run(label_quality_data())
