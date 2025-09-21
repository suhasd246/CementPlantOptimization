import pandas as pd
import numpy as np 
import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any
import ast

# --- Add Project Root to Python Path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.schemas.data_models import (
    RawMaterialData, GrindingData, ClinkerizationData, QualityData,
    OptimizationResult, OptimizationRecommendation, OptimizationLevel, ProcessType
)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, 'data')

# --- Self-Contained Rule-Based Optimizer for Labeling ---
# This simple version is used ONLY for creating the training data.
class RuleBasedOptimizer:
    async def optimize_raw_materials(self, data: RawMaterialData) -> OptimizationResult:
        recommendations = []
        if data.moisture_content > 0.15:
            recommendations.append(OptimizationRecommendation(parameter="moisture_content", action="Increase pre-drying time", current_value=data.moisture_content, target_value=0.12, impact="Reduce grinding energy", priority=OptimizationLevel.HIGH, estimated_savings=0.10, implementation_time=30))
        if data.particle_size_distribution[0] > 0.3:
            recommendations.append(OptimizationRecommendation(parameter="particle_size", action="Adjust crusher settings", current_value=data.particle_size_distribution[0], target_value=0.25, impact="Improve grinding efficiency", priority=OptimizationLevel.MEDIUM, estimated_savings=0.08, implementation_time=15))
        return OptimizationResult(timestamp=datetime.now(), process=ProcessType.RAW_MATERIALS, recommendations=recommendations, expected_improvement=0.15 if recommendations else 0.0, confidence_score=0.85, energy_savings=sum(r.estimated_savings for r in recommendations), quality_improvement=0.05, sustainability_score=0.75)

    async def optimize_grinding(self, data: GrindingData) -> OptimizationResult:
        recommendations = []
        if data.energy_consumption > data.mill_power * 0.8:
            recommendations.append(OptimizationRecommendation(parameter="energy_efficiency", action="Optimize mill speed and grinding media", current_value=data.energy_consumption, target_value=data.mill_power * 0.75, impact="Reduce energy consumption", priority=OptimizationLevel.HIGH, estimated_savings=0.12, implementation_time=45))
        if data.product_fineness < 0.9:
            recommendations.append(OptimizationRecommendation(parameter="product_fineness", action="Adjust separator settings", current_value=data.product_fineness, target_value=0.92, impact="Improve product quality", priority=OptimizationLevel.MEDIUM, estimated_savings=0.05, implementation_time=20))
        return OptimizationResult(timestamp=datetime.now(), process=ProcessType.GRINDING, recommendations=recommendations, expected_improvement=0.12 if recommendations else 0.0, confidence_score=0.88, energy_savings=sum(r.estimated_savings for r in recommendations), quality_improvement=0.08, sustainability_score=0.80)

    async def optimize_clinkerization(self, data: ClinkerizationData) -> OptimizationResult:
        recommendations = []
        if data.kiln_temperature > 1450:
            recommendations.append(OptimizationRecommendation(parameter="kiln_temperature", action="Reduce temperature by 20-30°C", current_value=data.kiln_temperature, target_value=1420, impact="Reduce fuel consumption", priority=OptimizationLevel.HIGH, estimated_savings=0.10, implementation_time=60))
        if data.alternative_fuel_ratio < 0.3:
            recommendations.append(OptimizationRecommendation(parameter="alternative_fuel_ratio", action="Increase alternative fuel usage", current_value=data.alternative_fuel_ratio, target_value=0.35, impact="Reduce fossil fuel dependency", priority=OptimizationLevel.CRITICAL, estimated_savings=0.18, implementation_time=120))
        return OptimizationResult(timestamp=datetime.now(), process=ProcessType.CLINKERIZATION, recommendations=recommendations, expected_improvement=0.18 if recommendations else 0.0, confidence_score=0.82, energy_savings=sum(r.estimated_savings for r in recommendations), quality_improvement=0.06, sustainability_score=0.85)

    async def optimize_quality(self, data: Dict[str, Any]) -> OptimizationResult:
        recommendations = []
        if data.get('quality_variance', 0) > 0.05:
            recommendations.append(OptimizationRecommendation(parameter="quality_consistency", action="Implement real-time quality monitoring", current_value=data.get('quality_variance', 0), target_value=0.03, impact="Improve quality consistency", priority=OptimizationLevel.HIGH, estimated_savings=0.15, implementation_time=90))
        return OptimizationResult(timestamp=datetime.now(), process=ProcessType.QUALITY, recommendations=recommendations, expected_improvement=0.20 if recommendations else 0.0, confidence_score=0.90, energy_savings=sum(r.estimated_savings for r in recommendations), quality_improvement=0.25, sustainability_score=0.70)


# --- Labeling Functions ---

async def label_raw_materials_data():
    input_csv = os.path.join(DATA_DIR, 'raw_materials.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_raw_materials.csv')
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    for index, row in df.iterrows():
        row['particle_size_distribution'] = ast.literal_eval(row['particle_size_distribution'])
        input_data = RawMaterialData(**row.to_dict())
        result = await optimizer.optimize_raw_materials(input_data)
        labeled_row = row.to_dict()
        if result.recommendations:
            rec = result.recommendations[0] # Simplified to one rec per row for training
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'none', 'rec_target_value': 0.0})
        labeled_results.append(labeled_row)
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_grinding_data():
    input_csv = os.path.join(DATA_DIR, 'grinding.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_grinding.csv')
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    for index, row in df.iterrows():
        input_data = GrindingData(**row.to_dict())
        result = await optimizer.optimize_grinding(input_data)
        labeled_row = row.to_dict()
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'none', 'rec_target_value': 0.0})
        labeled_results.append(labeled_row)
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_clinkerization_data():
    input_csv = os.path.join(DATA_DIR, 'clinkerization.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_clinkerization.csv')
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    for index, row in df.iterrows():
        input_data = ClinkerizationData(**row.to_dict())
        result = await optimizer.optimize_clinkerization(input_data)
        labeled_row = row.to_dict()
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'none', 'rec_target_value': 0.0})
        labeled_results.append(labeled_row)
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_quality_data():
    input_csv = os.path.join(DATA_DIR, 'quality.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_quality.csv')
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    df['quality_variance'] = np.random.uniform(0.01, 0.09, size=len(df))
    for index, row in df.iterrows():
        input_dict = row.to_dict()
        result = await optimizer.optimize_quality(input_dict)
        del input_dict['quality_variance']
        labeled_row = input_dict
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'none', 'rec_target_value': 0.0})
        labeled_results.append(labeled_row)
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")


# --- Main Execution ---
async def main():
    await label_raw_materials_data()
    await label_grinding_data()
    await label_clinkerization_data()
    await label_quality_data()

if __name__ == "__main__":
    asyncio.run(main())
    print("\n🎉 All labeling complete!")