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

try:
    from src.schemas.data_models import (
        RawMaterialData, GrindingData, ClinkerizationData, QualityData,
        OptimizationResult, OptimizationRecommendation, OptimizationLevel, ProcessType
    )
except ImportError as e:
    print(f"Import Error: {e}. Please ensure src/schemas/data_models.py exists.")
    sys.exit(1)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, 'data')

# --- "The Brain": Rule-Based Optimizer for Labeling ---
# This class defines the "ground truth" rules for our synthetic data.
# We are making these rules more complex and interconnected.
class RuleBasedOptimizer:
    
    async def optimize_raw_materials(self, data: RawMaterialData) -> OptimizationResult:
        recommendations = []
        energy_savings = 0.0
        quality_improvement = 0.0
        
        coarse_particle = data.particle_size_distribution[0] > 0.3 # Coarsest bin
        high_moisture = data.moisture_content > 0.15
        low_quality = data.limestone_quality < 0.90
        high_flow = data.flow_rate > 400

        # --- New, Multi-Parameter Rules ---
        
        # Rule 1 (Critical): High moisture AND high throughput is a major problem
        if high_moisture and high_flow:
            savings = (data.moisture_content - 0.12) * 0.7
            recommendations.append(OptimizationRecommendation(
                parameter="moisture_content", 
                action="CRITICAL: Reduce moisture, high flow rate detected", 
                current_value=data.moisture_content, 
                target_value=0.12, 
                impact="Prevent mill overload and reduce energy", 
                priority=OptimizationLevel.CRITICAL, 
                estimated_savings=savings, 
                implementation_time=30))
            energy_savings += savings
            quality_improvement += 0.05
        
        # Rule 2 (High): Coarse particles AND low-quality material is a double-hit
        elif coarse_particle and low_quality:
            savings = (data.particle_size_distribution[0] - 0.25) * 0.4
            recommendations.append(OptimizationRecommendation(
                parameter="particle_size", 
                action="HIGH: Adjust crusher settings, low-quality mix", 
                current_value=data.particle_size_distribution[0], 
                target_value=0.25, 
                impact="Improve grindability for low-quality material", 
                priority=OptimizationLevel.HIGH, 
                estimated_savings=savings, 
                implementation_time=20))
            energy_savings += savings
            quality_improvement += 0.03

        # Rule 3 (Medium): Only high moisture
        elif high_moisture:
            savings = (data.moisture_content - 0.12) * 0.5
            recommendations.append(OptimizationRecommendation(
                parameter="moisture_content", 
                action="Increase pre-drying time", 
                current_value=data.moisture_content, 
                target_value=0.12, 
                impact="Reduce grinding energy", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=savings, 
                implementation_time=30))
            energy_savings += savings

        # Rule 4 (Low): Only coarse particles
        elif coarse_particle:
            savings = (data.particle_size_distribution[0] - 0.25) * 0.3
            recommendations.append(OptimizationRecommendation(
                parameter="particle_size", 
                action="Adjust crusher settings", 
                current_value=data.particle_size_distribution[0], 
                target_value=0.25, 
                impact="Improve grinding efficiency", 
                priority=OptimizationLevel.LOW, 
                estimated_savings=savings, 
                implementation_time=15))
            energy_savings += savings

        expected_improvement = (energy_savings + quality_improvement) / 2
        
        return OptimizationResult(
            timestamp=data.timestamp, 
            process=ProcessType.RAW_MATERIALS, 
            recommendations=recommendations, 
            expected_improvement=expected_improvement, 
            confidence_score=0.99,
            energy_savings=energy_savings, 
            quality_improvement=quality_improvement, 
            sustainability_score=0.75 + (energy_savings * 0.1)
        )

    async def optimize_grinding(self, data: GrindingData, specific_energy: float) -> OptimizationResult:
        recommendations = []
        energy_savings = 0.0
        quality_improvement = 0.0
        
        target_specific_energy = 22.0
        
        # --- New, Multi-Parameter Rules (Trade-offs) ---
        is_inefficient = specific_energy > 23.5
        is_coarse = data.product_fineness < 0.90
        is_vibrating = data.vibration_level > 4.5

        # Rule 1 (Critical): High vibration is a maintenance/safety alert
        if is_vibrating:
            recommendations.append(OptimizationRecommendation(
                parameter="vibration_level", 
                action="ALERT: Inspect mill for high vibration", 
                current_value=data.vibration_level, 
                target_value=3.0, 
                impact="Prevent mechanical failure", 
                priority=OptimizationLevel.CRITICAL, 
                estimated_savings=0.0, 
                implementation_time=60))
            # Don't add other recommendations; this is the priority
        
        # Rule 2 (High): Inefficient AND Coarse (Quality is the priority)
        elif is_inefficient and is_coarse:
            recommendations.append(OptimizationRecommendation(
                parameter="product_fineness", 
                action="Adjust separator settings, energy high", 
                current_value=data.product_fineness, 
                target_value=0.92, 
                impact="Improve product quality; energy is secondary issue", 
                priority=OptimizationLevel.HIGH, 
                estimated_savings=0.0, 
                implementation_time=20))
            quality_improvement += 0.08
        
        # Rule 3 (Medium): Only Inefficient (Energy is the priority)
        elif is_inefficient:
            savings = (specific_energy - target_specific_energy) / specific_energy
            recommendations.append(OptimizationRecommendation(
                parameter="specific_energy", 
                action="Optimize mill speed and grinding media", 
                current_value=specific_energy, 
                target_value=target_specific_energy, 
                impact="Reduce specific energy consumption", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=savings, 
                implementation_time=45))
            energy_savings += savings
        
        # Rule 4 (Medium): Only Coarse (Quality is the priority)
        elif is_coarse:
            recommendations.append(OptimizationRecommendation(
                parameter="product_fineness", 
                action="Adjust separator settings", 
                current_value=data.product_fineness, 
                target_value=0.92, 
                impact="Improve product quality", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=0.0,
                implementation_time=20))
            quality_improvement += 0.05
        
        expected_improvement = (energy_savings + quality_improvement) / 2
        
        return OptimizationResult(
            timestamp=data.timestamp, 
            process=ProcessType.GRINDING, 
            recommendations=recommendations, 
            expected_improvement=expected_improvement, 
            confidence_score=0.99, 
            energy_savings=energy_savings, 
            quality_improvement=quality_improvement, 
            sustainability_score=0.80 + (energy_savings * 0.1)
        )

    async def optimize_clinkerization(self, data: ClinkerizationData) -> OptimizationResult:
        recommendations = []
        energy_savings = 0.0
        quality_improvement = 0.0
        sustainability_boost = 0.0
        
        # --- New, Multi-Parameter Rules ---
        is_temp_high = data.kiln_temperature > 1450
        is_alt_fuel_low = data.alternative_fuel_ratio < 0.3
        is_quality_low = data.clinker_quality < 0.92

        # Rule 1 (Critical): Quality is the #1 priority
        if is_quality_low:
            recommendations.append(OptimizationRecommendation(
                parameter="clinker_quality", 
                action="CRITICAL: Stabilize kiln, clinker quality low", 
                current_value=data.clinker_quality, 
                target_value=0.95, 
                impact="Improve clinker quality, stabilize temperature", 
                priority=OptimizationLevel.CRITICAL, 
                estimated_savings=0.0, 
                implementation_time=45))
            quality_improvement += 0.15
        
        # Rule 2 (High): Temp high AND Alt Fuel low (Holistic optimization)
        elif is_temp_high and is_alt_fuel_low:
            savings = (0.35 - data.alternative_fuel_ratio) * 0.5
            recommendations.append(OptimizationRecommendation(
                parameter="alternative_fuel_ratio", 
                action="Optimize fuel mix: Increase alt fuel, reduce temp", 
                current_value=data.alternative_fuel_ratio, 
                target_value=0.35, 
                impact="Reduce fossil fuel and total thermal load", 
                priority=OptimizationLevel.HIGH, 
                estimated_savings=savings, 
                implementation_time=120))
            energy_savings += savings
            sustainability_boost += 0.1
        
        # Rule 3 (Medium): Only Temp high
        elif is_temp_high:
            savings = (data.kiln_temperature - 1420) / 1450 * 0.2
            recommendations.append(OptimizationRecommendation(
                parameter="kiln_temperature", 
                action="Reduce temperature by 20-30°C", 
                current_value=data.kiln_temperature, 
                target_value=1420, 
                impact="Reduce fuel consumption", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=savings, 
                implementation_time=60))
            energy_savings += savings

        # Rule 4 (Medium): Only Alt Fuel low
        elif is_alt_fuel_low:
            savings = (0.35 - data.alternative_fuel_ratio) * 0.5
            recommendations.append(OptimizationRecommendation(
                parameter="alternative_fuel_ratio", 
                action="Increase alternative fuel usage", 
                current_value=data.alternative_fuel_ratio, 
                target_value=0.35, 
                impact="Reduce fossil fuel dependency", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=savings, 
                implementation_time=120))
            energy_savings += savings
            sustainability_boost += 0.1

        expected_improvement = (energy_savings + quality_improvement) / 2
        
        return OptimizationResult(
            timestamp=data.timestamp, 
            process=ProcessType.CLINKERIZATION, 
            recommendations=recommendations, 
            expected_improvement=expected_improvement, 
            confidence_score=0.99, 
            energy_savings=energy_savings, 
            quality_improvement=quality_improvement, 
            sustainability_score=0.85 + sustainability_boost
        )

    async def optimize_quality(self, data: QualityData) -> OptimizationResult:
        recommendations = []
        energy_savings = 0.0
        quality_improvement = 0.0

        # --- New, Multi-Parameter Rules ---
        is_strength_low = data.compressive_strength < 45.0
        is_set_time_short = data.setting_time < 60.0 # Flash set
        is_set_time_long = data.setting_time > 100.0

        # Rule 1 (Critical): Flash set is a major quality failure
        if is_set_time_short:
            target_gypsum = data.gypsum_added * 1.05
            recommendations.append(OptimizationRecommendation(
                parameter="gypsum_added", 
                action="Increase gypsum dosing", 
                current_value=data.gypsum_added, 
                target_value=target_gypsum, 
                impact="CRITICAL: Prevent flash setting", 
                priority=OptimizationLevel.CRITICAL, 
                estimated_savings=0.0, 
                implementation_time=10))
            quality_improvement += 0.15
        
        # Rule 2 (High): Strength is low
        elif is_strength_low:
            # (This is a proxy. A real fix would be fineness/clinker)
            target_gypsum = data.gypsum_added * 1.02 # Slight adjustment
            recommendations.append(OptimizationRecommendation(
                parameter="gypsum_added", 
                action="Adjust gypsum for low strength", 
                current_value=data.gypsum_added, 
                target_value=target_gypsum, 
                impact="Optimize mix to improve compressive strength", 
                priority=OptimizationLevel.HIGH, 
                estimated_savings=0.0, 
                implementation_time=20))
            quality_improvement += 0.10
        
        # Rule 3 (Medium): Setting time is too long
        elif is_set_time_long:
            target_gypsum = data.gypsum_added * 0.95
            recommendations.append(OptimizationRecommendation(
                parameter="gypsum_added", 
                action="Reduce gypsum dosing", 
                current_value=data.gypsum_added, 
                target_value=target_gypsum, 
                impact="Correct long setting time", 
                priority=OptimizationLevel.MEDIUM, 
                estimated_savings=0.02, # Savings from less gypsum
                implementation_time=10))
            energy_savings += 0.02
            quality_improvement += 0.05
            
        expected_improvement = (energy_savings + quality_improvement) / 2
        
        return OptimizationResult(
            timestamp=data.timestamp, 
            process=ProcessType.QUALITY, 
            recommendations=recommendations, 
            expected_improvement=expected_improvement, 
            confidence_score=0.99, 
            energy_savings=energy_savings, 
            quality_improvement=quality_improvement, 
            sustainability_score=0.70 + quality_improvement * 0.1
        )

# --- Labeling Functions (UPDATED) ---
# These functions now write all the new KPI labels to the CSVs.

async def label_raw_materials_data():
    input_csv = os.path.join(DATA_DIR, 'raw_materials.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_raw_materials.csv')
    if not os.path.exists(input_csv):
        print(f"Warning: {input_csv} not found. Skipping Raw Materials.")
        return
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        row_dict['particle_size_distribution'] = ast.literal_eval(row_dict['particle_size_distribution'])
        
        input_data = RawMaterialData(**row_dict)
        result = await optimizer.optimize_raw_materials(input_data)
        
        labeled_row = row_dict
        
        # --- Add all KPI labels ---
        labeled_row['energy_savings'] = result.energy_savings
        labeled_row['quality_improvement'] = result.quality_improvement
        labeled_row['sustainability_score'] = result.sustainability_score
        labeled_row['expected_improvement'] = result.expected_improvement
        
        if result.recommendations:
            rec = result.recommendations[0] 
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'maintain_parameters', 'rec_target_value': 0.0})
            
        labeled_results.append(labeled_row)
        
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_grinding_data():
    input_csv = os.path.join(DATA_DIR, 'grinding.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_grinding.csv')
    if not os.path.exists(input_csv):
        print(f"Warning: {input_csv} not found. Skipping Grinding.")
        return
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    
    if 'specific_energy' not in df.columns:
        df['specific_energy'] = df['energy_consumption'] / (df['feed_rate'] + 1e-6)

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        specific_energy_val = row_dict.pop('specific_energy') 
        
        input_data = GrindingData(**row_dict)
        result = await optimizer.optimize_grinding(input_data, specific_energy_val)
        
        labeled_row = row_dict
        labeled_row['specific_energy'] = specific_energy_val
        
        # --- Add all KPI labels ---
        labeled_row['energy_savings'] = result.energy_savings
        labeled_row['quality_improvement'] = result.quality_improvement
        labeled_row['sustainability_score'] = result.sustainability_score
        labeled_row['expected_improvement'] = result.expected_improvement
        
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'maintain_parameters', 'rec_target_value': 0.0})
            
        labeled_results.append(labeled_row)
        
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_clinkerization_data():
    input_csv = os.path.join(DATA_DIR, 'clinkerization.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_clinkerization.csv')
    if not os.path.exists(input_csv):
        print(f"Warning: {input_csv} not found. Skipping Clinkerization.")
        return
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    
    for index, row in df.iterrows():
        row_dict = row.to_dict()
        input_data = ClinkerizationData(**row_dict)
        result = await optimizer.optimize_clinkerization(input_data)
        
        labeled_row = row_dict
        
        # --- Add all KPI labels ---
        labeled_row['energy_savings'] = result.energy_savings
        labeled_row['quality_improvement'] = result.quality_improvement
        labeled_row['sustainability_score'] = result.sustainability_score
        labeled_row['expected_improvement'] = result.expected_improvement
        
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'maintain_parameters', 'rec_target_value': 0.0})
            
        labeled_results.append(labeled_row)
        
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")

async def label_quality_data():
    input_csv = os.path.join(DATA_DIR, 'quality.csv')
    output_csv = os.path.join(DATA_DIR, 'labeled_quality.csv')
    if not os.path.exists(input_csv):
        print(f"Warning: {input_csv} not found. Skipping Quality.")
        return
    df = pd.read_csv(input_csv)
    optimizer = RuleBasedOptimizer()
    labeled_results = []
    print(f"Starting to label {len(df)} rows from {input_csv}...")
    
    if 'gypsum_added' not in df.columns:
        df['gypsum_added'] = df['setting_time'].apply(lambda x: (x - np.random.uniform(25.0, 90.0)) / 5.0)
        df['gypsum_added'] = df['gypsum_added'].clip(2.5, 5.5)

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        input_data = QualityData(**row_dict)
        result = await optimizer.optimize_quality(input_data)
        
        labeled_row = row_dict
        
        # --- Add all KPI labels ---
        labeled_row['energy_savings'] = result.energy_savings
        labeled_row['quality_improvement'] = result.quality_improvement
        labeled_row['sustainability_score'] = result.sustainability_score
        labeled_row['expected_improvement'] = result.expected_improvement
        
        if result.recommendations:
            rec = result.recommendations[0]
            labeled_row.update({'rec_action': rec.action, 'rec_target_value': rec.target_value})
        else:
            labeled_row.update({'rec_action': 'maintain_parameters', 'rec_target_value': 0.0})
            
        labeled_results.append(labeled_row)
        
    pd.DataFrame(labeled_results).to_csv(output_csv, index=False)
    print(f"✅ Successfully created labeled dataset at {output_csv}")


# --- Main Execution ---
async def main():
    print("Starting data labeling process with complex rules...")
    await label_raw_materials_data()
    await label_grinding_data()
    await label_clinkerization_data()
    await label_quality_data()
    print("\n🎉 All labeling complete!")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    asyncio.run(main())