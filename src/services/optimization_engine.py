# ~/Documents/cement-operations-optimization/src/services/optimization_engine.py
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from ..schemas.data_models import (
    RawMaterialData, GrindingData, ClinkerizationData, QualityData,
    OptimizationResult, OptimizationRecommendation, OptimizationLevel, ProcessType
)
import os
import joblib
import pandas as pd

logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(project_root, 'artifacts')

class CementPlantOptimizer:
    def __init__(self):
        self.optimization_history = []
        model_path = os.path.join(MODEL_DIR, 'raw_materials_model.pkl')
        encoder_path = os.path.join(MODEL_DIR, 'raw_materials_encoder.pkl')

        try:
            logger.info("Loading AI model for raw materials...")
            self.raw_materials_model = joblib.load(model_path)
            self.raw_materials_encoder = joblib.load(encoder_path)
            logger.info("✅ Raw materials model loaded successfully.")
            # Define the feature names in the exact order the model was trained on
            self.feature_names = [
                'limestone_quality', 'clay_content', 'iron_ore_grade', 'moisture_content',
                'temperature', 'flow_rate', 'psd_1', 'psd_2', 'psd_3', 'psd_4', 'psd_5'
            ]
        except FileNotFoundError:
            logger.error(f"Model or encoder not found at {MODEL_DIR}. Please run the training script.")
            self.raw_materials_model = None
            self.raw_materials_encoder = None

        grinding_model_path = os.path.join(MODEL_DIR, 'grinding_model.pkl')
        grinding_encoder_path = os.path.join(MODEL_DIR, 'grinding_encoder.pkl')
        
        try:
            logger.info("Loading AI model for grinding...")
            self.grinding_model = joblib.load(grinding_model_path)
            self.grinding_encoder = joblib.load(grinding_encoder_path)
            logger.info("✅ Grinding model loaded successfully.")
            # Define the feature names for the grinding model
            self.grinding_feature_names = [
                'mill_power', 'feed_rate', 'product_fineness', 'energy_consumption',
                'temperature', 'vibration_level', 'noise_level'
            ]
        except FileNotFoundError:
            logger.error(f"Grinding model not found at {MODEL_DIR}.")
            self.grinding_model = None
            self.grinding_encoder = None

        clinker_model_path = os.path.join(MODEL_DIR, 'clinkerization_model.pkl')
        clinker_encoder_path = os.path.join(MODEL_DIR, 'clinkerization_encoder.pkl')
        
        try:
            logger.info("Loading AI model for clinkerization...")
            self.clinkerization_model = joblib.load(clinker_model_path)
            self.clinkerization_encoder = joblib.load(clinker_encoder_path)
            logger.info("✅ Clinkerization model loaded successfully.")
            self.clinkerization_feature_names = [
                'kiln_temperature', 'residence_time', 'fuel_consumption', 
                'alternative_fuel_ratio', 'clinker_quality', 
                'exhaust_gas_temperature', 'oxygen_content'
            ]
        except FileNotFoundError:
            logger.error(f"Clinkerization model not found at {MODEL_DIR}.")
            self.clinkerization_model = None
            self.clinkerization_encoder = None

        quality_model_path = os.path.join(MODEL_DIR, 'quality_model.pkl')
        quality_encoder_path = os.path.join(MODEL_DIR, 'quality_encoder.pkl')
        
        try:
            logger.info("Loading AI model for quality...")
            self.quality_model = joblib.load(quality_model_path)
            self.quality_encoder = joblib.load(quality_encoder_path)
            logger.info("✅ Quality model loaded successfully.")
            self.quality_feature_names = [
                'compressive_strength', 'fineness', 'consistency', 
                'setting_time', 'temperature', 'humidity'
            ]
        except FileNotFoundError:
            logger.error(f"Quality model not found at {MODEL_DIR}.")
            self.quality_model = None
            self.quality_encoder = None
        
    async def optimize_raw_materials(self, data: RawMaterialData) -> OptimizationResult:
        """Optimize raw material processing and grinding efficiency"""
        recommendations = []
        
        if not self.raw_materials_model:
            # Fallback or error if the model failed to load
            raise RuntimeError("Raw materials optimization model is not available.")
            
        # 1. PREPARE INPUT DATA: Convert Pydantic object to a DataFrame format that the model expects.
        input_data = data.model_dump()
        
        # Expand the particle size distribution list into separate columns, just like in training
        psd_features = input_data.pop('particle_size_distribution')
        for i, val in enumerate(psd_features):
            input_data[f'psd_{i+1}'] = val
            
        # Create a single-row DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure the column order is exactly what the model was trained on
        input_df = input_df[self.feature_names]

        # 2. GET PREDICTION: Use the model to predict the encoded action (e.g., 0, 1, 2...).
        predicted_action_encoded = self.raw_materials_model.predict(input_df)[0]
        
        # 3. DECODE PREDICTION: Convert the numeric prediction back to a meaningful string (e.g., 'Increase pre-drying time').
        predicted_action = self.raw_materials_encoder.inverse_transform([predicted_action_encoded])[0]
        
        logger.info(f"AI Model Prediction: '{predicted_action}'")
        
        # 4. CONSTRUCT RECOMMENDATION: If the model recommends an action, build the response.
        if predicted_action != 'none':
            # This is a simple lookup for recommendation details.
            # In a more advanced system, other models could predict these values.
            rec_details = {
                'Increase pre-drying time': {
                    'parameter': 'moisture_content',
                    'impact': 'Reduce grinding energy by 8-12%',
                    'priority': OptimizationLevel.HIGH,
                    'target': 0.12,
                    'savings': 0.10,
                    'time': 30
                },
                'Adjust crusher settings': {
                    'parameter': 'particle_size',
                    'impact': 'Improve grinding efficiency by 5-8%',
                    'priority': OptimizationLevel.MEDIUM,
                    'target': 0.25,
                    'savings': 0.08,
                    'time': 15
                }
            }
            
            details = rec_details.get(predicted_action)
            if details:
                recommendations.append(OptimizationRecommendation(
                    parameter=details['parameter'],
                    action=predicted_action,
                    current_value=getattr(data, details['parameter']) if hasattr(data, details['parameter']) else input_df['psd_1'].iloc[0],
                    target_value=details['target'],
                    impact=details['impact'],
                    priority=details['priority'],
                    estimated_savings=details['savings'],
                    implementation_time=details['time']
                ))

        # Build and return the final result object
        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.RAW_MATERIALS,
            recommendations=recommendations,
            expected_improvement=0.15 if recommendations else 0.0,
            confidence_score=0.95, # High confidence as it comes from the model
            energy_savings=sum(r.estimated_savings for r in recommendations),
            quality_improvement=0.05,
            sustainability_score=0.78
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_grinding(self, data: GrindingData) -> OptimizationResult:
        """Optimize grinding process for energy efficiency"""
        recommendations = []
        if not self.grinding_model:
            raise RuntimeError("Grinding optimization model is not available.")
            
        # 1. PREPARE INPUT DATA
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[self.grinding_feature_names]

        # 2. GET PREDICTION
        predicted_action_encoded = self.grinding_model.predict(input_df)[0]
        
        # 3. DECODE PREDICTION
        predicted_action = self.grinding_encoder.inverse_transform([predicted_action_encoded])[0]
        logger.info(f"AI Grinding Model Prediction: '{predicted_action}'")
        
        # 4. CONSTRUCT RECOMMENDATION
        if predicted_action != 'none':
            rec_details = {
                'Optimize mill speed and grinding media': {
                    'parameter': 'energy_efficiency',
                    'impact': 'Reduce energy consumption by 10-15%',
                    'priority': OptimizationLevel.HIGH,
                    'target': data.mill_power * 0.75,
                    'savings': 0.12,
                    'time': 45
                },
                'Adjust separator settings': {
                    'parameter': 'product_fineness',
                    'impact': 'Improve product quality consistency',
                    'priority': OptimizationLevel.MEDIUM,
                    'target': 0.92,
                    'savings': 0.05,
                    'time': 20
                }
            }
            
            details = rec_details.get(predicted_action)
            if details:
                current_val = data.energy_consumption if details['parameter'] == 'energy_efficiency' else data.product_fineness
                recommendations.append(OptimizationRecommendation(
                    parameter=details['parameter'],
                    action=predicted_action,
                    current_value=current_val,
                    target_value=details['target'],
                    impact=details['impact'],
                    priority=details['priority'],
                    estimated_savings=details['savings'],
                    implementation_time=details['time']
                ))

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.GRINDING,
            recommendations=recommendations,
            expected_improvement=0.12 if recommendations else 0.0,
            confidence_score=0.96,
            energy_savings=sum(r.estimated_savings for r in recommendations),
            quality_improvement=0.08,
            sustainability_score=0.82
        )
        
        self.optimization_history.append(result)
        return result    
    
    async def optimize_clinkerization(self, data: ClinkerizationData) -> OptimizationResult:
        """Optimize clinkerization process for energy and quality"""
        recommendations = []
        
        # Temperature optimization
        if not self.clinkerization_model:
            raise RuntimeError("Clinkerization optimization model is not available.")
            
        # 1. PREPARE INPUT DATA
        input_df = pd.DataFrame([data.model_dump()])
        input_df = input_df[self.clinkerization_feature_names]

        # 2. GET PREDICTION
        predicted_action_encoded = self.clinkerization_model.predict(input_df)[0]
        
        # 3. DECODE PREDICTION
        predicted_action = self.clinkerization_encoder.inverse_transform([predicted_action_encoded])[0]
        logger.info(f"AI Clinkerization Model Prediction: '{predicted_action}'")
        
        # 4. CONSTRUCT RECOMMENDATION
        if predicted_action != 'none':
            rec_details = {
                'Reduce temperature by 20-30°C': {
                    'parameter': 'kiln_temperature',
                    'impact': 'Reduce fuel consumption by 8-12%',
                    'priority': OptimizationLevel.HIGH,
                    'target': 1420,
                    'savings': 0.10,
                    'time': 60
                },
                'Increase alternative fuel usage': {
                    'parameter': 'alternative_fuel_ratio',
                    'impact': 'Reduce fossil fuel dependency by 15-20%',
                    'priority': OptimizationLevel.CRITICAL,
                    'target': 0.35,
                    'savings': 0.18,
                    'time': 120
                }
            }
            
            details = rec_details.get(predicted_action)
            if details:
                recommendations.append(OptimizationRecommendation(
                    parameter=details['parameter'],
                    action=predicted_action,
                    current_value=getattr(data, details['parameter']),
                    target_value=details['target'],
                    impact=details['impact'],
                    priority=details['priority'],
                    estimated_savings=details['savings'],
                    implementation_time=details['time']
                ))

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.CLINKERIZATION,
            recommendations=recommendations,
            expected_improvement=0.18 if recommendations else 0.0,
            confidence_score=0.94,
            energy_savings=sum(r.estimated_savings for r in recommendations),
            quality_improvement=0.06,
            sustainability_score=0.88
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_quality(self, data: QualityData) -> OptimizationResult:
        """Optimize product quality consistency"""
        recommendations = []
        
        if not self.quality_model:
            raise RuntimeError("Quality optimization model is not available.")
            
        # 1. PREPARE INPUT DATA
        input_df = pd.DataFrame([data.model_dump()])
        # Ensure we only use the features the model was trained on
        input_df = input_df[self.quality_feature_names]

        # 2. GET PREDICTION & 3. DECODE PREDICTION
        predicted_action_encoded = self.quality_model.predict(input_df)[0]
        predicted_action = self.quality_encoder.inverse_transform([predicted_action_encoded])[0]
        logger.info(f"AI Quality Model Prediction: '{predicted_action}'")
        
        # 4. CONSTRUCT RECOMMENDATION
        if predicted_action != 'none':
            details = {
                'parameter': 'quality_consistency',
                'impact': 'Improve quality consistency by 40-50%',
                'priority': OptimizationLevel.HIGH,
                'target': 0.03, # Target variance
                'savings': 0.15,
                'time': 90
            }
            recommendations.append(OptimizationRecommendation(
                parameter=details['parameter'],
                action=predicted_action,
                current_value=data.compressive_strength, # Using strength as a proxy indicator
                target_value=details['target'],
                impact=details['impact'],
                priority=details['priority'],
                estimated_savings=details['savings'],
                implementation_time=details['time']
            ))

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.QUALITY,
            recommendations=recommendations,
            expected_improvement=0.20 if recommendations else 0.0,
            confidence_score=0.92,
            energy_savings=sum(r.estimated_savings for r in recommendations),
            quality_improvement=0.25,
            sustainability_score=0.70
        )
        
        self.optimization_history.append(result)
        return result
    
    
    async def get_plant_status(self) -> Dict[str, Any]:
        """Get overall plant status"""
        if not self.optimization_history:
            return {
                "timestamp": datetime.now(),
                "overall_efficiency": 0.82,
                "energy_consumption": 2000.0,
                "quality_score": 0.85,
                "sustainability_score": 0.75,
                "active_recommendations": 0,
                "critical_alerts": 0
            }
        
        recent_results = [r for r in self.optimization_history 
                         if (datetime.now() - r.timestamp).total_seconds() < 3600]
        
        if not recent_results:
            return {
                "timestamp": datetime.now(),
                "overall_efficiency": 0.82,
                "energy_consumption": 2000.0,
                "quality_score": 0.85,
                "sustainability_score": 0.75,
                "active_recommendations": 0,
                "critical_alerts": 0
            }
        
        overall_efficiency = np.mean([r.expected_improvement for r in recent_results])
        energy_consumption = np.mean([r.energy_savings for r in recent_results])
        quality_score = np.mean([r.quality_improvement for r in recent_results])
        sustainability_score = np.mean([r.sustainability_score for r in recent_results])
        
        active_recommendations = sum(len(r.recommendations) for r in recent_results)
        critical_alerts = sum(1 for r in recent_results 
                             for rec in r.recommendations 
                             if rec.priority == OptimizationLevel.CRITICAL)
        
        return {
            "timestamp": datetime.now(),
            "overall_efficiency": float(overall_efficiency),
            "energy_consumption": float(energy_consumption),
            "quality_score": float(quality_score),
            "sustainability_score": float(sustainability_score),
            "active_recommendations": active_recommendations,
            "critical_alerts": critical_alerts
        }


