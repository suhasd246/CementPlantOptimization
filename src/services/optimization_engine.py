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
ARTIFACTS_DIR  = os.path.join(project_root, 'artifacts')

class CementPlantOptimizer:
    def __init__(self):
        self.optimization_history = []
        # --- Load Raw Materials Artifacts ---
        try:
            logger.info("Loading Raw Materials artifacts...")
            self.raw_materials_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_model.pkl'))
            self.raw_materials_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_encoder.pkl'))
            self.raw_materials_regressor = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_regressor.pkl'))
            self.raw_materials_features = [
                'limestone_quality', 'clay_content', 'iron_ore_grade', 'moisture_content', 'temperature', 
                'flow_rate', 'psd_1', 'psd_2', 'psd_3', 'psd_4', 'psd_5'
            ]
            logger.info("✅ Raw Materials artifacts loaded.")
        except FileNotFoundError:
            logger.error("A Raw Materials model file was not found. Please run the training script.")
            self.raw_materials_model = None

        # --- Load Grinding Artifacts ---
        try:
            logger.info("Loading Grinding artifacts...")
            self.grinding_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_model.pkl'))
            self.grinding_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_encoder.pkl'))
            self.grinding_regressor = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_regressor.pkl'))
            self.grinding_features = [
                'mill_power', 'feed_rate', 'product_fineness', 'energy_consumption',
                'temperature', 'vibration_level', 'noise_level'
            ]
            logger.info("✅ Grinding artifacts loaded.")
        except FileNotFoundError:
            logger.error("A Grinding model file was not found. Please run the training script.")
            self.grinding_model = None

        # --- Load Clinkerization Artifacts ---
        try:
            logger.info("Loading Clinkerization artifacts...")
            self.clinkerization_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_model.pkl'))
            self.clinkerization_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_encoder.pkl'))
            self.clinkerization_regressor = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_regressor.pkl'))
            self.clinkerization_features = [
                'kiln_temperature', 'residence_time', 'fuel_consumption', 
                'alternative_fuel_ratio', 'clinker_quality', 
                'exhaust_gas_temperature', 'oxygen_content'
            ]
            logger.info("✅ Clinkerization artifacts loaded.")
        except FileNotFoundError:
            logger.error("A Clinkerization model file was not found. Please run the training script.")
            self.clinkerization_model = None

        # --- Load Quality Artifacts ---
        try:
            logger.info("Loading AI model for quality...")
             # Quality
            self.quality_model = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_model.pkl'))
            self.quality_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_encoder.pkl'))
            self.quality_regressor = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_regressor.pkl'))
            self.quality_feature_names = ['compressive_strength', 'fineness', 'consistency', 'setting_time', 'temperature', 'humidity']
            logger.info("✅ Quality model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Quality model not found at {ARTIFACTS_DIR}.")
            self.quality_model = None


    async def optimize_raw_materials(self, data: RawMaterialData) -> OptimizationResult:
        """Optimize raw material processing and grinding efficiency"""
        if not self.raw_materials_model:
            raise RuntimeError("Raw materials optimization model is not available.")

        # 1. Prepare input data
        input_data = data.model_dump()
        psd_features = input_data.pop('particle_size_distribution')
        for i, val in enumerate(psd_features):
            input_data[f'psd_{i+1}'] = val
        input_df = pd.DataFrame([input_data])[self.raw_materials_features]

        # 2. Get Action and Confidence from CLASSIFIER using predict_proba
        probabilities = self.raw_materials_model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = self.raw_materials_encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        recommendations = []
        estimated_savings = 0.0

        if predicted_action != 'none':
            # 3. Get DYNAMIC Target Value from REGRESSOR
            predicted_target = self.raw_materials_regressor.predict(input_df)[0]

            # Simplified lookup for other details; this could also be a model
            action_details = {
                'Increase pre-drying time': {'savings': 0.10, 'param': 'moisture_content'},
                'Adjust crusher settings': {'savings': 0.08, 'param': 'particle_size'}
            }
            estimated_savings = action_details.get(predicted_action, {}).get('savings', 0.0)
            parameter = action_details.get(predicted_action, {}).get('param', 'N/A')
            current_val = getattr(data, parameter) if hasattr(data, parameter) else input_df['psd_1'].iloc[0]

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=float(predicted_target),
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.HIGH if confidence > 0.8 else OptimizationLevel.MEDIUM,
                estimated_savings=estimated_savings,
                implementation_time=30
            ))

        # 4. Calculate DYNAMIC summary scores
        expected_improvement = estimated_savings * confidence

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.RAW_MATERIALS,
            recommendations=recommendations,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            energy_savings=estimated_savings,
            quality_improvement=0.05 * confidence,  # Base other scores on confidence
            sustainability_score=0.78 * confidence
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_grinding(self, data: GrindingData) -> OptimizationResult:
        """Optimize grinding process for energy efficiency"""
        if not self.grinding_model:
            raise RuntimeError("Grinding optimization model is not available.")

        # 1. Prepare input data
        input_df = pd.DataFrame([data.model_dump()])[self.grinding_features]

        # 2. Get Action and Confidence from CLASSIFIER
        probabilities = self.grinding_model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = self.grinding_encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Grinding Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        recommendations = []
        estimated_savings = 0.0

        if predicted_action != 'none':
            # 3. Get DYNAMIC Target Value from REGRESSOR
            predicted_target = self.grinding_regressor.predict(input_df)[0]

            action_details = {
                'Optimize mill speed and grinding media': {'savings': 0.12, 'param': 'energy_consumption'},
                'Adjust separator settings': {'savings': 0.05, 'param': 'product_fineness'}
            }
            estimated_savings = action_details.get(predicted_action, {}).get('savings', 0.0)
            parameter = action_details.get(predicted_action, {}).get('param', 'N/A')
            current_val = getattr(data, parameter)

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=float(predicted_target),
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.HIGH if confidence > 0.8 else OptimizationLevel.MEDIUM,
                estimated_savings=estimated_savings,
                implementation_time=30
            ))

        # 4. Calculate DYNAMIC summary scores
        expected_improvement = estimated_savings * confidence

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.GRINDING,
            recommendations=recommendations,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            energy_savings=estimated_savings,
            quality_improvement=0.08 * confidence,
            sustainability_score=0.80 * confidence
        )
        
        self.optimization_history.append(result)
        return result    
    
    async def optimize_clinkerization(self, data: ClinkerizationData) -> OptimizationResult:
        """Optimize clinkerization process for energy and quality"""
        if not self.clinkerization_model:
            raise RuntimeError("Clinkerization optimization model is not available.")

        # 1. Prepare input data
        input_df = pd.DataFrame([data.model_dump()])[self.clinkerization_features]

        # 2. Get Action and Confidence from CLASSIFIER
        probabilities = self.clinkerization_model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = self.clinkerization_encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Clinkerization Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        recommendations = []
        estimated_savings = 0.0

        if predicted_action != 'none':
            # 3. Get DYNAMIC Target Value from REGRESSOR
            predicted_target = self.clinkerization_regressor.predict(input_df)[0]

            action_details = {
                'Reduce temperature by 20-30°C': {'savings': 0.10, 'param': 'kiln_temperature'},
                'Increase alternative fuel usage': {'savings': 0.18, 'param': 'alternative_fuel_ratio'}
            }
            estimated_savings = action_details.get(predicted_action, {}).get('savings', 0.0)
            parameter = action_details.get(predicted_action, {}).get('param', 'N/A')
            current_val = getattr(data, parameter)

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=float(predicted_target),
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.CRITICAL if confidence > 0.9 else OptimizationLevel.HIGH,
                estimated_savings=estimated_savings,
                implementation_time=60 
            ))

        # 4. Calculate DYNAMIC summary scores
        expected_improvement = estimated_savings * confidence

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.CLINKERIZATION,
            recommendations=recommendations,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            energy_savings=estimated_savings,
            quality_improvement=0.06 * confidence,
            sustainability_score=0.85 * confidence
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_quality(self, data: QualityData) -> OptimizationResult:
        """Optimize product quality consistency"""
        if not self.quality_model:
            raise RuntimeError("Quality optimization model is not available.")

        # 1. ROBUST DATA PREPARATION
        input_data = data.model_dump()
        input_series = pd.Series(input_data)
        input_df = input_series.to_frame().T
        # The variable name was 'quality_feature_names' in your __init__
        input_df = input_df[self.quality_feature_names] 
        input_df = input_df.astype(float)

        # 2. Get Action and Confidence from CLASSIFIER
        probabilities = self.quality_model.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = self.quality_encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Quality Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        recommendations = []
        estimated_savings = 0.0

        if predicted_action != 'none':
            # 3. Get DYNAMIC Target Value from REGRESSOR
            predicted_target = self.quality_regressor.predict(input_df)[0]

            # In this case, 'savings' refers to cost reduction from avoiding off-spec product
            action_details = {
                'Implement real-time quality monitoring': {'savings': 0.15, 'param': 'compressive_strength'}
            }
            estimated_savings = action_details.get(predicted_action, {}).get('savings', 0.0)
            parameter = action_details.get(predicted_action, {}).get('param', 'N/A')
            # We use compressive strength as a proxy for the current value indicator
            current_val = getattr(data, parameter)

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=float(predicted_target),
                impact="Improve quality consistency and reduce off-spec product.",
                priority=OptimizationLevel.HIGH,
                estimated_savings=estimated_savings,
                implementation_time=90
            ))

        # 4. Calculate DYNAMIC summary scores
        expected_improvement = estimated_savings * confidence

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.QUALITY,
            recommendations=recommendations,
            expected_improvement=expected_improvement,
            confidence_score=confidence,
            energy_savings=0, # This optimization is quality-focused
            quality_improvement=expected_improvement,
            sustainability_score=0.70 * confidence
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


