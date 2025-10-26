# ~/src/services/optimization_engine.py
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
        
        # --- Define Feature Lists ---
        self.raw_materials_features = [
            'limestone_quality', 'clay_content', 'iron_ore_grade', 'moisture_content',
            'temperature', 'flow_rate', 
            'psd_1', 'psd_2', 'psd_3', 'psd_4', 'psd_5', 'psd_mean', 'psd_std_dev'
        ]
        self.grinding_features = [
            'mill_power', 'feed_rate', 'product_fineness', 'energy_consumption',
            'temperature', 'vibration_level', 'noise_level', 'specific_energy'
        ]
        self.clinkerization_features = [
            'kiln_temperature', 'residence_time', 'fuel_consumption', 
            'alternative_fuel_ratio', 'clinker_quality', 
            'exhaust_gas_temperature', 'oxygen_content'
        ]
        self.quality_features = [
            'compressive_strength', 'fineness', 'consistency', 
            'setting_time', 'temperature', 'humidity', 'gypsum_added'
        ]

        # --- Load All Models ---
        self.models = {}
        try:
            logger.info("Loading all 20 optimization artifacts...")
            
            # Load Raw Materials Models
            self.models['raw_materials_model'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_model.pkl'))
            self.models['raw_materials_encoder'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_encoder.pkl'))
            self.models['raw_materials_target_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_target_regressor.pkl'))
            self.models['raw_materials_energy_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_energy_regressor.pkl'))
            self.models['raw_materials_quality_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_quality_regressor.pkl'))
            self.models['raw_materials_sustainability_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'raw_materials_sustainability_regressor.pkl'))
            
            # Load Grinding Models
            self.models['grinding_model'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_model.pkl'))
            self.models['grinding_encoder'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_encoder.pkl'))
            self.models['grinding_target_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_target_regressor.pkl'))
            self.models['grinding_energy_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_energy_regressor.pkl'))
            self.models['grinding_quality_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_quality_regressor.pkl'))
            self.models['grinding_sustainability_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'grinding_sustainability_regressor.pkl'))

            # Load Clinkerization Models
            self.models['clinkerization_model'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_model.pkl'))
            self.models['clinkerization_encoder'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_encoder.pkl'))
            self.models['clinkerization_target_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_target_regressor.pkl'))
            self.models['clinkerization_energy_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_energy_regressor.pkl'))
            self.models['clinkerization_quality_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_quality_regressor.pkl'))
            self.models['clinkerization_sustainability_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'clinkerization_sustainability_regressor.pkl'))

            # Load Quality Models
            self.models['quality_model'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_model.pkl'))
            self.models['quality_encoder'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_encoder.pkl'))
            self.models['quality_target_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_target_regressor.pkl'))
            self.models['quality_energy_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_energy_regressor.pkl'))
            self.models['quality_quality_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_quality_regressor.pkl'))
            self.models['quality_sustainability_regressor'] = joblib.load(os.path.join(ARTIFACTS_DIR, 'quality_sustainability_regressor.pkl'))

            logger.info("✅ All 24 optimization artifacts loaded successfully.")
            
        except FileNotFoundError as e:
            logger.error(f"FATAL: A model file was not found. Please re-run training script. Error: {e}")
            self.models = {} # Clear models so the app fails gracefully
    
    def _get_model(self, name: str):
        """Helper to safely get a model."""
        model = self.models.get(name)
        if model is None:
            logger.error(f"Model '{name}' is not loaded.")
            raise RuntimeError(f"Optimization model '{name}' is not available.")
        return model

    async def optimize_raw_materials(self, data: RawMaterialData) -> OptimizationResult:
        """Optimize raw material processing (New Ideology)"""
        
        # 1. Prepare input data
        input_data = data.model_dump()
        psd_features = input_data.pop('particle_size_distribution')
        
        # Flatten PSD
        psd_dict = {f'psd_{i+1}': val for i, val in enumerate(psd_features)}
        input_data.update(psd_dict)
        
        # Add summary PSD features
        input_data['psd_mean'] = np.mean(psd_features)
        input_data['psd_std_dev'] = np.std(psd_features)
        
        input_df = pd.DataFrame([input_data])[self.raw_materials_features]

        # 2. Get Action from CLASSIFIER (Diagnosis)
        classifier = self._get_model('raw_materials_model')
        encoder = self._get_model('raw_materials_encoder')
        probabilities = classifier.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        # 3. DYNAMICALLY PREDICT KPIs
        predicted_energy = float(self._get_model('raw_materials_energy_regressor').predict(input_df)[0])
        predicted_quality = float(self._get_model('raw_materials_quality_regressor').predict(input_df)[0])
        predicted_sustain = float(self._get_model('raw_materials_sustainability_regressor').predict(input_df)[0])
        
        if predicted_action == 'maintain_parameters':
            predicted_energy = 0.0
            predicted_quality = 0.0
            # Keep predicted_sustain as the 'base' score
        
        # 4. Get Target Value (if action is needed)
        recommendations = []
        if predicted_action != 'maintain_parameters':
            predicted_target = float(self._get_model('raw_materials_target_regressor').predict(input_df)[0])
            
            param_map = {
                'Increase pre-drying time': 'moisture_content',
                'Adjust crusher settings': 'particle_size' ,
                'CRITICAL: Reduce moisture, high flow rate detected': 'moisture_content',
                'HIGH: Adjust crusher settings, low-quality mix': 'particle_size'
            }
            parameter = param_map.get(predicted_action, 'N/A')
            current_val = getattr(data, parameter) if hasattr(data, parameter) else input_df['psd_1'].iloc[0]

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=predicted_target,
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.HIGH if confidence > 0.8 else OptimizationLevel.MEDIUM,
                estimated_savings=predicted_energy,
                implementation_time=30
            ))

        # 5. Calculate FINAL summary scores
        expected_improvement = (predicted_energy + predicted_quality) / 2.0
        
        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.RAW_MATERIALS,
            recommendations=recommendations,
            expected_improvement=expected_improvement * confidence,
            confidence_score=confidence,
            energy_savings=predicted_energy,
            quality_improvement=predicted_quality,
            sustainability_score=predicted_sustain
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_grinding(self, data: GrindingData) -> OptimizationResult:
        """Optimize grinding process (New Ideology)"""
        
        # 1. Prepare input data
        input_data = data.model_dump()
        input_data['specific_energy'] = input_data['energy_consumption'] / (input_data['feed_rate'] + 1e-6)
        input_df = pd.DataFrame([input_data])[self.grinding_features]

        # 2. Get Action from CLASSIFIER (Diagnosis)
        classifier = self._get_model('grinding_model')
        encoder = self._get_model('grinding_encoder')
        probabilities = classifier.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Grinding Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        # 3. DYNAMICALLY PREDICT KPIs
        predicted_energy = float(self._get_model('grinding_energy_regressor').predict(input_df)[0])
        predicted_quality = float(self._get_model('grinding_quality_regressor').predict(input_df)[0])
        predicted_sustain = float(self._get_model('grinding_sustainability_regressor').predict(input_df)[0])
        
        if predicted_action == 'maintain_parameters':
            predicted_energy = 0.0
            predicted_quality = 0.0

        # 4. Get Target Value (if action is needed)
        recommendations = []
        if predicted_action != 'maintain_parameters':
            predicted_target = float(self._get_model('grinding_target_regressor').predict(input_df)[0])

            param_map = {
                'Optimize mill speed and grinding media': 'specific_energy',
                'Adjust separator settings': 'product_fineness',
                'ALERT: Inspect mill for high vibration': 'vibration_level',
                'Adjust separator settings, energy high': 'product_fineness'
            }
            parameter = param_map.get(predicted_action, 'N/A')
            current_val = input_df[parameter].iloc[0] # Get value from the DataFrame

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=predicted_target,
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.HIGH if confidence > 0.8 else OptimizationLevel.MEDIUM,
                estimated_savings=predicted_energy,
                implementation_time=30
            ))

        # 5. Calculate DYNAMIC summary scores
        expected_improvement = (predicted_energy + predicted_quality) / 2.0
        
        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.GRINDING,
            recommendations=recommendations,
            expected_improvement=expected_improvement * confidence,
            confidence_score=confidence,
            energy_savings=predicted_energy,
            quality_improvement=predicted_quality,
            sustainability_score=predicted_sustain
        )
        
        self.optimization_history.append(result)
        return result    
    
    async def optimize_clinkerization(self, data: ClinkerizationData) -> OptimizationResult:
        """Optimize clinkerization process (New Ideology)"""
        
        # 1. Prepare input data
        input_df = pd.DataFrame([data.model_dump()])[self.clinkerization_features]

        # 2. Get Action from CLASSIFIER (Diagnosis)
        classifier = self._get_model('clinkerization_model')
        encoder = self._get_model('clinkerization_encoder')
        probabilities = classifier.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Clinkerization Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        # 3. DYNAMICALLY PREDICT KPIs
        predicted_energy = float(self._get_model('clinkerization_energy_regressor').predict(input_df)[0])
        predicted_quality = float(self._get_model('clinkerization_quality_regressor').predict(input_df)[0])
        predicted_sustain = float(self._get_model('clinkerization_sustainability_regressor').predict(input_df)[0])
        
        if predicted_action == 'maintain_parameters':
            predicted_energy = 0.0
            predicted_quality = 0.0

        # 4. Get Target Value (if action is needed)
        recommendations = []
        if predicted_action != 'maintain_parameters':
            predicted_target = float(self._get_model('clinkerization_target_regressor').predict(input_df)[0])

            param_map = {
                'Reduce temperature by 20-30°C': 'kiln_temperature',
                'Increase alternative fuel usage': 'alternative_fuel_ratio',
                'CRITICAL: Stabilize kiln, clinker quality low': 'clinker_quality',
                'Optimize fuel mix: Increase alt fuel, reduce temp': 'alternative_fuel_ratio'
            }
            parameter = param_map.get(predicted_action, 'N/A')
            current_val = getattr(data, parameter)

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=predicted_target,
                impact=f"Dynamic adjustment based on live data.",
                priority=OptimizationLevel.CRITICAL if confidence > 0.9 else OptimizationLevel.HIGH,
                estimated_savings=predicted_energy,
                implementation_time=60 
            ))

        # 5. Calculate DYNAMIC summary scores
        expected_improvement = (predicted_energy + predicted_quality) / 2.0

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.CLINKERIZATION,
            recommendations=recommendations,
            expected_improvement=expected_improvement * confidence,
            confidence_score=confidence,
            energy_savings=predicted_energy,
            quality_improvement=predicted_quality,
            sustainability_score=predicted_sustain
        )
        
        self.optimization_history.append(result)
        return result
    
    async def optimize_quality(self, data: QualityData) -> OptimizationResult:
        """Optimize product quality consistency (New Ideology)"""

        # 1. Prepare input data
        input_data = data.model_dump()
        input_df = pd.DataFrame([input_data])[self.quality_features]

        # 2. Get Action from CLASSIFIER (Diagnosis)
        classifier = self._get_model('quality_model')
        encoder = self._get_model('quality_encoder')
        probabilities = classifier.predict_proba(input_df)[0]
        confidence = float(np.max(probabilities))
        action_encoded = np.argmax(probabilities)
        predicted_action = encoder.inverse_transform([action_encoded])[0]
        
        logger.info(f"AI Quality Prediction: '{predicted_action}' with {confidence:.2%} confidence.")

        # 3. DYNAMICALLY PREDICT KPIs
        predicted_energy = float(self._get_model('quality_energy_regressor').predict(input_df)[0])
        predicted_quality = float(self._get_model('quality_quality_regressor').predict(input_df)[0])
        predicted_sustain = float(self._get_model('quality_sustainability_regressor').predict(input_df)[0])
        
        if predicted_action == 'maintain_parameters':
            predicted_energy = 0.0
            predicted_quality = 0.0

        # 4. Get Target Value (if action is needed)
        recommendations = []
        if predicted_action != 'maintain_parameters':
            predicted_target = float(self._get_model('quality_target_regressor').predict(input_df)[0])

            param_map = {
                'Reduce gypsum dosing': 'gypsum_added',
                'Increase gypsum dosing': 'gypsum_added',
                'Adjust gypsum for low strength': 'gypsum_added'
            }
            parameter = param_map.get(predicted_action, 'N/A')
            current_val = getattr(data, parameter)

            recommendations.append(OptimizationRecommendation(
                parameter=parameter,
                action=predicted_action,
                current_value=current_val,
                target_value=predicted_target,
                impact="Improve quality consistency and reduce off-spec product.",
                priority=OptimizationLevel.HIGH,
                estimated_savings=predicted_energy,
                implementation_time=90
            ))

        # 5. Calculate DYNAMIC summary scores
        expected_improvement = (predicted_energy + predicted_quality) / 2.0

        result = OptimizationResult(
            timestamp=datetime.now(),
            process=ProcessType.QUALITY,
            recommendations=recommendations,
            expected_improvement=expected_improvement * confidence,
            confidence_score=confidence,
            energy_savings=predicted_energy, 
            quality_improvement=predicted_quality,
            sustainability_score=predicted_sustain
        )
        
        self.optimization_history.append(result)
        return result
    
    async def get_plant_status(self) -> Dict[str, Any]:
        """Get overall plant status based on recent history"""
        
        # Return a sensible default if no history exists
        default_status = {
            "timestamp": datetime.now(),
            "overall_efficiency": 0.82,
            "energy_consumption": 2000.0, # This is a placeholder, could be a live value
            "quality_score": 0.85,
            "sustainability_score": 0.75,
            "active_recommendations": 0,
            "critical_alerts": 0
        }

        if not self.optimization_history:
            return default_status
        
        # Get results from the last hour
        recent_results = [r for r in self.optimization_history 
                         if (datetime.now() - r.timestamp).total_seconds() < 3600]
        
        if not recent_results:
            # If no recent results, return last known status or default
            last_result = self.optimization_history[-1]
            return {
                "timestamp": last_result.timestamp,
                "overall_efficiency": last_result.expected_improvement,
                "energy_consumption": 2000.0, # Placeholder
                "quality_score": last_result.quality_improvement,
                "sustainability_score": last_result.sustainability_score,
                "active_recommendations": 0,
                "critical_alerts": 0
            }
        
        # Aggregate recent results
        overall_efficiency = np.mean([r.expected_improvement for r in recent_results if r.expected_improvement > 0])
        quality_score = np.mean([r.quality_improvement for r in recent_results if r.quality_improvement > 0])
        sustainability_score = np.mean([r.sustainability_score for r in recent_results])
        
        active_recommendations = sum(len(r.recommendations) for r in recent_results)
        critical_alerts = sum(1 for r in recent_results 
                             for rec in r.recommendations 
                             if rec.priority == OptimizationLevel.CRITICAL)
        
        return {
            "timestamp": datetime.now(),
            "overall_efficiency": float(overall_efficiency) if not np.isnan(overall_efficiency) else 0.82,
            "energy_consumption": 2000.0, # Placeholder
            "quality_score": float(quality_score) if not np.isnan(quality_score) else 0.85,
            "sustainability_score": float(sustainability_score) if not np.isnan(sustainability_score) else 0.75,
            "active_recommendations": active_recommendations,
            "critical_alerts": critical_alerts
        }