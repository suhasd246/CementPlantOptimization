# ~/Documents/cement-operations-optimization/src/models/data_models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class ProcessType(str, Enum):
    RAW_MATERIALS = "raw_materials"
    GRINDING = "grinding"
    CLINKERIZATION = "clinkerization"
    QUALITY = "quality"
    UTILITIES = "utilities"

class OptimizationLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RawMaterialData(BaseModel):
    timestamp: datetime
    limestone_quality: float = Field(..., ge=0.0, le=1.0)
    clay_content: float = Field(..., ge=0.0, le=1.0)
    iron_ore_grade: float = Field(..., ge=0.0, le=1.0)
    moisture_content: float = Field(..., ge=0.0, le=1.0)
    particle_size_distribution: List[float] = Field(..., min_items=5, max_items=5)
    temperature: float = Field(..., ge=0.0, le=100.0)
    flow_rate: float = Field(..., ge=0.0)

class GrindingData(BaseModel):
    timestamp: datetime
    mill_power: float = Field(..., ge=0.0)
    feed_rate: float = Field(..., ge=0.0)
    product_fineness: float = Field(..., ge=0.0, le=1.0)
    energy_consumption: float = Field(..., ge=0.0)
    temperature: float = Field(..., ge=0.0, le=200.0)
    vibration_level: float = Field(..., ge=0.0, le=10.0)
    noise_level: float = Field(..., ge=0.0, le=120.0)

class ClinkerizationData(BaseModel):
    timestamp: datetime
    kiln_temperature: float = Field(..., ge=1000.0, le=1600.0)
    residence_time: float = Field(..., ge=0.0)
    fuel_consumption: float = Field(..., ge=0.0)
    alternative_fuel_ratio: float = Field(..., ge=0.0, le=1.0)
    clinker_quality: float = Field(..., ge=0.0, le=1.0)
    exhaust_gas_temperature: float = Field(..., ge=0.0, le=500.0)
    oxygen_content: float = Field(..., ge=0.0, le=25.0)

class QualityData(BaseModel):
    timestamp: datetime
    product_type: str
    compressive_strength: float = Field(..., ge=0.0)
    fineness: float = Field(..., ge=0.0, le=1.0)
    consistency: float = Field(..., ge=0.0, le=1.0)
    setting_time: float = Field(..., ge=0.0)
    temperature: float = Field(..., ge=0.0, le=100.0)
    humidity: float = Field(..., ge=0.0, le=100.0)

class OptimizationRecommendation(BaseModel):
    parameter: str
    action: str
    current_value: float
    target_value: float
    impact: str
    priority: OptimizationLevel
    estimated_savings: float = Field(..., ge=0.0)
    implementation_time: int = Field(..., ge=0)  # in minutes

class OptimizationResult(BaseModel):
    id: Optional[str] = None
    timestamp: datetime
    process: ProcessType
    recommendations: List[OptimizationRecommendation]
    expected_improvement: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    energy_savings: float = Field(..., ge=0.0)
    quality_improvement: float = Field(..., ge=0.0, le=1.0)
    sustainability_score: float = Field(..., ge=0.0, le=1.0)

class PlantStatus(BaseModel):
    timestamp: datetime
    overall_efficiency: float = Field(..., ge=0.0, le=1.0)
    energy_consumption: float = Field(..., ge=0.0)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    sustainability_score: float = Field(..., ge=0.0, le=1.0)
    active_recommendations: int = Field(..., ge=0)
    critical_alerts: int = Field(..., ge=0)


