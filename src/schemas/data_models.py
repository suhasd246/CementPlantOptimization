# ~/Documents/cement-operations-optimization/src/models/data_models.py
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

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
    timestamp: datetime = Field(
        ..., 
        example="2025-09-21T13:30:00.123Z"
    )
    limestone_quality: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.92
    )
    clay_content: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.15
    )
    iron_ore_grade: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.78
    )
    moisture_content: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.18
    )
    particle_size_distribution: List[float] = Field(
        ..., 
        min_items=5, 
        max_items=5,
        example=[0.15, 0.35, 0.25, 0.20, 0.05]
    )
    temperature: float = Field(
        ..., 
        ge=0.0, 
        le=100.0,
        example=25.5
    )
    flow_rate: float = Field(
        ..., 
        ge=0.0,
        example=450.7
    )

class GrindingData(BaseModel):
    timestamp: datetime = Field(
        ...,
        example="2025-09-21T13:45:10.543Z"
    )
    mill_power: float = Field(
        ..., 
        ge=0.0,
        example=4850.5
    )
    feed_rate: float = Field(
        ..., 
        ge=0.0,
        example=215.7
    )
    product_fineness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.92
    )
    energy_consumption: float = Field(
        ..., 
        ge=0.0,
        example=4275.0
    )
    temperature: float = Field(
        ..., 
        ge=0.0, 
        le=200.0,
        example=110.2
    )
    vibration_level: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        example=3.8
    )
    noise_level: float = Field(
        ..., 
        ge=0.0, 
        le=120.0,
        example=99.5
    )

class ClinkerizationData(BaseModel):
    timestamp: datetime = Field(
        ...,
        example="2025-09-21T14:00:00.000Z"
    )
    kiln_temperature: float = Field(
        ..., 
        ge=1000.0, 
        le=1600.0,
        example=1485.0
    )
    residence_time: float = Field(
        ..., 
        ge=0.0,
        example=28.5  # in minutes
    )
    fuel_consumption: float = Field(
        ..., 
        ge=0.0,
        example=3250.0 # in MJ/ton of clinker
    )
    alternative_fuel_ratio: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.25
    )
    exhaust_gas_temperature: float = Field(
        ..., 
        ge=0.0, 
        le=500.0,
        example=355.8
    )
    oxygen_content: float = Field(
        ..., 
        ge=0.0, 
        le=25.0,
        example=2.8 # in percent
    )
    clinker_quality: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.96 # A score representing quality (e.g., based on free lime)
    )

class QualityData(BaseModel):
    timestamp: datetime = Field(
        ...,
        example="2025-09-21T18:00:00.000Z"
    )
    product_type: str = Field(
        ...,
        example="OPC-53" # Ordinary Portland Cement, 53 Grade
    )
    compressive_strength: float = Field(
        ..., 
        ge=0.0,
        example=55.8 # in Megapascals (MPa)
    )
    fineness: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.94 # A score representing fineness (e.g., Blaine)
    )
    consistency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.31 # Standard consistency as a ratio
    )
    setting_time: float = Field(
        ..., 
        ge=0.0,
        example=115.0 # Initial setting time in minutes
    )
    temperature: float = Field(
        ..., 
        ge=0.0, 
        le=100.0,
        example=27.5 # Curing temperature in Celsius
    )
    humidity: float = Field(
        ..., 
        ge=0.0, 
        le=100.0,
        example=65.0 # Curing humidity in percent
    )
    gypsum_added: float = Field(
        ..., 
        ge=0.0, 
        le=10.0,
        example=4.5
    )

class OptimizationRecommendation(BaseModel):
    parameter: str = Field(
        ...,
        example="moisture_content"
    )
    action: str = Field(
        ...,
        example="Increase pre-drying time"
    )
    current_value: float = Field(
        ...,
        example=0.18
    )
    target_value: float = Field(
        ...,
        example=0.12
    )
    impact: str = Field(
        ...,
        example="Reduce grinding energy consumption by 8-12%"
    )
    priority: OptimizationLevel = Field(
        ...,
        example=OptimizationLevel.HIGH
    )
    estimated_savings: float = Field(
        ..., 
        ge=0.0,
        example=0.11 # Represents an 11% cost or energy saving
    )
    implementation_time: int = Field(
        ..., 
        ge=0,
        example=30 # in minutes
    )

class OptimizationResult(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime
    process: ProcessType
    recommendations: List[OptimizationRecommendation]
    expected_improvement: float = Field(..., ge=0.0, le=1.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    energy_savings: float = Field(..., ge=0.0)
    quality_improvement: float = Field(..., ge=0.0, le=1.0)
    sustainability_score: float = Field(..., ge=0.0, le=1.0)
    report: Optional[str] = None

    # This configuration provides a complete example for the entire model
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "timestamp": "2025-09-21T14:15:00Z",
                "process": "raw_materials",
                "recommendations": [
                    {
                        "parameter": "moisture_content",
                        "action": "Increase pre-drying time",
                        "current_value": 0.18,
                        "target_value": 0.12,
                        "impact": "Reduce grinding energy consumption by 8-12%",
                        "priority": "high",
                        "estimated_savings": 0.11,
                        "implementation_time": 30
                    }
                ],
                "expected_improvement": 0.12,
                "confidence_score": 0.95,
                "energy_savings": 0.11,
                "quality_improvement": 0.05,
                "sustainability_score": 0.78,
                "report": "Operational Summary for Raw Materials:\nThe system has detected high moisture content in the raw materials, which is impacting energy efficiency. It is recommended to increase the pre-drying time to reduce moisture before grinding."
            }
        }
    )


class PlantStatus(BaseModel):
    timestamp: datetime = Field(
        ...,
        example="2025-09-21T14:30:00Z"
    )
    overall_efficiency: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.88
    )
    energy_consumption: float = Field(
        ..., 
        ge=0.0,
        example=52300.5 # Example: kWh for the last hour
    )
    quality_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.95
    )
    sustainability_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        example=0.81
    )
    active_recommendations: int = Field(
        ..., 
        ge=0,
        example=3
    )
    critical_alerts: int = Field(
        ..., 
        ge=0,
        example=1
    )

