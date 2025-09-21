# ~/Documents/cement-operations-optimization/config/settings.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Settings:
    # Google Cloud Configuration
    PROJECT_ID: str = "cement-optimization"
    REGION: str = "us-central1"
    
    # BigQuery Configuration
    DATASET_ID: str = "cement_plant_data"
    RAW_MATERIALS_TABLE: str = "raw_materials"
    GRINDING_TABLE: str = "grinding"
    CLINKERIZATION_TABLE: str = "clinkerization"
    QUALITY_TABLE: str = "quality"
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_RAW_MATERIALS: str = "raw_materials"
    KAFKA_TOPIC_GRINDING: str = "grinding"
    KAFKA_TOPIC_CLINKERIZATION: str = "clinkerization"
    KAFKA_TOPIC_QUALITY: str = "quality"
    
    # Model Configuration
    MODEL_UPDATE_INTERVAL: int = 3600  # 1 hour
    PREDICTION_INTERVAL: int = 300  # 5 minutes
    
    # Optimization Thresholds
    ENERGY_EFFICIENCY_THRESHOLD: float = 0.8
    QUALITY_VARIANCE_THRESHOLD: float = 0.05
    ALTERNATIVE_FUEL_TARGET: float = 0.35
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    API_WORKERS: int = 4
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./cement_plant.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

# Create the settings instance
settings = Settings()
