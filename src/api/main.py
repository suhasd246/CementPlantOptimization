# ~/Documents/cement-operations-optimization/src/api/main.py
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import uuid
import sys
import os
from src.services.llm_service import generate_supervisor_report

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# Now import the modules
try:
    from src.schemas.data_models import (
        RawMaterialData, GrindingData, ClinkerizationData, QualityData,
        OptimizationResult, PlantStatus
    )
    from src.services.optimization_engine import CementPlantOptimizer
    from src.services.data_pipeline import DataPipeline
    from config.settings import settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Please run this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cement Plant Optimization Platform",
    description="AI-driven platform for autonomous cement plant operations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
optimizer = CementPlantOptimizer()
data_pipeline = DataPipeline(settings)

# In-memory storage for demo
optimization_results = []
plant_status_history = []


@app.post("/generate-report", response_model=str)
async def generate_llm_report(result: OptimizationResult):
    """
    Takes an optimization result and uses an LLM to generate a
    human-readable summary report.
    """
    try:
        # Call the dedicated service to handle the LLM logic
        report = await generate_supervisor_report(result)
        return report
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the report.")


# API Endpoints (without authentication)
@app.post("/optimize/raw-materials", response_model=OptimizationResult)
async def optimize_raw_materials(
    data: RawMaterialData,
    background_tasks: BackgroundTasks
):
    """Optimize raw material processing"""
    try:
        result = await optimizer.optimize_raw_materials(data)
        result.id = str(uuid.uuid4())
        result.report = await generate_llm_report(result)
        optimization_results.append(result)
        
        # Update plant status
        background_tasks.add_task(update_plant_status)
        
        return result
    except Exception as e:
        logger.error(f"Error optimizing raw materials: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/grinding", response_model=OptimizationResult)
async def optimize_grinding(
    data: GrindingData,
    background_tasks: BackgroundTasks
):
    """Optimize grinding process"""
    try:
        result = await optimizer.optimize_grinding(data)
        result.id = str(uuid.uuid4())
        result.report = await generate_llm_report(result)
        optimization_results.append(result)
        
        # Update plant status
        background_tasks.add_task(update_plant_status)
        
        return result
    except Exception as e:
        logger.error(f"Error optimizing grinding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/clinkerization", response_model=OptimizationResult)
async def optimize_clinkerization(
    data: ClinkerizationData,
    background_tasks: BackgroundTasks
):
    """Optimize clinkerization process"""
    try:
        result = await optimizer.optimize_clinkerization(data)
        result.id = str(uuid.uuid4())
        result.report = await generate_llm_report(result)
        optimization_results.append(result)
        
        # Update plant status
        background_tasks.add_task(update_plant_status)
        
        return result
    except Exception as e:
        logger.error(f"Error optimizing clinkerization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/quality", response_model=OptimizationResult)
async def optimize_quality(
    data: QualityData,
    background_tasks: BackgroundTasks
):
    """Optimize product quality"""
    try:
        result = await optimizer.optimize_quality(data)
        result.id = str(uuid.uuid4())
        result.report = await generate_llm_report(result)
        optimization_results.append(result)
        
        # Update plant status
        background_tasks.add_task(update_plant_status)
        
        return result
    except Exception as e:
        logger.error(f"Error optimizing quality: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=PlantStatus)
async def get_plant_status():
    """Get overall plant status"""
    try:
        status = await optimizer.get_plant_status()
        plant_status_history.append(status)
        return status
    except Exception as e:
        logger.error(f"Error getting plant status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimization-history")
async def get_optimization_history():
    """Get optimization history"""
    try:
        return optimization_results[-10:]  # Return last 10 results
    except Exception as e:
        logger.error(f"Error getting optimization history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/plant-status-history")
async def get_plant_status_history():
    """Get plant status history"""
    try:
        return plant_status_history[-24:]  # Return last 24 hours
    except Exception as e:
        logger.error(f"Error getting plant status history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cement Plant Optimization Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "http://localhost:8050"
    }

async def update_plant_status():
    """Background task to update plant status"""
    try:
        status = await optimizer.get_plant_status()
        plant_status_history.append(status)
    except Exception as e:
        logger.error(f"Error updating plant status: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Cement Plant Optimization API...")
    print("📚 API Documentation: http://localhost:8080/docs")
    print("🔗 Health Check: http://localhost:8080/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")