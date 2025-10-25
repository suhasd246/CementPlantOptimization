# ~/Documents/cement-operations-optimization/src/services/llm_service.py
import google.generativeai as genai
import logging
from ..schemas.data_models import OptimizationResult

# Configure your API key (best to use environment variables)
GOOGLE_API_KEY = "AIzaSyBfE1HM0qVu0Vw1wa6xJKTmNtP9xQJBiQc" 
genai.configure(api_key=GOOGLE_API_KEY)

logger = logging.getLogger(__name__)

async def generate_supervisor_report(result: OptimizationResult) -> str:
    """
    Takes an optimization result and generates a human-readable summary
    using the Gemini LLM with a robust, one-shot prompt.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        rec = result.recommendations[0] if result.recommendations else None
        if not rec:
            return "**No recommendations available for this process.**"

        # --- REFINED HIGH-INSIGHT ONE-SHOT PROMPT ---
        
        prompt = f"""
        You are an expert Cement Plant Shift Supervisor AI. Your task is to translate technical JSON data into a clear, professional, and *insightful* summary for a non-technical plant operator.

        ---
        **EXAMPLE**

        **JSON Data:**
        {{
          "process": "grinding",
          "recommendations": [
            {{
              "parameter": "mill_power",
              "action": "Optimize mill speed and grinding media",
              "current_value": 4100.0,
              "target_value": 3950.0,
              "priority": "medium",
              "impact": "Reduces energy use without harming product fineness.",
              "estimated_savings": 0.08,
              "implementation_time": 60
            }}
          ],
          "expected_improvement": 0.05,
          "confidence_score": 0.98,
          "energy_savings": 0.08,
          "quality_improvement": 0.02,
          "sustainability_score": 0.81
        }}

        **Generated Report:**
        **Operational Summary for Grinding**

        The AI has identified a key optimization opportunity in the grinding circuit. Our mill power is currently high at **4100.0 units**, while the optimal target for efficiency is **3950.0 units**. This over-consumption is a primary driver of unnecessary energy costs. The system is **98% confident** in this assessment (based on the 'confidence_score').

        **Recommended Actions:**
        * **Action:** Optimize mill speed and grinding media.
        * **Reason:** This will reduce energy use without harming product fineness (from the 'impact' field).
        * **Priority:** MEDIUM
        * **Est. Time:** 60 minutes

        **Expected Impact:**
        * **Overall Process Improvement:** This single change is projected to boost the grinding process efficiency score by **5%** (from 'expected_improvement').
        * **Energy Savings:** We project an **8% reduction** in energy consumption for this stage.
        * **Quality & Sustainability:** This will also lead to a **2% improvement** in product quality and support our plant's high sustainability score (currently at 0.81).
        ---

        **ACTUAL TASK**

        **JSON Data:**
        {result.model_dump_json(indent=2)}

        **Generated Report:**
        """
        
        response = await model.generate_content_async(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error communicating with LLM: {str(e)}")
        return (
            "**Error: Could not generate supervisor's report.**\n\n"
            f"An error occurred while communicating with the AI model: {str(e)}"
        )