# ~/Documents/cement-operations-optimization/src/services/llm_service.py
import google.generativeai as genai
import logging
from ..schemas.data_models import OptimizationResult

# Configure your API key (best to use environment variables)
GOOGLE_API_KEY = "AIzaSyBbUAFKJynjHuEfynvARR-0IId4tStGU3Y" 
genai.configure(api_key=GOOGLE_API_KEY)

logger = logging.getLogger(__name__)

async def generate_supervisor_report(result: OptimizationResult) -> str:
    """
    Takes an optimization result and generates a human-readable summary
    using the Gemini LLM.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        prompt = f"""
        You are an expert Cement Plant Shift Supervisor AI. Your task is to write a concise, clear, and professional summary based on the following JSON data from our optimization engine.

        **Instructions:**
        1. Start with a clear headline (e.g., "Operational Summary for Clinkerization").
        2. In a paragraph, explain the current situation in simple terms.
        3. Use a bulleted list to clearly state the recommended actions.
        4. Mention the potential impact (e.g., energy savings, quality improvement).
        5. Maintain a confident and professional tone.

        **Optimization Data:**
        {result.model_dump_json(indent=2)}
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error communicating with LLM: {str(e)}")
        # Return a user-friendly error message or re-raise the exception
        raise ValueError("Failed to generate report from LLM.")