from google import genai
from google.genai import types

class MedicalReportGenerator:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-3-flash-preview" 

    def generate_final_report(self, severity_score, status, context):
        prompt = f"""
        ROLE: Medical Advisory Assistant.
        TASK: Synthesize a professional clinical report based on deep learning analysis and clinical guidelines.

        DATA INPUTS:
        - Computed Severity Score: {severity_score}%
        - Classification: {status}

        CLINICAL CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Summarize the detected severity level.
        2. Provide clinical actions based strictly on the provided context chunks.
        3. Include specific dosages, thresholds, or therapy protocols mentioned in the context.
        4. Maintain a formal clinical tone.
        5. Include a standard medical disclaimer at the end.

        FORMAT: Output must be in Markdown.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error during report generation: {str(e)}"