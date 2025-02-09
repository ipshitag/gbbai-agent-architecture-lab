# WIP
import os
import logging
from typing import Annotated
import json
from semantic_kernel.functions import kernel_function
from utils.ml_logging import get_logger
from src.aoai.azure_openai import AzureOpenAIManager
from semantic_kernel.utils.logging import setup_logging
from src.prompts.prompt_manager import PromptManager

# Set up logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)

TRACING_CLOUD_ENABLED = os.getenv("TRAINING_CLOUD_ENABLED") or False

class AIQueryFormulationPlugin:
    """
    A plugin for creating an optimized search query (JSON) for prior authorization,
    using query expansion techniques for clinical data.
    """

    def __init__(self, prompt_manager=None) -> None:
        """
        Initialize the AIQueryFormulationPlugin with the necessary client configurations
        and an optional PromptManager for retrieving system/user prompts.

        Args:
            prompt_manager: An instance of your PromptManager (or None if not used).
        """
        self.logger = get_logger(
            name="AIQueryFormulationPlugin",
            level=logging.DEBUG,
            tracing_enabled=TRACING_CLOUD_ENABLED
        )
        if prompt_manager is None:
            self.prompt_manager = PromptManager()

        try:
            azure_openai_chat_deployment_id = os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
            azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")

            if not all([azure_openai_chat_deployment_id, azure_openai_key, azure_endpoint]):
                raise ValueError("One or more environment variables for OpenAI are missing.")

            self.azure_openai_client = AzureOpenAIManager(
                api_key=azure_openai_key,
                completion_model_name=azure_openai_chat_deployment_id,
                azure_endpoint=azure_endpoint,
            )

            # Inline fallback system prompt if no PromptManager is used
            self.DEFAULT_SYSTEM_PROMPT = """
            You are an expert in optimizing prior authorization searches using query expansion techniques for clinical data.

            ## Task:
            Your goal is to construct an optimized search query based on the provided Diagnosis and Medical Justification
            and Plan for Treatment or Request for Prior Authorization policies. Use query expansion to generate similar terms,
            synonyms, and related medical concepts, improving retrieval recall and ensuring accurate policy matching in Azure AI Search.

            ## Step-by-Step Instructions:
            1. Focus on the Key Elements:
               - Diagnosis and Medical Justification
               - Plan for Treatment or Request for Prior Authorization (medication, code, dosage, duration, rationale)
            2. Apply Query Expansion Techniques:
               - Generate synonyms and related terms for both diagnosis and medication/procedure.
               - Include relevant medical codes if provided, otherwise proceed with synonyms.
            3. Construct a Focused Yet Comprehensive Query:
               - Combine key elements into a cohesive query string.
               - Use synonyms to maximize recall, but ensure you maintain precision.
            4. Handle Insufficient Information:
               - If either the Diagnosis or Medication is missing, return: {"optimized_query": "Need more information to construct the query."}
            5. Return the final query in JSON:
               {
                 "optimized_query": "<your constructed query here>"
               }
            """

            self.logger.info("OpenAI client initialized successfully.")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise e
        
    @kernel_function(
        name="generate_expanded_query",
        description="Creates an optimized search query in JSON format using query expansion techniques."
    )
    async def generate_expanded_query(
        self,
        diagnosis: Annotated[str, "Diagnosis and Medical Justification"],
        medication_or_procedure: Annotated[str, "Medication or Procedure"],
        code: Annotated[str, "Relevant Code (if any)"],
        dosage: Annotated[str, "Dosage or Plan"],
        duration: Annotated[str, "Treatment Duration"],
        rationale: Annotated[str, "Clinical Rationale for Treatment"]
    ) -> Annotated[dict, "A JSON object containing the 'optimized_query' for prior authorization searches."]:
        """
        Generate an expanded prior authorization search query JSON using the provided clinical details.
        
        :param diagnosis: The diagnosis or medical justification.
        :param medication_or_procedure: The medication name or procedure.
        :param code: Relevant code (ICD-10, CPT, etc.), if any.
        :param dosage: Dosage or plan.
        :param duration: Treatment duration.
        :param rationale: Clinical rationale or justification.
        :return: A JSON object containing the optimized query { "optimized_query": "...query..." }
        """
        try:
            self.logger.info("Creating expanded query for prior authorization...")

            system_prompt = self.prompt_manager.get_prompt("formulator_system_prompt.jinja")
            user_prompt = self.prompt_manager.create_prompt_formulator_user(
                diagnosis=diagnosis,
                medication_or_procedure=medication_or_procedure,
                code=code,
                dosage=dosage,
                duration=duration,
                rationale=rationale
            )
    
            if not diagnosis.strip() or not medication_or_procedure.strip():
                return {"optimized_query": "Need more information to construct the query."}

            response = await self.azure_openai_client.generate_chat_response(
                query=user_prompt,
                system_message_content=system_prompt,
                conversation_history=[],
                response_format="json_object",
                max_tokens=1000,
                temperature=0.7
            )

            llm_reply = response["response"]

            # if not llm_reply.startswith("{"):
            #     llm_reply = f'{{"optimized_query":"{llm_reply}"}}'

            # verified_json = self.verify_json_structure(llm_reply)
            self.logger.info(f"Generated expanded query: {llm_reply}")
            return llm_reply

        except Exception as e:
            self.logger.error(f"Error creating expanded query: {e}")
            # If something goes wrong, return a fallback JSON
            return {"optimized_query": "Unable to generate query due to an error."}

    def verify_json_structure(self, json_string: str) -> dict:
        """
        Verify the JSON structure to ensure it contains the 'optimized_query' key.
        
        :param json_string: The JSON string to verify.
        :return: A correctly structured JSON object.
        """
        try:
            json_obj = json.loads(json_string)
            if "optimized_query" not in json_obj:
                json_obj = {"optimized_query": json_string}
            return json_obj
        except json.JSONDecodeError:
            return {"optimized_query": "Invalid JSON structure."}
