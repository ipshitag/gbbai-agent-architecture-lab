import os
import logging
from typing import Annotated, List, Dict, Any
from semantic_kernel.functions import kernel_function
from utils.ml_logging import get_logger
from src.aoai.azure_openai import AzureOpenAIManager
from semantic_kernel.utils.logging import setup_logging

# Set up logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)

TRACING_CLOUD_ENABLED = os.getenv("TRAINING_CLOUD_ENABLED") or False

class AIQueryClassificationPlugin:
    """
    A plugin for intelligent query classification, leveraging LLMs for 'keyword' or 'semantic' decisions.
    Now updated to demonstrate separate system and user prompts.
    """

    def __init__(self, prompt_manager=None) -> None:
        """
        Initialize the AIQueryClassificationPlugin with the necessary client configurations
        and (optionally) a PromptManager for retrieving system/user prompts.

        Args:
            prompt_manager: An instance of your PromptManager (or None if not used).
        """
        self.logger = get_logger(
            name="AIQueryClassificationPlugin",
            level=logging.DEBUG,
            tracing_enabled=TRACING_CLOUD_ENABLED
        )
        self.prompt_manager = prompt_manager

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

            # If you are *not* using a PromptManager, you can define system prompts inline:
            self.DEFAULT_SYSTEM_PROMPT = """
            You are an intelligent Query Classification Assistant.
            Your role is to analyze and classify the following search query into 'keyword' or 'semantic'.
            ... (system instructions, chain-of-thought, examples, etc.) ...
            """

            self.logger.info("OpenAI client initialized successfully.")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise e

    @kernel_function(
        name="classify_search_query", 
        description="Classifies a query into 'keyword' or 'semantic' using LLM reasoning."
    )
    async def classify_query(
        self,
        query_text: Annotated[str, "The user's search query to be classified."]
    ) -> Annotated[dict, "A JSON object containing the classification result."]:
        """
        Classify the query as 'keyword', 'semantic', or fallback to 'semantic' if uncertain.
        
        :param query_text: The user's search query.
        :return: A JSON object containing the classification result.
        """
        try:
            self.logger.info(f"Classifying query: {query_text}")
            system_prompt = self.prompt_manager.get_prompt(
                "query_classifier_system_prompt.jinja"
            )
            user_prompt = self.prompt_manager.create_prompt_query_classifier_user(
                query=query_text
            )

            response = await self.azure_openai_client.generate_chat_response(
                query=user_prompt,                
                system_message_content=system_prompt, 
                conversation_history=[],
                response_format="json_object",
                max_tokens=20,
                temperature=0  # ensures more deterministic output
            )

            classification = response["response"].strip().lower()

            if classification not in {"keyword", "semantic"}:
                self.logger.warning(f"Invalid classification: '{classification}', defaulting to 'semantic'.")
                classification = "semantic"

            self.logger.info(f"Query classified as: {classification}")
            return {"classification": classification}

        except Exception as e:
            self.logger.error(f"Error during query classification: {e}")
            # fallback to 'semantic' if anything goes wrong
            return {"classification": "semantic"}