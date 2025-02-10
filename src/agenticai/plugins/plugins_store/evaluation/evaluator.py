# WIP

import json
import logging
import os
from typing import Annotated, Any, Dict, List

from semantic_kernel.functions import kernel_function
from semantic_kernel.utils.logging import setup_logging

from src.aoai.azure_openai import AzureOpenAIManager
from src.prompts.prompt_manager import PromptManager
from utils.ml_logging import get_logger

# Set up logging
setup_logging()
logging.getLogger("kernel").setLevel(logging.DEBUG)

TRACING_CLOUD_ENABLED = os.getenv("TRAINING_CLOUD_ENABLED") or False


class AIPolicyEvaluationPlugin:
    """
    A plugin for evaluating policy search results against a user's query about prior authorization.
    Produces a JSON-like object with 'policies', 'reasoning', and 'retry'.
    """

    def __init__(self, prompt_manager=None) -> None:
        """
        Initialize the AIPolicyEvaluationPlugin with the necessary LLM configurations
        and an optional PromptManager for retrieving system/user prompts.

        Args:
            prompt_manager: An instance of your PromptManager (or None if not used).
        """
        self.logger = get_logger(
            name="AIPolicyEvaluationPlugin",
            level=logging.DEBUG,
            tracing_enabled=TRACING_CLOUD_ENABLED,
        )
        if prompt_manager is None:
            self.prompt_manager = PromptManager()

        try:
            azure_openai_chat_deployment_id = os.getenv(
                "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID"
            )
            azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
            azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")

            if not all(
                [azure_openai_chat_deployment_id, azure_openai_key, azure_endpoint]
            ):
                raise ValueError(
                    "One or more environment variables for OpenAI are missing."
                )

            self.azure_openai_client = AzureOpenAIManager(
                api_key=azure_openai_key,
                completion_model_name=azure_openai_chat_deployment_id,
                azure_endpoint=azure_endpoint,
            )

            # Inline fallback system prompt if no PromptManager is used
            self.DEFAULT_SYSTEM_PROMPT = """
            You are a Policy Results Evaluator.
            Evaluate the list of search results—each representing a retrieved policy—and determine
            which ones most accurately and completely address the user’s query about prior authorization.

            ## Tasks:
            - Compare each policy’s content against the user’s query to verify relevance.
            - Cross-reference policy details to avoid duplicates and select the highest-quality information.
            - Determine if there is enough information to produce a definitive conclusion.
            - Set retry to true if search results are incomplete or insufficient.

            ## Expected JSON Output:
            {
              "policies": [
                // A list of URLs or doc paths that match the user’s query
              ],
              "reasoning": [
                // Short statements explaining each policy’s approval or rejection
              ],
              "retry": false // or true if insufficient data
            }

            Remember:
            - Do not add fields other than 'policies', 'reasoning', and 'retry'.
            - Base approval on whether the policy mentions coverage criteria, conditions, dosage requirements, or other critical details relevant to the query.
            - If insufficient info is found, set "retry" to true.
            """

            self.logger.info("OpenAI client initialized successfully.")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise e

    @kernel_function(
        name="evaluate_policies",
        description="Evaluates a set of policy search results against the user's query, returning a JSON object with 'policies', 'reasoning', and 'retry'.",
    )
    async def evaluate_policies(
        self,
        query: Annotated[str, "The user's prior authorization query."],
        search_results: Annotated[
            List[Dict[str, Any]],
            "A list of policy search results (ID, Path, Content, etc.).",
        ],
    ) -> Annotated[
        dict, "A JSON object with 'policies', 'reasoning', and 'retry' fields."
    ]:
        """
        Evaluate each retrieved policy against the user's query.

        :param query: The user's question (e.g., "What is the prior authorization policy for Epidiolex...?").
        :param search_results: A list of dictionary items representing policy search results.
        :return: A JSON object:
            {
              "policies": [...],
              "reasoning": [...],
              "retry": false
            }
        """
        try:
            self.logger.info("Evaluating policy search results...")

            system_prompt = self.prompt_manager.get_prompt(
                "evaluator_system_prompt.jinja"
            )
            user_prompt = self.prompt_manager.create_prompt_evaluator_user(
                query=query, search_results=search_results
            )
            response = await self.azure_openai_client.generate_chat_response(
                query=user_prompt,
                system_message_content=system_prompt,
                conversation_history=[],
                response_format="json_object",
                max_tokens=3000,
                temperature=0.2,
            )

            llm_reply = response["response"]

            self.logger.info(f"Evaluation result: {llm_reply}")
            return llm_reply

        except Exception as e:
            self.logger.error(f"Error evaluating policies: {e}")
            # Fallback to a minimal JSON with "retry" = true
            return {
                "policies": [],
                "reasoning": ["Error during evaluation"],
                "retry": True,
            }

    # def verify_json_structure(self, json_string: str) -> dict:
    #     """
    #     Verify the JSON structure to ensure it contains the 'policies', 'reasoning', and 'retry' keys.

    #     :param json_string: The JSON string to verify.
    #     :return: A correctly structured JSON object.
    #     """
    #     try:
    #         json_obj = json.loads(json_string)
    #         if not all(key in json_obj for key in ["policies", "reasoning", "retry"]):
    #             json_obj = {"policies": [], "reasoning": ["Invalid JSON structure"], "retry": True}
    #         return json_obj
    #     except json.JSONDecodeError:
    #         return {"policies": [], "reasoning": ["Invalid JSON structure"], "retry": True}
