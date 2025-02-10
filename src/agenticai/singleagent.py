import logging
import os
import random
import string
from typing import Any, Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import \
    ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import \
    FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import \
    AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.prompt_execution_settings import \
    PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.utils.logging import setup_logging

from src.agenticai.skills import Skills
from utils.ml_logging import get_logger


class ChatAgent:
    """
    A comprehensive ChatAgent class wrapping reasoning, memory, and plugin functionalities.
    The agent integrates Azure OpenAI chat services, dynamic plugins, and memory capabilities.
    """

    def __init__(
        self,
        service_id: Optional[str] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        description: Optional[str] = None,
        instructions: Optional[str] = None,
        skills: Optional[List[Literal["retrieval", "rewriting", "evaluation"]]] = None,
        planner_config: Optional[PromptExecutionSettings] = None,
        planner_behavior: Optional[FunctionChoiceBehavior] = None,
        tracing_enabled: bool = False,
        azure_openai_key: Optional[str] = None,
        azure_openai_endpoint: Optional[str] = None,
        azure_openai_api_version: Optional[str] = None,
        azure_openai_chat_deployment_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the ChatAgent with optional parameters for service configuration,
        agent instructions, skills to load, and an agentic "planner" configuration/behavior.

        :param service_id: (optional) Azure OpenAI service ID (default: "openai-chat" if not provided).
        :param name: (optional) A friendly name for the agent (for identification/logging).
        :param id: (optional) A unique ID for this agent instance.
        :param description: (optional) A textual description of the agent's role or purpose.
        :param instructions: (optional) High-level instructions or role definition for the agent.
        :param skills: (optional) List of plugin (skill) names to load upon initialization
                       (e.g., ["retrieval", "main"]).
        :param planner_config: (optional) PromptExecutionSettings controlling how the LLM (planner) behaves.
        :param planner_behavior: (optional) A FunctionChoiceBehavior that determines how the agent
                                 invokes or avoids plugin functions (e.g., Auto, NoneInvoke, Required).
        :param tracing_enabled: (optional) Flag to enable more detailed logs (DEBUG level).
        :param azure_openai_key: (optional) Override for the Azure OpenAI key
                                 (otherwise taken from environment variable).
        :param azure_openai_endpoint: (optional) Override for the Azure OpenAI endpoint URI
                                      (otherwise from environment variable).
        :param azure_openai_api_version: (optional) Override for the Azure OpenAI API version
                                         (otherwise from environment variable).
        :param azure_openai_chat_deployment_id: (optional) Override for the Azure OpenAI model deployment name
                                                (otherwise from environment variable).

        :raises ValueError: If required environment variables for Azure OpenAI are missing
                           (and are not passed explicitly).
        """

        load_dotenv()

        self.name = name
        self.tracing_enabled = tracing_enabled
        if id is None:
            self.id = self._generate_8digit_id()
        else:
            self.id = id

        # Logging
        self._setup_logging()
        self.logger = get_logger(
            name=f"ChatAgent-{self.name}" if self.name else "ChatAgent",
            level=10,
        )
        if tracing_enabled:
            pass
        self.description = description
        self.instructions = instructions

        # Immediately record these core pieces of context in the conversation
        self.chat_history = ChatHistory()
        if description:
            self.add_system_message(description)
        if instructions:
            self.add_user_message(instructions)

        self.AZURE_OPENAI_KEY = azure_openai_key or os.getenv("AZURE_OPENAI_KEY")
        self.AZURE_OPENAI_API_ENDPOINT = azure_openai_endpoint or os.getenv(
            "AZURE_OPENAI_API_ENDPOINT"
        )
        self.AZURE_OPENAI_API_VERSION = azure_openai_api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION"
        )
        self.AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID = (
            azure_openai_chat_deployment_id
            or os.getenv("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
        )

        # Define the four required variables in a dictionary
        required_vars = {
            "AZURE_OPENAI_KEY": self.AZURE_OPENAI_KEY,
            "AZURE_OPENAI_API_ENDPOINT": self.AZURE_OPENAI_API_ENDPOINT,
            "AZURE_OPENAI_API_VERSION": self.AZURE_OPENAI_API_VERSION,
            "AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID": self.AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID,
        }

        # Collect any that are missing or empty
        missing_vars = [
            var_name for var_name, var_value in required_vars.items() if not var_value
        ]

        if missing_vars:
            # Print or log the missing variables
            self.logger.error(
                "The following Azure OpenAI settings are missing: %s",
                ", ".join(missing_vars),
            )
            raise ValueError(
                f"Missing the following required Azure OpenAI settings: {', '.join(missing_vars)}. "
                "Check environment variables or pass them explicitly."
            )

        self.service_id = service_id or "openai-chat"
        # Initialize kernel w/ Azure OpenAI service
        self.kernel = self._initialize_kernel(self.service_id)

        self.planner_config = planner_config or self._configure_planner_config()
        self.planner_behavior = (
            planner_behavior or self.planner_config.function_choice_behavior
        )

        # Skill management
        self._skills_manager = Skills(
            parent_directory=os.path.abspath("src/agenticai/plugins/plugins_store")
        )
        if skills:
            self.load_skills(skills)

        # Default KernelArguments storage
        self._default_kernel_arguments: Dict[str, Any] = {}
        self._default_kernel_settings: Optional[
            Union[
                PromptExecutionSettings,
                List[PromptExecutionSettings],
                Dict[str, PromptExecutionSettings],
            ]
        ] = None

        self.chat_completion: Optional[AzureChatCompletion] = None

        # Log the agent creation with agentic planner information
        self.logger.info(
            "Created ChatAgent '%s' (ID: %s) with service '%s', planner_config=%s, planner_behavior=%s, skills=%s",
            self.name or "Unnamed",
            self.id or "NoID",
            self.service_id,
            repr(self.planner_config),
            repr(self.planner_behavior),
            skills or "None",
        )

    @staticmethod
    def _generate_8digit_id() -> str:
        """
        Generate an 8-digit numeric string (e.g., '49382716')
        using random.choices for clarity and flexibility.
        """
        return "".join(random.choices(string.digits, k=8))

    def _initialize_kernel(self, service_id: str) -> Kernel:
        """
        Initialize the kernel with Azure OpenAI services.

        :param service_id: Service ID for the Azure OpenAI integration.
        :return: Configured Kernel instance.
        :raises Exception: If adding or retrieving the Azure ChatCompletion service fails.
        """
        try:
            kernel = Kernel()
            kernel.add_service(
                AzureChatCompletion(
                    service_id=service_id,
                    deployment_name=self.AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID,
                    api_key=self.AZURE_OPENAI_KEY,
                    endpoint=self.AZURE_OPENAI_API_ENDPOINT,
                    api_version=self.AZURE_OPENAI_API_VERSION,
                )
            )
            # Attempt to retrieve the service by type, if available in your environment
            return kernel

        except Exception as e:
            self.logger.error(
                "Failed to initialize the Kernel or retrieve ChatCompletion service: %s",
                e,
                exc_info=True,
            )
            raise

    def _configure_planner_config(self) -> AzureChatPromptExecutionSettings:
        """
        Configure default execution settings for Azure OpenAI prompts.

        :return: Execution settings instance.
        """
        planner_config = AzureChatPromptExecutionSettings()
        planner_config.function_choice_behavior = FunctionChoiceBehavior.Auto()
        return planner_config

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging for the agent.

        :return: Configured Logger instance.
        """
        logging.basicConfig(
            format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        setup_logging()
        logger = get_logger("ChatAgent")
        logger.setLevel(logging.INFO)
        return logger

    def set_planner_execution_settings(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        number_of_responses: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        user: Optional[str] = None,
        function_call: Optional[str] = None,
    ) -> None:
        """
        Configure the AzureChatPromptExecutionSettings for the agent.
        These settings control how the model will generate responses.

        :param temperature: Controls the "creativity" or randomness of the output. Higher values = more random.
            Range is typically [0.0, 2.0]. Default in OpenAI is ~0.7.
        :param top_p: Alternative to temperature; controls the nucleus sampling. 1.0 means no nucleus sampling.
            Lower values limit the model to top tokens accounting for top_p probability mass.
        :param presence_penalty: How much the model penalizes new tokens based on whether they appear in the text so far.
        :param frequency_penalty: How much the model penalizes new tokens based on their frequency in the text so far.
        :param max_tokens: The maximum number of tokens to generate in the output.
        :param number_of_responses: How many chat completions to generate for each prompt.
        :param stop_sequences: A list of strings that will cause the model to stop generating further text.
        :param stream: If True, the model will stream the output in a token-by-token fashion.
        :param user: An optional user identifier for the request.
        :param function_call: If provided, instructs the model to call a specific function or how to handle function calls
            (e.g., "auto", "none", or a specific function name).
        """
        if temperature is not None:
            self.planner_config.temperature = temperature
        if top_p is not None:
            self.planner_config.top_p = top_p
        if presence_penalty is not None:
            self.planner_config.presence_penalty = presence_penalty
        if frequency_penalty is not None:
            self.planner_config.frequency_penalty = frequency_penalty
        if max_tokens is not None:
            self.planner_config.max_tokens = max_tokens
        if number_of_responses is not None:
            self.planner_config.number_of_responses = number_of_responses
        if stop_sequences is not None:
            self.planner_config.stop = stop_sequences
        if user is not None:
            self.planner_config.user = user
        if function_call is not None:
            self.planner_config.function_call = function_call

        self.planner_config.stream = stream

        self.logger.info("Prompt settings updated: %s", self.planner_config.dict())

    def set_planner_behavior(
        self,
        enable_kernel_functions: bool = True,
        max_auto_invoke_attempts: int = 5,
        filters: Optional[Dict[str, List[str]]] = None,
        behavior_type: Optional[str] = None,
    ) -> None:
        """
        Set and configure function choice behavior for the agent.

        :param enable_kernel_functions: Whether to enable kernel functions at all.
        :param max_auto_invoke_attempts: Maximum number of auto-invoke attempts the model can make.
        :param filters: A dict specifying which plugins/functions to include or exclude.
            Example:
                {
                    "included_plugins": ["some_plugin_name"],
                    "excluded_plugins": ["some_other_plugin"],
                    "included_functions": ["functionA"],
                    "excluded_functions": ["functionB"]
                }
        :param behavior_type: The type of function choice behavior.
            Possible values:
                - "Auto": Auto invoke functions if relevant.
                - "NoneInvoke": Do not invoke, but can describe them.
                - "Required": Must use at least one function to answer.
        """
        behavior = FunctionChoiceBehavior(
            enable_kernel_functions=enable_kernel_functions,
            maximum_auto_invoke_attempts=max_auto_invoke_attempts,
            filters=filters,
            type_=behavior_type,
        )
        self.planner_config.function_choice_behavior = behavior
        self.logger.info("Function choice behavior updated: %s", behavior)

    def configure_kernel_arguments(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        settings: Optional[
            Union[
                PromptExecutionSettings,
                List[PromptExecutionSettings],
                Dict[str, PromptExecutionSettings],
            ]
        ] = None,
    ) -> None:
        """
        Configure default KernelArguments that will be used in the `.run(...)` method
        if the user does not explicitly provide them at runtime.

        :param arguments: A dictionary of custom argument pairs (key=value).
            These can be any fields your custom functions might require beyond just `input`.
        :param settings: Optionally provide custom PromptExecutionSettings.
            This can be:
                - A single PromptExecutionSettings object
                - A list of PromptExecutionSettings (each with a unique service_id)
                - A dict[str, PromptExecutionSettings] keyed by service_id
        """
        self._default_kernel_arguments = arguments or {}
        self._default_kernel_settings = settings
        self.logger.info(
            "Default kernel arguments updated: %s, with settings: %s",
            self._default_kernel_arguments,
            self._default_kernel_settings,
        )

    def add_system_message(self, message: str) -> None:
        """
        Add a system message to the chat history.

        :param message: System message that defines the assistant's role or higher-level instructions.
        """
        self.chat_history.add_system_message(message)

    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the chat history.

        :param message: User query or input.
        """
        self.chat_history.add_user_message(message)

    def clear_chat_history(self) -> None:
        """
        Clear the current chat history. This can be useful if you want to reset the conversation context.
        """
        self.chat_history = ChatHistory()
        self.logger.info("Chat history cleared.")

    async def run(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        run_arguments: Optional[Dict[str, Any]] = None,
        run_settings: Optional[
            Union[
                PromptExecutionSettings,
                List[PromptExecutionSettings],
                Dict[str, PromptExecutionSettings],
            ]
        ] = None,
    ) -> str:
        """
        Execute the agent's main functionality with the given prompts,
        optionally overriding default kernel arguments and/or settings.

        :param system_prompt: The system prompt to set the assistant's role or global instructions.
        :param user_prompt: The user query or task description.
        :param run_arguments: A dict of additional KernelArguments to pass for this single call.
            These will be merged with any default arguments from `configure_kernel_arguments(...)`.
            If a key overlaps, the value here overrides the default.
        :param run_settings: A custom PromptExecutionSettings, list of them, or dict of them
            keyed by service_id to override default kernel settings for this single call.
        :return: AI response as a string.
        """
        if system_prompt is not None:
            self.add_system_message(system_prompt)
        if user_prompt is not None:
            self.add_user_message(user_prompt)

        try:
            # Prepare the KernelArguments:
            # 1. Start with any user-defined default args
            # 2. Merge in run_arguments if provided (overrides defaults)
            final_args = self._default_kernel_arguments.copy()
            if run_arguments is not None:
                final_args.update(run_arguments)

            # Also handle custom settings if provided, else fall back to defaults
            final_settings = (
                run_settings
                if run_settings is not None
                else self._default_kernel_settings
            )

            # Construct KernelArguments
            args_for_kernel = KernelArguments(settings=final_settings, **final_args)
            args_for_kernel["input"] = user_prompt

            if self.chat_completion is None:
                self.chat_completion: AzureChatCompletion = self.kernel.get_service(
                    type=ChatCompletionClientBase
                )

            result = await self.chat_completion.get_chat_message_contents(
                chat_history=self.chat_history,
                settings=self.planner_config,
                kernel=self.kernel,
                arguments=args_for_kernel,
            )
            response = result[0] if result else ""
            return response

        except Exception as e:
            self.logger.error("Error while processing request: %s", e, exc_info=True)
            raise

    def add_plugin(self, parent_directory: str, plugin_name: str) -> None:
        """
        Add a plugin to the kernel.

        :param parent_directory: Directory path where the plugin resides.
        :param plugin_name: Name of the plugin to load.
        """
        try:
            plugin = self.kernel.add_plugin(
                parent_directory=parent_directory, plugin_name=plugin_name
            )
            self.logger.info("Successfully added plugin: %s", plugin_name)
        except Exception as e:
            self.logger.error("Failed to add plugin: %s", plugin_name, exc_info=True)
            raise

    def load_skills(
        self, skills: List[Literal["retrieval", "main", "rewriting", "evaluation"]]
    ) -> None:
        """
        Load a list of specified "skills" (plugins) into the kernel.

        :param skills: A list of skill names to load, where valid options might be
            ["retrieval", "main", "rewriting", "evaluation"], etc.
        """
        self._skills_manager.load_skills(skills)
        for skill_name in skills:
            plugin = self._skills_manager.get_skill(skill_name)
            plugin_path = plugin.directory
            parent_dir = os.path.dirname(plugin_path)
            name = os.path.basename(plugin_path)
            try:
                self.kernel.add_plugin(parent_directory=parent_dir, plugin_name=name)
                self.logger.info("Skill loaded: %s", skill_name)
            except Exception as e:
                self.logger.error("Failed to load skill: %s", skill_name, exc_info=True)
                raise
