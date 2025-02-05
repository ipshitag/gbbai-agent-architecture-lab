import os
import json
import logging
from typing import Optional, List, Dict, Any

from tabulate import tabulate
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageRole, MessageTextContent
from azure.core.exceptions import HttpResponseError
from utils.ml_logging import get_logger

class AzureAIAgents:
    """
    A unified class for working with Azure AI Foundry (azure.ai.projects).
    It manages:
      - Agent listing and retrieval
      - Agent creation
      - Running conversations
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        credential: Optional[DefaultAzureCredential] = None
    ):
        """
        Initialize the AzureAIAgents class. If no connection string is provided,
        the class will look for 'AZURE_AI_FOUNDRY_CONNECTION_STRING' in environment variables.

        :param connection_string: Azure AI Foundry connection string.
        :param credential: Azure credential object (DefaultAzureCredential is used if None).
        """
        self.logger = get_logger("AzureAIAgents")

        if not connection_string:
            connection_string = os.getenv("AZURE_AI_FOUNDRY_CONNECTION_STRING")

        if not connection_string:
            raise ValueError("No Azure AI Foundry connection string provided or found in environment variable.")

        if credential is None:
            credential = DefaultAzureCredential()

        # Create the AIProjectClient
        self.project = AIProjectClient.from_connection_string(
            conn_str=connection_string,
            credential=credential
        )
        
        self.logger.info("AI Foundry project client created successfully.")

    def list_agents(self) -> None:
        """
        Retrieves and displays all registered agents from Azure AI Foundry
        in a formatted table (ID, Name, Model, Created At, Owner).
        """
        self.logger.debug("Retrieving agent list...")
        agents_response = self.project.agents.list_agents()

        # Prepare table headers.
        table_headers = ["ID", "Name", "Model", "Created At", "Owner"]
        table_rows = []

        for agent_data in agents_response.get("data", []):
            agent_id = agent_data.get("id", "N/A")
            name = agent_data.get("name", "N/A")
            model = agent_data.get("model", "N/A")
            created_at = agent_data.get("created_at", "N/A")
            owner = agent_data.get("metadata", {}).get("owner", "N/A")

            table_rows.append([
                agent_id,
                name,
                model,
                created_at,
                owner
            ])

        # Print the table using tabulate.
        print(tabulate(table_rows, headers=table_headers, tablefmt="grid"))
        self.logger.info("Agent list retrieval complete.")

    def get_agent(self, assistant_id: str) -> Dict[str, Any]:
        """
        Retrieves an agent's information by its assistant ID.
        
        :param assistant_id: The unique ID for the agent.
        :return: Dictionary containing agent info if found.
        """
        try:
            agent_info = self.project.agents.get_agent(assistant_id=assistant_id)
            self.logger.info(f"Retrieved Agent: {assistant_id}")
            return agent_info
        except HttpResponseError as e:
            self.logger.error(f"Failed to retrieve agent {assistant_id}", exc_info=True)
            raise e

    def create_agent(
        self,
        deployment_name: Optional[str] = None,
        name: str = "my-basic-agent",
        description: str = "Basic agent for technology support",
        instructions: str = "You are a friendly assistant who loves answering technology questions.",
        tools: Optional[List[Any]] = None,
        tool_resources: Optional[Any] = None,
        toolset: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        response_format: Optional[Any] = None,
        metadata: Optional[Dict[str, str]] = None,
        content_type: str = 'application/json',
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Creates a basic AI agent using the project instance and extended parameters.
        
        :param deployment_name: The deployment model name (defaults to env variable).
        :param name: Name of the agent.
        :param description: Description for the agent.
        :param instructions: Instruction set for the agent's behavior.
        :param tools: Tools available to the agent.
        :param tool_resources: Configuration for the agent's tool resources.
        :param toolset: A set of tools for the agent.
        :param temperature: Controls response creativity.
        :param top_p: Controls nucleus sampling.
        :param response_format: The expected response format.
        :param metadata: Metadata for the agent.
        :param content_type: Default content type (JSON).
        :param kwargs: Additional keyword arguments.
        :return: Dictionary representing the created agent.
        """
        if not deployment_name:
            deployment_name = os.environ.get("AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID")
            if not deployment_name:
                self.logger.error("Environment variable 'AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID' is not set.")
                raise ValueError("Missing deployment_name and environment variable for Azure OpenAI model.")

        if metadata is None:
            metadata = {"owner": "IT Support"}

        try:
            created_agent = self.project.agents.create_agent(
                model=deployment_name,
                name=name,
                description=description,
                instructions=instructions,
                tools=tools,
                tool_resources=tool_resources,
                toolset=toolset,
                temperature=temperature,
                top_p=top_p,
                response_format=response_format,
                metadata=metadata,
                content_type=content_type,
                **kwargs
            )
            self.logger.info(f"Created Agent ID: {created_agent.id}")
            self.logger.info(f"Agent Metadata: {created_agent.metadata}")
            return created_agent
        except HttpResponseError as e:
            error_content = e.response.content
            try:
                error_json = json.loads(error_content)
                self.logger.error(f"Error Message: {error_json.get('Message')}")
            except json.JSONDecodeError:
                self.logger.error(f"Non-JSON Error Content: {error_content}")
            raise e

    def run_agent_conversation(self, agent: Dict[str, Any], query: str) -> str:
        """
        Creates a conversation thread, sends a user message, and processes the run
        for the given agent (by its ID). Returns the last message response as a string.

        :param agent: Dictionary containing the agent info (including the agent.id).
        :param query: The user query to send to the agent.
        :return: The final text response from the agent.
        """
        agent_id = agent.get("id")
        if not agent_id:
            raise ValueError("Agent object missing 'id' field. Cannot run conversation.")

        try:
            thread = self.project.agents.create_thread()
            self.logger.info(f"[Agent {agent_id}] Created Thread: {thread.id}")
            
            user_msg = self.project.agents.create_message(
                thread_id=thread.id,
                role=MessageRole.USER,
                content=query
            )
            self.logger.info(f"[Agent {agent_id}] Created user message ID: {user_msg.id}")
            
            run = self.project.agents.create_and_process_run(
                thread_id=thread.id,
                assistant_id=agent_id
            )
            self.logger.info(f"[Agent {agent_id}] Run finished with status: {run.status}")
            
            if run.status == "failed":
                self.logger.error(f"[Agent {agent_id}] Run failed.")
            
            all_messages = self.project.agents.list_messages(thread_id=thread.id)
            self.logger.info(f"----- Conversation for Agent {agent_id} -----")

            final_text = "No response from agent."
            for msg in reversed(all_messages.data):
                if msg.content and len(msg.content):
                    last_content = msg.content[-1]
                    if isinstance(last_content, MessageTextContent):
                        content_text = last_content.text.value
                        self.logger.info(f"{msg.role.upper()}: {content_text}")
                        if msg.role == MessageRole.ASSISTANT:
                            final_text = content_text
                            break

            return final_text

        except Exception as e:
            self.logger.error("Failed to create and process run", exc_info=True)
            raise e
        
    def delete_agent(self, agent_id: str) -> None:
        """
        Deletes an existing agent by its ID.

        :param agent_id: The unique ID of the agent to delete.
        :return: None
        """
        self.logger.debug(f"Attempting to delete agent with ID: {agent_id}")
        try:
            self.project.agents.delete_agent(assistant_id=agent_id)
            self.logger.info(f"Successfully deleted agent with ID: {agent_id}")
        except HttpResponseError as e:
            self.logger.error(f"Failed to delete agent {agent_id}", exc_info=True)
            raise e
