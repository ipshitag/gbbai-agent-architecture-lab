import os
import logging
from typing import Annotated, Dict, Any, List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizableTextQuery,
)
from utils.ml_logging import get_logger

# Set up logging
logger = get_logger(name="AzureSearchFunctions", level=logging.DEBUG)

def init_search_client() -> SearchClient:
    """
    Initializes the Azure Search client using environment variables.
    """
    endpoint = os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT_SCENARIO_1")
    index_name = os.getenv("AZURE_SEARCH_INDEX_NAME_SCENARIO_1")
    api_key = os.getenv("AZURE_AI_SEARCH_ADMIN_KEY_SCENARIO_1")
    if not all([endpoint, index_name, api_key]):
        logger.error("One or more environment variables for Azure Search are missing.")
        raise ValueError("Missing environment variable for Azure Search.")
    logger.info(f"Initializing SearchClient with endpoint: {endpoint}, index_name: {index_name}")
    credential = AzureKeyCredential(api_key)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
    logger.info("SearchClient initialized successfully.")
    return search_client

# Initialize the search client at startup
search_client = init_search_client()

def _format_azure_search_results(results: List[Dict[str, Any]], truncate: int = 1000) -> str:
    """
    Formats Azure AI Search results into a structured, readable string.
    
    Each result contains:
    - Chunk ID
    - Reranker Score
    - Source Document Path
    - Content (truncated if too long)
    - Caption (if available)
    
    :param results: List of results from the Azure AI Search API.
    :param truncate: Maximum number of characters to include from the content.
    :return: A formatted string representation of the search results.
    """
    formatted_results = []
    for result in results:
        chunk_id = result.get('chunk_id', 'N/A')
        reranker_score = result.get('@search.reranker_score', 'N/A')
        source_doc_path = result.get('parent_path', 'N/A')
        content = result.get('chunk', 'N/A')
        content = content[:truncate] + "..." if len(content) > truncate else content

        # Handle captions if available
        captions = result.get('@search.captions', [])
        caption = "Caption not available"
        if captions:
            first_caption = captions[0]
            # Check if the first caption has 'highlights' or 'text'
            if hasattr(first_caption, 'highlights') and first_caption.highlights:
                caption = first_caption.highlights
            elif hasattr(first_caption, 'text') and first_caption.text:
                caption = first_caption.text

        result_string = (
            "========================================\n"
            f"🆔 ID: {chunk_id}\n"
            f"📂 Source Doc Path: {source_doc_path}\n"
            f"📜 Content: {content}\n"
            f"💡 Caption: {caption}\n"
            "========================================"
        )
        formatted_results.append(result_string)
    return "\n\n".join(formatted_results)

def keyword_search(search_text: str, top: int = 5) -> Annotated[str, "A formatted string of search results for the keyword query"]:
    """
    Executes a keyword-based search on the index.
    
    :param search_text: The text to search for using keyword search.
    :param top: The maximum number of results to return.
    :return: A formatted string of search results.
    """
    logger.info("keyword_search function called.")
    try:
        results = search_client.search(
            search_text=search_text,
            query_type=QueryType.SIMPLE,
            top=top
        )
        formatted = _format_azure_search_results(results)
        logger.info(f"Extracted results: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"keyword_search - Error during keyword search: {e}")
        return "Error during keyword search."

def semantic_search(search_text: str, top: int = 5) -> Annotated[str, "A formatted string of search results for the semantic query"]:
    """
    Executes a semantic search on the index.
    
    :param search_text: The text to search for using semantic search.
    :param top: The maximum number of results to return.
    :return: A formatted string of search results.
    """
    logger.info("semantic_search function called.")
    try:
        vector_query = VectorizableTextQuery(
            text=search_text, k_nearest_neighbors=5, fields="vector", weight=0.5
        )
        results = search_client.search(
            search_text=search_text,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="my-semantic-config",
            query_caption=QueryCaptionType.EXTRACTIVE,
            query_answer=QueryAnswerType.EXTRACTIVE,
            filter="",
            top=top
        )
        formatted = _format_azure_search_results(results)
        logger.info(f"Extracted results: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"semantic_search - Error during semantic search: {e}")
        return "Error during semantic search."

def hybrid_search(search_text: str, top: int = 5) -> Annotated[str, "A formatted string of search results for the hybrid query"]:
    """
    Executes a hybrid search on the index by combining keyword and vector-based search.
    
    :param search_text: The text to search for using the hybrid search approach.
    :param top: The maximum number of results to return.
    :return: A formatted string of search results.
    """
    logger.info("hybrid_search function called.")
    try:
        vector_query = VectorizableTextQuery(
            text=search_text, k_nearest_neighbors=5, fields="vector", weight=0.5
        )
        results = search_client.search(
            search_text=search_text,
            vector_queries=[vector_query],
            query_type=QueryType.SIMPLE,
            top=top
        )
        formatted = _format_azure_search_results(results)
        logger.info(f"Extracted results: {formatted}")
        return formatted
    except Exception as e:
        logger.error(f"hybrid_search - Error during hybrid search: {e}")
        return "Error during hybrid search."
