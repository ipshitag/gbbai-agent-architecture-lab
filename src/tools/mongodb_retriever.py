import os
import urllib.parse
import pymongo
import logging
from typing import List
from src.aoai.azure_openai import AzureOpenAIManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Retrieve environment variables for MongoDB connection
COSMOS_MONGO_USER = os.environ.get('COSMOS_MONGO_USER')
COSMOS_MONGO_PWD = os.environ.get('COSMOS_MONGO_PWD')
COSMOS_MONGO_SERVER = os.environ.get('COSMOS_MONGO_SERVER')

# Optionally, define the database and collection names via environment variables; otherwise, use defaults.
DEFAULT_DATABASE = os.environ.get('COSMOS_MONGO_DATABASE', 'ExampleDB')
DEFAULT_COLLECTION = os.environ.get('COSMOS_MONGO_COLLECTION', 'ExampleCollection')

# Construct the MongoDB connection string
mongo_conn = (
    "mongodb+srv://"
    + urllib.parse.quote(COSMOS_MONGO_USER)
    + ":"
    + urllib.parse.quote(COSMOS_MONGO_PWD)
    + "@"
    + COSMOS_MONGO_SERVER
    + "?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
)

# Initialize Azure OpenAI Manager
aoai_helper = AzureOpenAIManager(
    api_key=os.getenv('AZURE_OPENAI_KEY'),
    api_version=os.getenv('AZURE_OPENAI_API_VERSION', "2023-05-15"),
    azure_endpoint=os.getenv('AZURE_OPENAI_API_ENDPOINT'),
    embedding_model_name=os.getenv('AZURE_AOAI_EMBEDDINGS_MODEL_NAME_DEPLOYMENT_ID'),
    completion_model_name=os.getenv('AZURE_AOAI_CHAT_MODEL_NAME_DEPLOYMENT_ID'),
)

def generate_embeddings(text: str) -> List[float]:
    """
    Generates an embedding vector for the given text using Azure OpenAI.

    Args:
        text (str): The text to be converted into an embedding.

    Returns:
        List[float]: The generated embedding vector.
    """
    try:
        embedding_response = aoai_helper.generate_embedding(text)
        return embedding_response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding for text: {text}. Error: {e}")
        return []

# Establish connection and select database
try:
    mongo_client = pymongo.MongoClient(mongo_conn)
    db = mongo_client[DEFAULT_DATABASE]
    print("✅ Connected to MongoDB.")
except pymongo.errors.ConnectionError as e:
    print(f"❌ MongoDB connection error: {e}")

def retrieve_document(query: dict) -> dict:
    """
    Retrieve a single document from the default collection that matches the query.

    Args:
        query (dict): The MongoDB query filter.

    Returns:
        dict: The first document matching the query or None if no document is found.
    """
    try:
        collection = db[DEFAULT_COLLECTION]
        document = collection.find_one(query)
        return document
    except Exception as e:
        print(f"Error retrieving document: {e}")
        return None

def upsert_document(query: dict, document: dict) -> dict:
    """
    Insert a new document or update an existing document in the default collection.

    This function performs an upsert operation: if a document matching the query exists,
    it replaces the document; otherwise, it inserts the provided document.

    Args:
        query (dict): The query filter to match an existing document.
        document (dict): The document to insert or update.

    Returns:
        dict: A summary of the upsert operation including matched count, modified count, and upserted ID.
    """
    try:
        collection = db[DEFAULT_COLLECTION]
        result = collection.replace_one(query, document, upsert=True)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": result.upserted_id
        }
    except Exception as e:
        print(f"Error during upsert: {e}")
        return {}

def update_document(query: dict, update: dict) -> int:
    """
    Update a single document in the default collection.

    Args:
        query (dict): The query filter to select the document.
        update (dict): The update operations to apply (e.g., {"$set": {...}}).

    Returns:
        int: The number of documents modified (0 or 1).
    """
    try:
        collection = db[DEFAULT_COLLECTION]
        result = collection.update_one(query, update)
        return result.modified_count
    except Exception as e:
        print(f"Error updating document: {e}")
        return 0

def delete_document(query: dict) -> int:
    """
    Delete a single document from the default collection that matches the query.

    Args:
        query (dict): The query filter to match the document to delete.

    Returns:
        int: The number of documents deleted (0 or 1).
    """
    try:
        collection = db[DEFAULT_COLLECTION]
        result = collection.delete_one(query)
        return result.deleted_count
    except Exception as e:
        print(f"Error deleting document: {e}")
        return 0

def query_documents(query: dict) -> list:
    """
    Retrieve all documents from the default collection that match the query.

    Args:
        query (dict): The MongoDB query filter.

    Returns:
        list: A list of documents matching the query.
    """
    try:
        collection = db[DEFAULT_COLLECTION]
        documents = list(collection.find(query))
        return documents
    except Exception as e:
        print(f"Error querying documents: {e}")
        return []

def vector_search(query: str):
    """
    Searches for semantically similar documents in CosmosDB using vector search.

    If a high-confidence result (similarity ≥ similarity_threshold) is found, returns its response.
    Otherwise, returns None to indicate that a fallback to an LLM-generated response may be needed.

    Args:
        query (str): The query text.
    Returns:
        The response from the best matching document if confidence is high; otherwise, None.
    """
    SIMILARITY_THRESHOLD=0.96
    
    # Generate the embeddings for the query text
    embedding_response = generate_embeddings(query)

    # Define search pipeline with required fields
    search_stage = {
        "$vectorSearch": {
            "index": "VectorSearchIndex",  # Ensure this matches the actual index name
            "path": "queryVector",         # Must match the field storing embeddings
            "queryVector": embedding_response,
            "numCandidates": 5,
            "limit": 5
        }
    }

    # Projection stage to include similarity score and response field
    project_stage = {
        "$project": {
            "similarityScore": {"$meta": "searchScore"},
            "response": 1
        }
    }

    # Assemble and execute the pipeline
    pipeline = [search_stage, project_stage]
    collection = db[DEFAULT_COLLECTION]
    
    try:
        results = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"❌ MongoDB vector search failed: {e}")
        return None

    if not results:
        print("⚠️ No matching vector results found in CosmosDB.")
        return None

    # Retrieve the best result by similarity score
    best_result = max(results, key=lambda x: x.get("similarityScore", 0), default=None)

    return best_result["response"]
