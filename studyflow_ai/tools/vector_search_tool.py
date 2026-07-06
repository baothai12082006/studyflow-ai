"""
Vector Search Tool for StudyFlow AI.
Interfaces with VectorService to embed/index docs and retrieve matched context chunks.
"""
import logging
from typing import List
from google.adk import tool
from studyflow_ai.services.vector_service import VectorService

logger = logging.getLogger(__name__)

# Mock config
vector_service = VectorService(api_key="mock", environment="mock", index_name="mock")

@tool
async def index_academic_document(course_id: str, text_chunks: List[str]) -> bool:
    """
    Indexes academic slide or document chunks inside the vector database.
    
    Args:
        course_id: Unique UUID of the course
        text_chunks: List of text pages to embed
    """
    try:
        formatted_chunks = [{"text": chunk} for chunk in text_chunks]
        await vector_service.upsert_embeddings(course_id, formatted_chunks)
        return True
    except Exception as e:
        logger.error(f"Index execution failed: {e}")
        return False

@tool
async def query_academic_chunks(query: str, course_id: str, top_k: int = 5) -> List[dict]:
    """
    Queries the vector database for document segments relevant to the user query.
    
    Args:
        query: Concept query terms
        course_id: Scope filter course
        top_k: Number of chunks to return
    """
    try:
        return await vector_service.similarity_search(query, course_id, top_k)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []
