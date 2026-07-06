# RAG Knowledge Base Query Tool Wrapper
from google.adk import tool
from studyflow_ai.services.vector_db import VectorDBService

@tool
async def query_knowledge_base(query: str, course_id: str, top_k: int = 5) -> dict:
    """Queries the Vector Database for semantic slide and reading matches.
    
    Args:
        query: Embeddings search term
        course_id: Scope filter
        top_k: Max chunks to return
    """
    vector_service = VectorDBService()
    results = await vector_service.similarity_search(query, course_id, top_k)
    return {"results": results}
