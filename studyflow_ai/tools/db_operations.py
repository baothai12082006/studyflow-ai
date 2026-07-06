# Shared Database Query abstraction Tool Wrapper
from google.adk import tool
from studyflow_ai.repositories.db_client import DBClient

@tool
def db_read_write_tool(query: str, params: dict = None) -> dict:
    """Executes database mutations or selects on the shared SQL DB.
    
    Args:
        query: Target SQL query
        params: Bind values dictionary
    """
    client = DBClient()
    return client.execute(query, params)
