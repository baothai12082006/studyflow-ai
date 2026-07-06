"""
Request payloads mapping to Coordinator and sub-agent invocations.
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class CoordinatorRequest(BaseModel):
    """Payload sent directly from the client interface to CoordinatorAgent."""
    user_id: str = Field(..., description="Unique UUID of the student user")
    session_id: str = Field(..., description="Active session ID for history tracing")
    prompt: str = Field(..., description="User chat content text")
    attachments: Optional[List[dict]] = Field(default=None, description="Metadata list of uploaded syllabi/notes")

class IngestionRequest(BaseModel):
    """Payload delegated to the Ingestion Agent."""
    file_uri: str = Field(..., description="Local or remote path to the document file")
    mime_type: str = Field(..., description="Document type, e.g., application/pdf")
    course_id: str = Field(..., description="Destination course UUID")

class PlanningRequest(BaseModel):
    """Payload delegated to the Planning Agent."""
    user_id: str
    course_id: str

class TutoringRequest(BaseModel):
    """Payload delegated to the Tutoring Agent."""
    user_id: str
    course_id: str
    question: str
    chat_history: List[dict] = Field(default_factory=list)
