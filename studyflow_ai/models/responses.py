"""
Response payloads mapping to Coordinator and sub-agent outputs.
"""
from pydantic import BaseModel, Field
from typing import List, Optional

class CoordinatorResponse(BaseModel):
    """Response returned from StudyFlow AI to the client UI."""
    status: str = Field(..., description="Execution status: SUCCESS, WAITING, or ERROR")
    display_text: str = Field(..., description="Markdown response to display to the student")
    ui_action: Optional[str] = Field(default=None, description="Triggers specific frontend UI transitions")

class IngestionResponse(BaseModel):
    """Response from Ingestion Agent extraction process."""
    extraction_status: str
    confidence_score: float
    extracted_json: dict = Field(..., description="Parsed Course schema information")

class PlanningResponse(BaseModel):
    """Response from Planning Agent schedule updates."""
    tasks_created: int
    calendar_sync_status: bool
    error_msg: Optional[str] = None

class TutoringResponse(BaseModel):
    """Response containing Socratic tutoring guides and citations."""
    answer: str = Field(..., description="Grounded response from TEA")
    citations: List[dict] = Field(default_factory=list, description="Documents and slides cited")
