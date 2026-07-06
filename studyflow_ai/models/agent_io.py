# Agent Input/Output schemas
from pydantic import BaseModel
from typing import List, Optional

class CoordinatorRequest(BaseModel):
    user_id: str
    session_id: str
    prompt: str
    attachments: Optional[List[dict]] = None

class CoordinatorResponse(BaseModel):
    status: str
    display_text: str
    ui_action: Optional[str] = None
