"""
Shared state models mapping to DB tables and local session states.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class ConversationState(BaseModel):
    """Saves rolling history and active callbacks for current session."""
    session_id: str
    user_id: str
    active_intent: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    awaiting_callback: bool = False

class UserState(BaseModel):
    """Student demographic and target persona information."""
    user_id: str
    persona: str = Field(default="FRESHMAN")
    timezone: str = Field(default="UTC")
    preferences: Dict[str, str] = Field(default_factory=dict)

class AcademicState(BaseModel):
    """Represents course workload metadata."""
    course_id: str
    user_id: str
    title: str
    deadlines: List[Dict[str, str]] = Field(default_factory=list)

class ProgressState(BaseModel):
    """Activity streak metrics tracked by Monitoring Agent."""
    user_id: str
    current_streak_days: int = 0
    highest_streak_days: int = 0
    total_tasks_completed: int = 0
    struggle_topics: List[str] = Field(default_factory=list)
