"""
Helper data schemas mapping to internal objects.
"""
from pydantic import BaseModel
from datetime import datetime

class StudyTaskSchema(BaseModel):
    """Individual task block schema."""
    task_id: str
    course_id: str
    title: str
    start_time: datetime
    end_time: datetime
    is_completed: bool = False
