"""
Pydantic schemas representing all asynchronous system events.
"""
from pydantic import BaseModel, Field
from typing import List

class DocumentUploaded(BaseModel):
    user_id: str
    file_uri: str
    course_id: str

class SyllabusParsed(BaseModel):
    course_id: str
    deadlines: List[dict] = Field(default_factory=list, description="Extracted assignments and exams list")

class ScheduleCreated(BaseModel):
    user_id: str
    task_ids: List[str]

class TaskCompleted(BaseModel):
    user_id: str
    task_id: str
    timestamp: str

class StudySessionStarted(BaseModel):
    user_id: str
    task_id: str
    topic: str

class AcademicQuestionAsked(BaseModel):
    user_id: str
    course_id: str
    question: str
