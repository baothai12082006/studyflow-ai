# Tutoring & Explainer Agent
import os
from google.adk import Agent
from studyflow_ai.tools.knowledge_base import query_knowledge_base
from studyflow_ai.tools.db_operations import db_read_write_tool

prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "tutoring.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    tutoring_instruction = f.read()

tutoring_agent = Agent(
    name="TutoringAgent",
    instruction=tutoring_instruction,
    tools=[query_knowledge_base, db_read_write_tool]
)

async def tutor_student(question: str, course_id: str) -> dict:
    pass
