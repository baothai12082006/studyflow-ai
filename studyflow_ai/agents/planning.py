# Planning & Scheduling Agent
import os
from google.adk import Agent
from studyflow_ai.tools.calendar_sync import read_calendar_tool, write_calendar_tool

prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "planning.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    planning_instruction = f.read()

planning_agent = Agent(
    name="PlanningAgent",
    instruction=planning_instruction,
    tools=[read_calendar_tool, write_calendar_tool]
)

async def create_study_blocks(user_id: str, course_id: str) -> dict:
    pass
