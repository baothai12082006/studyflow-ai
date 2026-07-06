# StudyFlow Coordinator Agent (Orchestrator)
import os
from google.adk import Agent

# Load prompt template from prompts package
prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "coordinator.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    coordinator_instruction = f.read()

coordinator_agent = Agent(
    name="CoordinatorAgent",
    instruction=coordinator_instruction,
)

async def route_request(session_id: str, prompt: str):
    pass
