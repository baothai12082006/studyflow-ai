# Monitoring & Streak Agent
import os
from google.adk import Agent
from studyflow_ai.tools.notifications import push_notification_tool

prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "monitoring.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    monitoring_instruction = f.read()

monitoring_agent = Agent(
    name="MonitoringAgent",
    instruction=monitoring_instruction,
    tools=[push_notification_tool]
)

async def update_streak(user_id: str, task_id: str) -> dict:
    pass
