# Ingestion & Parsing Agent
import os
from google.adk import Agent
from studyflow_ai.tools.parse_document import parse_document_tool
from studyflow_ai.tools.ocr_image import ocr_image_tool

prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "ingestion.txt")
with open(prompt_path, "r", encoding="utf-8") as f:
    ingestion_instruction = f.read()

ingestion_agent = Agent(
    name="IngestionAgent",
    instruction=ingestion_instruction,
    tools=[parse_document_tool, ocr_image_tool]
)

async def ingest_document(file_uri: str) -> dict:
    pass
