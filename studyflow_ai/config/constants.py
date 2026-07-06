"""
System-wide architectural constants for StudyFlow AI.
"""
# Latency & Timeouts
DEFAULT_TIMEOUT_SECONDS: float = 15.0
OCR_TIMEOUT_SECONDS: float = 30.0
CALENDAR_TIMEOUT_SECONDS: float = 5.0
VECTOR_SEARCH_TIMEOUT_SECONDS: float = 3.0

# LLM Configs
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL_DIMENSION: int = 768
MAX_HISTORY_MESSAGES_BUFFER: int = 10

# Domain Constraints
MAX_DAILY_STUDY_HOURS: int = 4
