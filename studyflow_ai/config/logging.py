"""
Central logging configuration for StudyFlow AI.
Sets log levels and uniform formatting.
"""
import logging
import sys

def configure_logging(level: int = logging.INFO) -> None:
    """Configures system-wide logging formatting."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

configure_logging()
logger = logging.getLogger("studyflow_ai")
