"""
Settings and configuration management for StudyFlow AI.
Loads environmental variables and handles defaults.
"""
import os
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Configurations
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    vertex_ai_project_id: Optional[str] = Field(default=None, validation_alias="VERTEX_AI_PROJECT_ID")
    vertex_ai_location: str = Field(default="us-central1", validation_alias="VERTEX_AI_LOCATION")

    # Database Configuration
    db_host: str = Field(default="localhost", validation_alias="DB_HOST")
    db_port: int = Field(default=5432, validation_alias="DB_PORT")
    db_name: str = Field(default="studyflow_db", validation_alias="DB_NAME")
    db_user: str = Field(default="postgres", validation_alias="DB_USER")
    db_password: str = Field(default="", validation_alias="DB_PASSWORD")

    # Vector DB
    pinecone_api_key: Optional[str] = Field(default=None, validation_alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1", validation_alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="studyflow-index", validation_alias="PINECONE_INDEX_NAME")

    # External APIs
    google_calendar_client_id: Optional[str] = Field(default=None, validation_alias="GOOGLE_CALENDAR_CLIENT_ID")
    google_calendar_client_secret: Optional[str] = Field(default=None, validation_alias="GOOGLE_CALENDAR_CLIENT_SECRET")

    # Notifications & SMTP
    smtp_host: str = Field(default="smtp.gmail.com", validation_alias="SMTP_HOST")
    smtp_port: int = Field(default=587, validation_alias="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, validation_alias="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, validation_alias="SMTP_PASSWORD")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Instantiate settings
settings = Settings()
