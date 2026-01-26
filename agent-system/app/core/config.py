import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_ENV = os.getenv("APP_ENV", "development")
    APP_NAME = os.getenv("APP_NAME", "agent-system")
    PORT = int(os.getenv("PORT", 8000))

    DATABASE_URL = os.getenv("DATABASE_URL")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "info")


settings = Settings()
