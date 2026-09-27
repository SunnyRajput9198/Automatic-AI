from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    ENV: str = "development"
    DATABASE_URL: str = ""
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY:    str | None = None
    OPENAI_BASE_URL:   str = "https://aicredits.in/v1"
    WORKSPACE_DIR:     str = "/app/workspace"
    SHARED_WORKSPACE:  str = "/app/workspace/shared"
    SANDBOX_DIR:       str = "/app/sandbox"
    COSTS_DIR:         str = "/app/costs"
    ENABLE_SHELL:            bool = False
    ENABLE_PYTHON_EXECUTOR:  bool = True
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    DEFAULT_OPENAI_MODEL:    str = "gpt-5-mini"
    DEFAULT_ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    NEWSAPI_KEY: str | None = None
    GITHUB_TOKEN: str | None = None
    RESEARCH_MAX_ITERATIONS: int = Field(default=2, ge=0, le=5)
    RESEARCH_MAX_TOOL_CALLS: int = Field(default=8, ge=1, le=20)
    RESEARCH_MAX_SECONDS: int = Field(default=180, ge=10, le=900)
    RESEARCH_MAX_REPEAT_STEPS: int = Field(default=1, ge=0, le=3)

settings = Settings()
