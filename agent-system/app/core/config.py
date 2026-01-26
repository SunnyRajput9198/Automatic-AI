from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    ENV: str = "development"

    DATABASE_URL: str
    ANTHROPIC_API_KEY: str | None = None

    WORKSPACE_DIR: str = "/app/workspace"
    SANDBOX_DIR: str = "/app/sandbox"

    ENABLE_SHELL: bool = False
    ENABLE_PYTHON_EXECUTOR: bool = True

    model_config = ConfigDict(
        env_file=".env",
        extra="allow"
    )


settings = Settings()
