from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    ENV: str = "development"

    DATABASE_URL: str = ""
    ANTHROPIC_API_KEY: str | None = None

    WORKSPACE_DIR: str = "/app/workspace"
    SANDBOX_DIR: str = "/app/sandbox"

    ENABLE_SHELL: bool = False
    ENABLE_PYTHON_EXECUTOR: bool = True


settings = Settings()
