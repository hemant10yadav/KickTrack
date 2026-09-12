from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Football Analysis Tool"
    upload_dir: str = "data/uploads"
    processed_dir: str = "data/processed"

    class Config:
        env_file = ".env"


settings = Settings()
