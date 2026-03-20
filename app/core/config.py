from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General
    app_name: str = "AI Knowledge Hub - Minimal RAG"
    env: str = "local"

    # OpenSearch
    opensearch_host: str = Field("http://localhost:9200", alias="OPENSEARCH_HOST")
    opensearch_username: str | None = Field(None, alias="OPENSEARCH_USERNAME")
    opensearch_password: str | None = Field(None, alias="OPENSEARCH_PASSWORD")
    opensearch_index: str = Field("rag_chunks", alias="OPENSEARCH_INDEX")

    # Embeddings / LLM
    embedding_model: str = Field("nomic-embed-text", alias="EMBEDDING_MODEL")
    llm_model: str = Field("llama3.2:3b", alias="LLM_MODEL")
    llm_provider: str = Field("ollama", alias="LLM_PROVIDER")
    embedding_dimension: int = Field(768, alias="EMBEDDING_DIMENSION")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")

    class Config:
        env_file = ".env"
        extra = "ignore"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
