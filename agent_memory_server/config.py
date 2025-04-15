import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    window_size: int = 20
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    generation_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    port: int = 8000
    mcp_port: int = 9000

    # Topic and NER model settings
    topic_model: str = "MaartenGr/BERTopic_Wikipedia"
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_topic_extraction: bool = True
    enable_ner: bool = True

    # RedisVL Settings
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_name: str = "memory"
    redisvl_index_prefix: str = "memory"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True


settings = Settings()
