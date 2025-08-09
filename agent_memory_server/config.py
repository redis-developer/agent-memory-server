import os
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


load_dotenv()


# Model configuration mapping
MODEL_CONFIGS = {
    "gpt-4o": {"provider": "openai", "embedding_dimensions": None},
    "gpt-4o-mini": {"provider": "openai", "embedding_dimensions": None},
    "gpt-4": {"provider": "openai", "embedding_dimensions": None},
    "gpt-3.5-turbo": {"provider": "openai", "embedding_dimensions": None},
    "text-embedding-3-small": {"provider": "openai", "embedding_dimensions": 1536},
    "text-embedding-3-large": {"provider": "openai", "embedding_dimensions": 3072},
    "text-embedding-ada-002": {"provider": "openai", "embedding_dimensions": 1536},
    "claude-3-opus-20240229": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-haiku-20240307": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-5-sonnet-20240620": {
        "provider": "anthropic",
        "embedding_dimensions": None,
    },
    "claude-3-5-sonnet-20241022": {
        "provider": "anthropic",
        "embedding_dimensions": None,
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "embedding_dimensions": None,
    },
    "claude-3-7-sonnet-20250219": {
        "provider": "anthropic",
        "embedding_dimensions": None,
    },
    "claude-3-7-sonnet-latest": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-5-sonnet-latest": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-5-haiku-latest": {"provider": "anthropic", "embedding_dimensions": None},
    "claude-3-opus-latest": {"provider": "anthropic", "embedding_dimensions": None},
    "o1": {"provider": "openai", "embedding_dimensions": None},
    "o1-mini": {"provider": "openai", "embedding_dimensions": None},
    "o3-mini": {"provider": "openai", "embedding_dimensions": None},
}


class Settings(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    long_term_memory: bool = True
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_base: str | None = None
    anthropic_api_base: str | None = None
    generation_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    port: int = 8000
    mcp_port: int = 9000

    # Vector store factory configuration
    # Python dotted path to function that returns VectorStore or VectorStoreAdapter
    # Function signature: (embeddings: Embeddings) -> Union[VectorStore, VectorStoreAdapter]
    # Examples:
    #   - "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    #   - "my_module.my_vectorstore_factory"
    #   - "my_package.adapters.create_custom_adapter"
    vectorstore_factory: str = (
        "agent_memory_server.vectorstore_factory.create_redis_vectorstore"
    )

    # RedisVL configuration (used by default Redis factory)
    redisvl_index_name: str = "memory_records"

    # The server indexes messages in long-term memory by default. If this
    # setting is enabled, we also extract discrete memories from message text
    # and save them as separate long-term memory records.
    enable_discrete_memory_extraction: bool = True

    # Topic modeling
    topic_model_source: Literal["BERTopic", "LLM"] = "LLM"
    # If using BERTopic, use a supported model, such as
    # "MaartenGr/BERTopic_Wikipedia"
    topic_model: str = "gpt-4o-mini"
    enable_topic_extraction: bool = True
    top_k_topics: int = 3

    # Used for extracting entities from text
    ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english"
    enable_ner: bool = True
    index_all_messages_in_long_term_memory: bool = False

    # RedisVL Settings
    # TODO: Adapt to vector store settings
    redisvl_distance_metric: str = "COSINE"
    redisvl_vector_dimensions: str = "1536"
    redisvl_index_prefix: str = "memory_idx"
    redisvl_indexing_algorithm: str = "HNSW"

    # Docket settings
    docket_name: str = "memory-server"
    use_docket: bool = True

    # Authentication settings
    disable_auth: bool = True
    auth_mode: Literal["disabled", "token", "oauth2"] = "disabled"

    # OAuth2/JWT Authentication settings
    oauth2_issuer_url: str | None = None
    oauth2_audience: str | None = None
    oauth2_jwks_url: str | None = None
    oauth2_algorithms: list[str] = ["RS256"]

    # Token Authentication settings
    token_auth_enabled: bool = False

    # Auth0 Client Credentials (for testing and client applications)
    auth0_client_id: str | None = None
    auth0_client_secret: str | None = None

    # Working memory settings
    summarization_threshold: float = (
        0.7  # Fraction of context window that triggers summarization
    )

    # Other Application settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    default_mcp_user_id: str | None = None
    default_mcp_namespace: str | None = None

    # Forgetting settings
    forgetting_enabled: bool = False
    forgetting_every_minutes: int = 60
    forgetting_max_age_days: float | None = None
    forgetting_max_inactive_days: float | None = None
    # Keep only top N most recent (by recency score) when budget is set
    forgetting_budget_keep_top_n: int | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables

    @property
    def generation_model_config(self) -> dict[str, Any]:
        """Get configuration for the generation model."""
        return MODEL_CONFIGS.get(self.generation_model, {})

    @property
    def embedding_model_config(self) -> dict[str, Any]:
        """Get configuration for the embedding model."""
        return MODEL_CONFIGS.get(self.embedding_model, {})

    def load_yaml_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}


settings = Settings()


def get_config():
    """Get configuration from environment and settings files."""
    config_data = {}

    # If REDIS_MEMORY_CONFIG is set, load config from file
    config_file = os.getenv("REDIS_MEMORY_CONFIG")
    if config_file:
        try:
            with open(config_file) as f:
                if config_file.endswith((".yaml", ".yml")):
                    config_data = yaml.safe_load(f) or {}
                else:
                    # Assume JSON
                    import json

                    config_data = json.load(f) or {}
        except FileNotFoundError:
            print(f"Warning: Config file {config_file} not found")
        except Exception as e:
            print(f"Warning: Error loading config file {config_file}: {e}")

    # Environment variables override file config
    for key, value in os.environ.items():
        if key.startswith("REDIS_MEMORY_"):
            config_key = key[13:].lower()  # Remove REDIS_MEMORY_ prefix
            config_data[config_key] = value

    return config_data
