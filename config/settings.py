"""
Configuration settings for the Advanced SQL Agent System
Centralized configuration management with environment variable support
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    Follows the 12-factor app methodology for configuration.
    """
    
    # === Application Settings ===
    app_name: str = Field(default="Advanced SQL Agent System", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # === Snowflake Database Settings ===
    snowflake_account: str = Field(..., description="Snowflake account identifier")
    snowflake_user: str = Field(..., description="Snowflake username")
    snowflake_password: str = Field(..., description="Snowflake password")
    snowflake_warehouse: str = Field(..., description="Snowflake warehouse")
    snowflake_database: str = Field(..., description="Snowflake database")
    snowflake_schema: str = Field(default="PUBLIC", description="Snowflake schema")
    snowflake_role: Optional[str] = Field(default=None, description="Snowflake role")
    
    # === LLM Provider Settings ===
    llm_provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    
    # === Memory System Settings ===
    memory_backend: str = Field(default="postgresql", description="Memory storage backend")
    memory_connection_string: Optional[str] = Field(default=None, description="Memory database connection string")
    redis_url: Optional[str] = Field(default="redis://localhost:6379", description="Redis URL for caching")
    
    # === Vector Store Settings ===
    vector_store_provider: str = Field(default="chromadb", description="Vector store provider")
    vector_store_path: str = Field(default="./data/vector_store", description="Vector store data path")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    
    # === Performance Settings ===
    max_concurrent_queries: int = Field(default=10, description="Maximum concurrent queries")
    query_timeout_seconds: int = Field(default=300, description="Query timeout in seconds")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    max_memory_usage_mb: int = Field(default=1024, description="Maximum memory usage in MB")
    
    # === Security Settings ===
    secret_key: str = Field(default="your-secret-key-here", description="Application secret key")
    allowed_hosts: list = Field(default=["*"], description="Allowed hosts for CORS")
    enable_rate_limiting: bool = Field(default=True, description="Enable API rate limiting")
    max_requests_per_minute: int = Field(default=60, description="Max requests per minute per user")
    
    # === Monitoring Settings ===
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")
    enable_tracing: bool = Field(default=False, description="Enable distributed tracing")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_database_url(self) -> str:
        """Constructs the Snowflake connection URL."""
        return f"snowflake://{self.snowflake_user}:{self.snowflake_password}@{self.snowflake_account}/{self.snowflake_database}/{self.snowflake_schema}?warehouse={self.snowflake_warehouse}"
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Returns LLM configuration based on provider."""
        if self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "temperature": 0.1,
                "max_tokens": 4000
            }
        elif self.llm_provider == "anthropic":
            return {
                "provider": "anthropic", 
                "api_key": self.anthropic_api_key,
                "model": "claude-3-sonnet-20240229",
                "temperature": 0.1,
                "max_tokens": 4000
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def validate_configuration(self) -> bool:
        """Validates that all required configuration is present."""
        
        errors = []
        
        # Check Snowflake settings
        required_snowflake = [
            "snowflake_account", "snowflake_user", "snowflake_password",
            "snowflake_warehouse", "snowflake_database"
        ]
        
        for field in required_snowflake:
            if not getattr(self, field):
                errors.append(f"Missing required Snowflake setting: {field}")
        
        # Check LLM settings
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            errors.append("Anthropic API key is required when using Anthropic provider")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Returns the global settings instance."""
    return settings

def create_directories():
    """Creates necessary directories for the application."""
    
    directories = [
        "./data",
        "./data/vector_store", 
        "./logs",
        "./cache",
        Path(settings.vector_store_path).parent
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Test configuration
    create_directories()
    if settings.validate_configuration():
        print("âœ… Configuration is valid")
        print(f"Database URL: {settings.get_database_url()}")
        print(f"LLM Config: {settings.get_llm_config()}")
    else:
        print("âŒ Configuration validation failed")
