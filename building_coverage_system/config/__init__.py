"""
Configuration module for building coverage system.

This module provides configuration management utilities and environment-specific
settings for the building coverage match system.
"""

import os
from typing import Dict, Any, Optional
from .development import get_development_config, DEV_SQL_QUERIES, DEV_RAG_PARAMS
from .production import get_production_config, PROD_SQL_QUERIES, PROD_RAG_PARAMS


class ConfigManager:
    """
    Configuration manager for the building coverage system.
    
    This class handles loading and managing configuration settings
    based on the current environment.
    """
    
    def __init__(self, environment: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            environment (Optional[str]): Environment name ('development', 'production')
                                       If None, will be read from ENVIRONMENT variable
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self._config = None
        self._sql_queries = None
        self._rag_params = None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the complete configuration for the current environment.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        if self._config is None:
            self._load_config()
        return self._config
    
    def get_sql_queries(self) -> Dict[str, str]:
        """
        Get SQL queries for the current environment.
        
        Returns:
            Dict[str, str]: SQL queries dictionary
        """
        if self._sql_queries is None:
            self._load_config()
        return self._sql_queries
    
    def get_rag_params(self) -> Dict[str, Any]:
        """
        Get RAG parameters for the current environment.
        
        Returns:
            Dict[str, Any]: RAG parameters dictionary
        """
        if self._rag_params is None:
            self._load_config()
        return self._rag_params
    
    def _load_config(self):
        """
        Load configuration based on the current environment.
        """
        if self.environment.lower() == 'production':
            region = os.getenv('AWS_REGION', 'us-east-1')
            self._config = get_production_config(region)
            self._sql_queries = PROD_SQL_QUERIES
            self._rag_params = PROD_RAG_PARAMS
        else:
            override_env = os.getenv('DEV_OVERRIDE')  # 'local', 'docker', etc.
            self._config = get_development_config(override_env)
            self._sql_queries = DEV_SQL_QUERIES
            self._rag_params = DEV_RAG_PARAMS
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """
        Apply environment variable overrides to the configuration.
        """
        # Database overrides
        if 'DB_SERVER' in os.environ:
            self._config['database']['server'] = os.environ['DB_SERVER']
        if 'DB_NAME' in os.environ:
            self._config['database']['database'] = os.environ['DB_NAME']
        if 'DB_USERNAME' in os.environ:
            self._config['database']['username'] = os.environ['DB_USERNAME']
        if 'DB_PASSWORD' in os.environ:
            self._config['database']['password'] = os.environ['DB_PASSWORD']
        
        # Performance overrides
        if 'MAX_WORKERS' in os.environ:
            try:
                max_workers = int(os.environ['MAX_WORKERS'])
                self._config['pipeline']['parallel_processing']['max_workers'] = max_workers
            except ValueError:
                pass
        
        # Logging overrides
        if 'LOG_LEVEL' in os.environ:
            self._config['logging']['level'] = os.environ['LOG_LEVEL'].upper()
    
    def get_database_config(self) -> Dict[str, Any]:
        """
        Get database configuration.
        
        Returns:
            Dict[str, Any]: Database configuration
        """
        return self.get_config().get('database', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration.
        
        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        return self.get_config().get('pipeline', {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """
        Get storage configuration.
        
        Returns:
            Dict[str, Any]: Storage configuration
        """
        return self.get_config().get('storage', {})
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        Get monitoring configuration.
        
        Returns:
            Dict[str, Any]: Monitoring configuration
        """
        return self.get_config().get('monitoring', {})
    
    def is_development(self) -> bool:
        """
        Check if running in development environment.
        
        Returns:
            bool: True if development environment
        """
        return self.environment.lower() == 'development'
    
    def is_production(self) -> bool:
        """
        Check if running in production environment.
        
        Returns:
            bool: True if production environment
        """
        return self.environment.lower() == 'production'


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> Dict[str, Any]:
    """
    Get the current configuration.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    return get_config_manager().get_config()


def get_sql_queries() -> Dict[str, str]:
    """
    Get SQL queries for the current environment.
    
    Returns:
        Dict[str, str]: SQL queries dictionary
    """
    return get_config_manager().get_sql_queries()


def get_rag_params() -> Dict[str, Any]:
    """
    Get RAG parameters for the current environment.
    
    Returns:
        Dict[str, Any]: RAG parameters dictionary
    """
    return get_config_manager().get_rag_params()


__all__ = [
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'get_sql_queries',
    'get_rag_params'
]