"""
Environment configuration for building coverage system.

This module provides the DatabricksEnv class that maintains compatibility
with the original Codebase 1 environment setup while integrating with
the new modular architecture.
"""

import logging
from typing import Dict, Any, Optional
from .credentials import get_database_credentials
from .sql import get_sql_queries
from .rag_params import get_rag_parameters
from .prompts import get_gpt_prompts


class DatabricksEnv:
    """
    Databricks environment configuration class.
    
    This class maintains compatibility with the original Codebase 1
    environment structure while providing integration points for
    the new modular architecture.
    """
    
    def __init__(self, databricks_dict: Dict[str, Any]):
        """
        Initialize the Databricks environment.
        
        Args:
            databricks_dict (Dict[str, Any]): Databricks configuration dictionary
        """
        self.databricks_dict = databricks_dict
        self.logger = self._setup_logger()
        
        # Initialize core components
        self.credentials_dict = get_database_credentials(databricks_dict)
        self.sql_queries = get_sql_queries()
        self.rag_params = get_rag_parameters()
        self.prompts = get_gpt_prompts()
        
        # Legacy components for backward compatibility
        self.crypto_spark = self._initialize_crypto_spark()
        self.SQL_QUERY_CONFIGS = self._get_sql_query_configs()
        
        self.logger.info("DatabricksEnv initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging for the environment.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('building_coverage.environment')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_crypto_spark(self) -> Optional[Any]:
        """
        Initialize crypto spark for legacy compatibility.
        
        Returns:
            Optional[Any]: Crypto spark instance or None
        """
        # Legacy crypto spark initialization
        # This would normally initialize the actual crypto spark component
        # For now, returning None to maintain compatibility
        return None
    
    def _get_sql_query_configs(self) -> Dict[str, Any]:
        """
        Get SQL query configurations for legacy compatibility.
        
        Returns:
            Dict[str, Any]: SQL query configurations
        """
        return {
            'timeout': 300,
            'retry_attempts': 3,
            'batch_size': 1000,
            'connection_pool_size': 5
        }
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration compatible with both old and new systems.
        
        Returns:
            Dict[str, Any]: Pipeline configuration
        """
        return {
            'credentials_dict': self.credentials_dict,
            'sql_queries': self.sql_queries,
            'rag_params': self.rag_params,
            'crypto_spark': self.crypto_spark,
            'logger': self.logger,
            'SQL_QUERY_CONFIGS': self.SQL_QUERY_CONFIGS
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the environment configuration.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate credentials
        if not self.credentials_dict or not self.credentials_dict.get('server'):
            validation_results['errors'].append('Database server not configured')
            validation_results['is_valid'] = False
        
        # Validate SQL queries
        if not self.sql_queries:
            validation_results['errors'].append('SQL queries not configured')
            validation_results['is_valid'] = False
        
        # Validate RAG parameters
        if not self.rag_params:
            validation_results['errors'].append('RAG parameters not configured')
            validation_results['is_valid'] = False
        
        # Log validation results
        if validation_results['is_valid']:
            self.logger.info("Environment configuration validation passed")
        else:
            self.logger.error(f"Environment configuration validation failed: {validation_results['errors']}")
        
        return validation_results
    
    def get_feature_queries(self) -> Dict[str, str]:
        """
        Get feature extraction queries.
        
        Returns:
            Dict[str, str]: Feature queries
        """
        return self.sql_queries.get('feature_queries', {})
    
    def get_data_sources(self) -> Dict[str, Any]:
        """
        Get configured data sources.
        
        Returns:
            Dict[str, Any]: Data source configurations
        """
        return {
            'aip': {
                'server': self.credentials_dict.get('aip_server'),
                'database': self.credentials_dict.get('aip_database'),
                'enabled': True
            },
            'atlas': {
                'server': self.credentials_dict.get('atlas_server'),
                'database': self.credentials_dict.get('atlas_database'),
                'enabled': True
            },
            'snowflake': {
                'account': self.credentials_dict.get('snowflake_account'),
                'database': self.credentials_dict.get('snowflake_database'),
                'enabled': False  # Disabled by default
            }
        }
    
    def update_configuration(self, updates: Dict[str, Any]) -> None:
        """
        Update environment configuration.
        
        Args:
            updates (Dict[str, Any]): Configuration updates
        """
        if 'credentials' in updates:
            self.credentials_dict.update(updates['credentials'])
        
        if 'sql_queries' in updates:
            self.sql_queries.update(updates['sql_queries'])
        
        if 'rag_params' in updates:
            self.rag_params.update(updates['rag_params'])
        
        self.logger.info("Environment configuration updated")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get environment configuration summary.
        
        Returns:
            Dict[str, Any]: Configuration summary
        """
        return {
            'environment_type': 'databricks',
            'has_credentials': bool(self.credentials_dict),
            'sql_queries_count': len(self.sql_queries),
            'rag_params_configured': bool(self.rag_params),
            'data_sources': list(self.get_data_sources().keys()),
            'validation_status': self.validate_configuration()['is_valid']
        }