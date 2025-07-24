"""
Configuration loader for pipeline components.

This module provides the ConfigLoader class for managing configuration
settings across different environments and components of the building
coverage system.
"""

from typing import Dict, Any, Optional
import json
import os
import logging


class ConfigLoader:
    """
    Configuration loader for pipeline components.
    
    This class handles loading and merging of configuration settings from
    various sources including environment variables, configuration files,
    and runtime overrides.
    
    Attributes:
        logger: Logging instance for configuration operations
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration loader.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance for operations
        """
        self.logger = logger if logger else logging.getLogger(__name__)
    
    @staticmethod
    def load_config(base_config: Dict[str, Any], 
                   overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration with optional overrides.
        
        This method takes a base configuration and applies optional overrides
        to create the final configuration used by the pipeline.
        
        Args:
            base_config (Dict[str, Any]): Base configuration dictionary
            overrides (Optional[Dict[str, Any]]): Configuration overrides to apply
            
        Returns:
            Dict[str, Any]: Final merged configuration
        """
        # Default pipeline configuration
        default_config = {
            'parallel_processing': {
                'enabled': True,
                'max_workers': 4,
                'min_claims_threshold': 10,
                'batch_size': 50
            },
            'source_loading': {
                'enabled_sources': ['aip', 'atlas', 'snowflake'],
                'parallel_loading': True,
                'timeout_seconds': 300,
                'retry_attempts': 3
            },
            'hooks': {
                'pre_processing_enabled': False,
                'post_processing_enabled': False,
                'pre_hook_path': None,
                'post_hook_path': None
            },
            'monitoring': {
                'performance_tracking': True,
                'log_level': 'INFO',
                'metrics_enabled': True
            },
            'rag_processing': {
                'chunk_size': 8000,
                'overlap_size': 200,
                'max_tokens': 4000,
                'temperature': 0.1
            },
            'rules_engine': {
                'confidence_threshold': 0.7,
                'validation_enabled': True
            }
        }
        
        # Merge with base configuration
        config = {**base_config, 'pipeline': default_config}
        
        # Apply overrides if provided
        if overrides:
            config = ConfigLoader.merge_configs(config, overrides)
        
        return config
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration dictionaries recursively.
        
        This method performs a deep merge of configuration dictionaries,
        with override values taking precedence over base values.
        
        Args:
            base (Dict[str, Any]): Base configuration dictionary
            overrides (Dict[str, Any]): Override configuration dictionary
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @classmethod
    def load_from_file(cls, config_path: str, 
                      base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path (str): Path to the configuration JSON file
            base_config (Optional[Dict[str, Any]]): Base configuration to merge with
            
        Returns:
            Dict[str, Any]: Loaded and merged configuration
            
        Raises:
            FileNotFoundError: If configuration file is not found
            json.JSONDecodeError: If configuration file is invalid JSON
        """
        logger = logging.getLogger(__name__)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            logger.info(f"Successfully loaded configuration from {config_path}")
            
            if base_config:
                return cls.load_config(base_config, file_config)
            else:
                return cls.load_config({}, file_config)
                
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
            raise
    
    @classmethod
    def load_from_env(cls, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        This method reads configuration settings from environment variables
        following a specific naming convention.
        
        Args:
            base_config (Optional[Dict[str, Any]]): Base configuration to merge with
            
        Returns:
            Dict[str, Any]: Configuration with environment variable overrides
        """
        logger = logging.getLogger(__name__)
        
        env_config = {}
        
        # Pipeline configuration from environment
        if os.getenv('PIPELINE_MAX_WORKERS'):
            env_config.setdefault('pipeline', {}).setdefault('parallel_processing', {})['max_workers'] = int(os.getenv('PIPELINE_MAX_WORKERS'))
        
        if os.getenv('PIPELINE_PARALLEL_ENABLED'):
            env_config.setdefault('pipeline', {}).setdefault('parallel_processing', {})['enabled'] = os.getenv('PIPELINE_PARALLEL_ENABLED').lower() == 'true'
        
        # Source loading configuration
        if os.getenv('SOURCE_TIMEOUT'):
            env_config.setdefault('pipeline', {}).setdefault('source_loading', {})['timeout_seconds'] = int(os.getenv('SOURCE_TIMEOUT'))
        
        if os.getenv('ENABLED_SOURCES'):
            sources = os.getenv('ENABLED_SOURCES').split(',')
            env_config.setdefault('pipeline', {}).setdefault('source_loading', {})['enabled_sources'] = [s.strip() for s in sources]
        
        # Hook configuration
        if os.getenv('PRE_HOOK_PATH'):
            env_config.setdefault('pipeline', {}).setdefault('hooks', {})['pre_hook_path'] = os.getenv('PRE_HOOK_PATH')
            env_config.setdefault('pipeline', {}).setdefault('hooks', {})['pre_processing_enabled'] = True
        
        if os.getenv('POST_HOOK_PATH'):
            env_config.setdefault('pipeline', {}).setdefault('hooks', {})['post_hook_path'] = os.getenv('POST_HOOK_PATH')
            env_config.setdefault('pipeline', {}).setdefault('hooks', {})['post_processing_enabled'] = True
        
        # Logging configuration
        if os.getenv('LOG_LEVEL'):
            env_config.setdefault('pipeline', {}).setdefault('monitoring', {})['log_level'] = os.getenv('LOG_LEVEL')
        
        logger.info(f"Loaded configuration from environment variables: {len(env_config)} settings")
        
        if base_config:
            return cls.load_config(base_config, env_config)
        else:
            return cls.load_config({}, env_config)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration settings.
        
        This method validates the configuration settings and returns a summary
        of validation results including any errors or warnings.
        
        Args:
            config (Dict[str, Any]): Configuration to validate
            
        Returns:
            Dict[str, Any]: Validation results with errors and warnings
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate pipeline configuration
        if 'pipeline' in config:
            pipeline_config = config['pipeline']
            
            # Check parallel processing settings
            if 'parallel_processing' in pipeline_config:
                pp_config = pipeline_config['parallel_processing']
                
                if 'max_workers' in pp_config:
                    max_workers = pp_config['max_workers']
                    if not isinstance(max_workers, int) or max_workers < 1:
                        validation_results['errors'].append("max_workers must be a positive integer")
                        validation_results['is_valid'] = False
                    elif max_workers > 16:
                        validation_results['warnings'].append("max_workers > 16 may cause resource contention")
            
            # Check source loading settings
            if 'source_loading' in pipeline_config:
                sl_config = pipeline_config['source_loading']
                
                if 'enabled_sources' in sl_config:
                    valid_sources = ['aip', 'atlas', 'snowflake']
                    enabled_sources = sl_config['enabled_sources']
                    
                    if not isinstance(enabled_sources, list):
                        validation_results['errors'].append("enabled_sources must be a list")
                        validation_results['is_valid'] = False
                    else:
                        invalid_sources = [s for s in enabled_sources if s not in valid_sources]
                        if invalid_sources:
                            validation_results['errors'].append(f"Invalid sources: {invalid_sources}")
                            validation_results['is_valid'] = False
            
            # Check hook settings
            if 'hooks' in pipeline_config:
                hooks_config = pipeline_config['hooks']
                
                if hooks_config.get('pre_processing_enabled') and not hooks_config.get('pre_hook_path'):
                    validation_results['warnings'].append("Pre-processing enabled but no hook path provided")
                
                if hooks_config.get('post_processing_enabled') and not hooks_config.get('post_hook_path'):
                    validation_results['warnings'].append("Post-processing enabled but no hook path provided")
        
        return validation_results
    
    @staticmethod
    def get_config_summary(config: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of the configuration.
        
        Args:
            config (Dict[str, Any]): Configuration to summarize
            
        Returns:
            str: Human-readable configuration summary
        """
        summary_lines = ["Configuration Summary:"]
        
        if 'pipeline' in config:
            pipeline_config = config['pipeline']
            
            # Parallel processing summary
            if 'parallel_processing' in pipeline_config:
                pp_config = pipeline_config['parallel_processing']
                enabled = pp_config.get('enabled', False)
                workers = pp_config.get('max_workers', 4)
                summary_lines.append(f"  Parallel Processing: {'Enabled' if enabled else 'Disabled'} ({workers} workers)")
            
            # Source loading summary
            if 'source_loading' in pipeline_config:
                sl_config = pipeline_config['source_loading']
                sources = sl_config.get('enabled_sources', [])
                parallel = sl_config.get('parallel_loading', False)
                summary_lines.append(f"  Source Loading: {len(sources)} sources {'(parallel)' if parallel else '(sequential)'}")
                summary_lines.append(f"    Sources: {', '.join(sources)}")
            
            # Hooks summary
            if 'hooks' in pipeline_config:
                hooks_config = pipeline_config['hooks']
                pre_enabled = hooks_config.get('pre_processing_enabled', False)
                post_enabled = hooks_config.get('post_processing_enabled', False)
                summary_lines.append(f"  Custom Hooks: Pre-processing {'✓' if pre_enabled else '✗'}, Post-processing {'✓' if post_enabled else '✗'}")
        
        return "\\n".join(summary_lines)


class EnvironmentConfigLoader(ConfigLoader):
    """
    Environment-specific configuration loader.
    
    This class extends ConfigLoader to handle environment-specific configuration
    loading for development, staging, and production environments.
    """
    
    def __init__(self, environment: str = 'development', logger: Optional[logging.Logger] = None):
        """
        Initialize environment-specific configuration loader.
        
        Args:
            environment (str): Environment name (development, staging, production)
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__(logger)
        self.environment = environment
    
    def load_environment_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load environment-specific configuration.
        
        Args:
            base_config (Dict[str, Any]): Base configuration
            
        Returns:
            Dict[str, Any]: Environment-specific configuration
        """
        env_overrides = {}
        
        if self.environment == 'development':
            env_overrides = {
                'pipeline': {
                    'parallel_processing': {
                        'enabled': False,  # Easier debugging
                        'max_workers': 2
                    },
                    'source_loading': {
                        'enabled_sources': ['aip'],  # Single source for dev
                        'timeout_seconds': 60
                    },
                    'monitoring': {
                        'log_level': 'DEBUG'
                    }
                }
            }
        elif self.environment == 'staging':
            env_overrides = {
                'pipeline': {
                    'parallel_processing': {
                        'enabled': True,
                        'max_workers': 4
                    },
                    'source_loading': {
                        'enabled_sources': ['aip', 'atlas'],
                        'timeout_seconds': 180
                    }
                }
            }
        elif self.environment == 'production':
            env_overrides = {
                'pipeline': {
                    'parallel_processing': {
                        'enabled': True,
                        'max_workers': 8
                    },
                    'source_loading': {
                        'enabled_sources': ['aip', 'atlas', 'snowflake'],
                        'timeout_seconds': 300
                    },
                    'monitoring': {
                        'metrics_enabled': True
                    }
                }
            }
        
        return self.load_config(base_config, env_overrides)