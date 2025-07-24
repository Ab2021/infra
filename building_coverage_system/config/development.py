"""
Development configuration for building coverage system.

This module contains configuration settings optimized for development
and testing environments.
"""

from typing import Dict, Any

# Development database configuration
DATABASE_CONFIG = {
    'server': 'dev-sql-server.company.com',
    'database': 'building_coverage_dev',
    'username': '${DB_USERNAME}',  # Environment variable
    'password': '${DB_PASSWORD}',  # Environment variable
    'driver': 'ODBC Driver 17 for SQL Server',
    'connection_timeout': 30,
    'command_timeout': 300
}

# Pipeline configuration for development
PIPELINE_CONFIG = {
    'parallel_processing': {
        'max_workers': 2,  # Lower for dev to avoid resource contention
        'batch_size': 50,
        'timeout_seconds': 300
    },
    'source_loading': {
        'enabled_sources': ['aip', 'atlas'],  # Limited sources for dev
        'retry_attempts': 2,
        'retry_delay_seconds': 5
    },
    'rag_processing': {
        'chunk_size': 500,  # Smaller chunks for faster processing
        'overlap_size': 50,
        'max_tokens': 2000,  # Lower token limit for dev
        'temperature': 0.1,
        'enable_debug_logging': True
    },
    'hooks': {
        'pre_processing_enabled': True,
        'post_processing_enabled': True,
        'pre_hook_path': 'custom_hooks/pre_processing.py',
        'post_hook_path': 'custom_hooks/post_processing.py'
    },
    'validation': {
        'enable_data_quality_checks': True,
        'min_text_length': 10,  # Relaxed for dev
        'max_claim_age_days': 3650,
        'required_fields': ['CLAIMNO', 'clean_FN_TEXT']
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'DEBUG',  # Verbose logging for development
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'console': {
            'enabled': True,
            'level': 'DEBUG'
        },
        'file': {
            'enabled': True,
            'level': 'INFO',
            'filename': 'logs/building_coverage_dev.log',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        }
    }
}

# Performance monitoring configuration
MONITORING_CONFIG = {
    'enable_performance_tracking': True,
    'enable_memory_monitoring': True,
    'metrics_collection_interval': 30,  # seconds
    'alert_thresholds': {
        'processing_time_seconds': 600,  # 10 minutes
        'memory_usage_mb': 2048,  # 2GB
        'error_rate_percent': 5
    }
}

# Storage configuration
STORAGE_CONFIG = {
    'output_destinations': {
        'database': {
            'enabled': True,
            'table_name': 'building_coverage_predictions_dev',
            'batch_size': 100
        },
        'file_system': {
            'enabled': True,
            'output_path': 'output/dev/',
            'file_format': 'parquet',
            'include_timestamp': True
        },
        'api': {
            'enabled': False  # Disabled in dev to avoid external calls
        }
    }
}

# Feature flags for development
FEATURE_FLAGS = {
    'enable_experimental_features': True,
    'enable_advanced_validation': True,
    'enable_performance_profiling': True,
    'enable_debug_outputs': True,
    'skip_expensive_operations': True  # Skip slow operations in dev
}

# Test data configuration
TEST_CONFIG = {
    'use_sample_data': True,
    'sample_size': 100,
    'mock_external_services': True,
    'enable_test_hooks': True
}

# Security configuration (development - less strict)
SECURITY_CONFIG = {
    'enable_encryption': False,  # Disabled for easier debugging
    'log_sensitive_data': True,  # Enabled for debugging (never in prod!)
    'validate_certificates': False,  # Relaxed for dev environment
    'require_authentication': False
}

# Complete development configuration
DEVELOPMENT_CONFIG: Dict[str, Any] = {
    'environment': 'development',
    'debug': True,
    'database': DATABASE_CONFIG,
    'pipeline': PIPELINE_CONFIG,
    'logging': LOGGING_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'storage': STORAGE_CONFIG,
    'features': FEATURE_FLAGS,
    'testing': TEST_CONFIG,
    'security': SECURITY_CONFIG
}

# Environment-specific overrides
ENVIRONMENT_OVERRIDES = {
    'local': {
        'database': {
            'server': 'localhost',
            'database': 'building_coverage_local'
        },
        'pipeline': {
            'parallel_processing': {
                'max_workers': 1  # Single worker for local development
            }
        }
    },
    'docker': {
        'database': {
            'server': 'db-container',
            'database': 'building_coverage_docker'
        },
        'storage': {
            'output_destinations': {
                'file_system': {
                    'output_path': '/app/output/'
                }
            }
        }
    }
}


def get_development_config(override_env: str = None) -> Dict[str, Any]:
    """
    Get development configuration with optional environment-specific overrides.
    
    Args:
        override_env (str, optional): Environment override ('local', 'docker')
        
    Returns:
        Dict[str, Any]: Complete development configuration
    """
    config = DEVELOPMENT_CONFIG.copy()
    
    # Apply environment-specific overrides
    if override_env and override_env in ENVIRONMENT_OVERRIDES:
        overrides = ENVIRONMENT_OVERRIDES[override_env]
        config = _deep_merge_dict(config, overrides)
    
    return config


def _deep_merge_dict(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base_dict (Dict[str, Any]): Base dictionary
        override_dict (Dict[str, Any]): Override dictionary
        
    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


# SQL queries for development environment
DEV_SQL_QUERIES = {
    'feature_query': '''
        SELECT TOP 1000
            CLAIMNO,
            CLAIMKEY,
            clean_FN_TEXT,
            LOBCD,
            LOSSDESC,
            LOSSDT,
            REPORTEDDT
        FROM building_claims_dev
        WHERE LOBCD IN ('15', '17')
            AND clean_FN_TEXT IS NOT NULL
            AND LEN(clean_FN_TEXT) > 10
        ORDER BY LOSSDT DESC
    ''',
    'validation_query': '''
        SELECT COUNT(*) as claim_count
        FROM building_claims_dev
        WHERE LOBCD IN ('15', '17')
    ''',
    'sample_query': '''
        SELECT TOP 50
            CLAIMNO,
            clean_FN_TEXT,
            LOBCD
        FROM building_claims_dev
        WHERE LOBCD = '15'
            AND clean_FN_TEXT LIKE '%building%'
        ORDER BY NEWID()  -- Random sample
    '''
}

# RAG parameters for development
DEV_RAG_PARAMS = {
    'get_prompt': lambda: '''
        Analyze this insurance claim text and determine if it involves building coverage.
        
        Provide a brief summary and confidence score.
        Focus on structural damage, building materials, and property damage.
        
        Return format:
        Summary: [brief summary]
        Confidence: [0.0-1.0]
    ''',
    'params_for_chunking': {
        'chunk_size': 500,
        'chunk_overlap': 50,
        'separators': ['\n\n', '\n', '. ', ', ']
    },
    'rag_query': 'Does this claim involve building coverage?',
    'gpt_config_params': {
        'max_tokens': 2000,
        'temperature': 0.1,
        'top_p': 0.9,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0
    }
}