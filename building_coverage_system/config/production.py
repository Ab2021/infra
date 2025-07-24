"""
Production configuration for building coverage system.

This module contains configuration settings optimized for production
environments with emphasis on performance, reliability, and security.
"""

from typing import Dict, Any

# Production database configuration
DATABASE_CONFIG = {
    'server': 'prod-sql-cluster.company.com',
    'database': 'building_coverage_prod',
    'username': '${DB_USERNAME}',  # Environment variable
    'password': '${DB_PASSWORD}',  # Environment variable
    'driver': 'ODBC Driver 17 for SQL Server',
    'connection_timeout': 60,
    'command_timeout': 900,  # 15 minutes for large queries
    'connection_pooling': True,
    'pool_size': 10,
    'max_overflow': 20
}

# Pipeline configuration for production
PIPELINE_CONFIG = {
    'parallel_processing': {
        'max_workers': 8,  # Higher for production performance
        'batch_size': 500,  # Larger batches for efficiency
        'timeout_seconds': 1800,  # 30 minutes
        'retry_attempts': 3,
        'retry_delay_seconds': 30
    },
    'source_loading': {
        'enabled_sources': ['aip', 'atlas', 'snowflake'],  # All sources
        'retry_attempts': 5,
        'retry_delay_seconds': 60,
        'connection_pooling': True
    },
    'rag_processing': {
        'chunk_size': 1500,  # Larger chunks for better context
        'overlap_size': 150,
        'max_tokens': 4000,
        'temperature': 0.0,  # Deterministic for production
        'enable_debug_logging': False,
        'rate_limit_requests_per_minute': 60
    },
    'hooks': {
        'pre_processing_enabled': True,
        'post_processing_enabled': True,
        'pre_hook_path': 'custom_hooks/pre_processing.py',
        'post_hook_path': 'custom_hooks/post_processing.py',
        'enable_hook_monitoring': True
    },
    'validation': {
        'enable_data_quality_checks': True,
        'min_text_length': 100,  # Stricter for production
        'max_claim_age_days': 2190,  # 6 years
        'required_fields': ['CLAIMNO', 'clean_FN_TEXT', 'LOBCD'],
        'enable_schema_validation': True,
        'fail_on_validation_errors': True
    }
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',  # Less verbose for production
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    'handlers': {
        'console': {
            'enabled': False,  # Disabled in production
            'level': 'ERROR'  # Only errors to console if enabled
        },
        'file': {
            'enabled': True,
            'level': 'INFO',
            'filename': '/var/log/building_coverage/building_coverage_prod.log',
            'max_bytes': 104857600,  # 100MB
            'backup_count': 10,
            'rotation': 'time',  # Rotate by time
            'rotation_interval': 'midnight'
        },
        'syslog': {
            'enabled': True,
            'level': 'WARNING',
            'facility': 'local0',
            'address': ('syslog-server.company.com', 514)
        }
    },
    'structured_logging': {
        'enabled': True,
        'format': 'json',
        'include_trace_id': True
    }
}

# Performance monitoring configuration
MONITORING_CONFIG = {
    'enable_performance_tracking': True,
    'enable_memory_monitoring': True,
    'enable_health_checks': True,
    'metrics_collection_interval': 10,  # More frequent in prod
    'alert_thresholds': {
        'processing_time_seconds': 1800,  # 30 minutes
        'memory_usage_mb': 8192,  # 8GB
        'error_rate_percent': 1,  # Stricter error tolerance
        'queue_size': 1000,
        'cpu_usage_percent': 80
    },
    'metrics_export': {
        'prometheus': {
            'enabled': True,
            'endpoint': '/metrics',
            'port': 8080
        },
        'datadog': {
            'enabled': True,
            'api_key': '${DATADOG_API_KEY}',
            'tags': ['environment:production', 'service:building-coverage']
        }
    }
}

# Storage configuration
STORAGE_CONFIG = {
    'output_destinations': {
        'database': {
            'enabled': True,
            'table_name': 'building_coverage_predictions',
            'batch_size': 1000,  # Larger batches for production
            'enable_upsert': True,
            'enable_archiving': True,
            'archive_after_days': 90
        },
        'file_system': {
            'enabled': True,
            'output_path': '/data/building_coverage/output/',
            'file_format': 'parquet',
            'compression': 'snappy',
            'include_timestamp': True,
            'partition_by': ['processing_date'],
            'retention_days': 365
        },
        'api': {
            'enabled': True,
            'endpoint': 'https://api.company.com/building-coverage',
            'authentication': 'bearer_token',
            'timeout_seconds': 30,
            'retry_attempts': 3
        },
        's3': {
            'enabled': True,
            'bucket': 'company-building-coverage-prod',
            'prefix': 'predictions/',
            'region': 'us-east-1',
            'encryption': 'AES256'
        }
    }
}

# Feature flags for production
FEATURE_FLAGS = {
    'enable_experimental_features': False,  # Disabled in production
    'enable_advanced_validation': True,
    'enable_performance_profiling': False,  # Disabled for performance
    'enable_debug_outputs': False,
    'skip_expensive_operations': False,
    'enable_circuit_breaker': True,
    'enable_graceful_degradation': True
}

# Security configuration (production - strict)
SECURITY_CONFIG = {
    'enable_encryption': True,
    'encryption_key': '${ENCRYPTION_KEY}',
    'log_sensitive_data': False,  # Never log sensitive data in prod
    'validate_certificates': True,
    'require_authentication': True,
    'enable_audit_logging': True,
    'token_expiry_hours': 24,
    'allowed_hosts': ['*.company.com'],
    'rate_limiting': {
        'enabled': True,
        'requests_per_minute': 1000,
        'burst_limit': 100
    }
}

# High availability configuration
HA_CONFIG = {
    'enable_failover': True,
    'health_check_interval': 30,
    'max_consecutive_failures': 3,
    'circuit_breaker': {
        'failure_threshold': 5,
        'recovery_timeout': 60,
        'expected_exception_types': ['ConnectionError', 'TimeoutError']
    },
    'load_balancing': {
        'strategy': 'round_robin',
        'health_check_path': '/health',
        'timeout_seconds': 5
    }
}

# Backup and disaster recovery
BACKUP_CONFIG = {
    'enable_continuous_backup': True,
    'backup_interval_hours': 6,
    'retention_days': 30,
    'backup_locations': [
        's3://company-backups/building-coverage/',
        '/backup/building_coverage/'
    ],
    'enable_point_in_time_recovery': True,
    'cross_region_replication': True
}

# Complete production configuration
PRODUCTION_CONFIG: Dict[str, Any] = {
    'environment': 'production',
    'debug': False,
    'database': DATABASE_CONFIG,
    'pipeline': PIPELINE_CONFIG,
    'logging': LOGGING_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'storage': STORAGE_CONFIG,
    'features': FEATURE_FLAGS,
    'security': SECURITY_CONFIG,
    'high_availability': HA_CONFIG,
    'backup': BACKUP_CONFIG
}

# Multi-region configuration
REGION_CONFIGS = {
    'us-east-1': {
        'database': {
            'server': 'prod-sql-east.company.com',
        },
        'storage': {
            'output_destinations': {
                's3': {
                    'bucket': 'company-building-coverage-east',
                    'region': 'us-east-1'
                }
            }
        }
    },
    'us-west-2': {
        'database': {
            'server': 'prod-sql-west.company.com',
        },
        'storage': {
            'output_destinations': {
                's3': {
                    'bucket': 'company-building-coverage-west',
                    'region': 'us-west-2'
                }
            }
        }
    }
}


def get_production_config(region: str = 'us-east-1') -> Dict[str, Any]:
    """
    Get production configuration with region-specific overrides.
    
    Args:
        region (str): AWS region for deployment
        
    Returns:
        Dict[str, Any]: Complete production configuration
    """
    config = PRODUCTION_CONFIG.copy()
    
    # Apply region-specific overrides
    if region in REGION_CONFIGS:
        region_overrides = REGION_CONFIGS[region]
        config = _deep_merge_dict(config, region_overrides)
    
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


# Production SQL queries (optimized for performance)
PROD_SQL_QUERIES = {
    'feature_query': '''
        SELECT 
            c.CLAIMNO,
            c.CLAIMKEY,
            c.clean_FN_TEXT,
            c.LOBCD,
            c.LOSSDESC,
            c.LOSSDT,
            c.REPORTEDDT,
            c.STATUSCD,
            c.RESERVE_TOTAL
        FROM building_claims c WITH (NOLOCK)
        INNER JOIN claim_status cs ON c.CLAIMNO = cs.CLAIMNO
        WHERE c.LOBCD IN ('15', '17')
            AND c.clean_FN_TEXT IS NOT NULL
            AND LEN(c.clean_FN_TEXT) >= 100
            AND cs.STATUS = 'ACTIVE'
            AND c.LOSSDT >= DATEADD(YEAR, -6, GETDATE())
        ORDER BY c.LOSSDT DESC
    ''',
    'incremental_query': '''
        SELECT 
            c.CLAIMNO,
            c.CLAIMKEY,
            c.clean_FN_TEXT,
            c.LOBCD,
            c.LOSSDESC,
            c.LOSSDT,
            c.REPORTEDDT,
            c.LAST_MODIFIED
        FROM building_claims c WITH (NOLOCK)
        WHERE c.LOBCD IN ('15', '17')
            AND c.clean_FN_TEXT IS NOT NULL
            AND c.LAST_MODIFIED > ?
        ORDER BY c.LAST_MODIFIED
    ''',
    'validation_query': '''
        SELECT 
            COUNT(*) as total_claims,
            COUNT(CASE WHEN clean_FN_TEXT IS NOT NULL THEN 1 END) as claims_with_text,
            AVG(LEN(clean_FN_TEXT)) as avg_text_length
        FROM building_claims WITH (NOLOCK)
        WHERE LOBCD IN ('15', '17')
            AND LOSSDT >= DATEADD(YEAR, -1, GETDATE())
    '''
}

# Production RAG parameters (optimized for accuracy)
PROD_RAG_PARAMS = {
    'get_prompt': lambda: '''
        You are an expert insurance claim analyst. Analyze the following claim text to determine if it requires building coverage.
        
        Consider these factors:
        1. Structural damage to buildings, foundations, walls, roofs
        2. Building materials and construction elements
        3. Permanent fixtures and architectural features
        4. Exclude personal property, vehicles, and landscaping
        
        Provide a detailed analysis with:
        - Clear determination: BUILDING COVERAGE or NO BUILDING COVERAGE
        - Confidence score (0.0-1.0)
        - Key factors supporting your decision
        - Relevant building components mentioned
        
        Format your response as:
        Determination: [BUILDING COVERAGE/NO BUILDING COVERAGE]
        Confidence: [0.0-1.0]
        Summary: [Brief explanation]
        Key Factors: [List of supporting evidence]
    ''',
    'params_for_chunking': {
        'chunk_size': 1500,
        'chunk_overlap': 150,
        'separators': ['\n\n', '\n', '. ', '; ', ', ', ' '],
        'length_function': len
    },
    'rag_query': 'Analyze this insurance claim for building coverage requirements',
    'gpt_config_params': {
        'max_tokens': 4000,
        'temperature': 0.0,  # Deterministic for production
        'top_p': 1.0,
        'frequency_penalty': 0.0,
        'presence_penalty': 0.0,
        'stop_sequences': ['\n\n---', 'END_ANALYSIS']
    }
}