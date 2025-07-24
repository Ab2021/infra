"""
RAG parameters configuration for building coverage system.

This module contains configuration parameters for the RAG (Retrieval-Augmented Generation)
processing components, including chunking, GPT settings, and processing options.
"""

from typing import Dict, Any, Callable, Optional
from .prompts import get_building_coverage_prompt


def get_rag_parameters() -> Dict[str, Any]:
    """
    Get RAG processing parameters.
    
    Returns:
        Dict[str, Any]: Complete RAG parameters configuration
    """
    return {
        'get_prompt': get_default_prompt_function(),
        'params_for_chunking': get_chunking_parameters(),
        'rag_query': get_default_rag_query(),
        'gpt_config_params': get_gpt_configuration(),
        'processing_options': get_processing_options(),
        'validation_settings': get_validation_settings()
    }


def get_default_prompt_function() -> Callable[[], str]:
    """
    Get the default prompt function for building coverage analysis.
    
    Returns:
        Callable[[], str]: Function that returns the main analysis prompt
    """
    def prompt_function() -> str:
        return get_building_coverage_prompt()
    
    return prompt_function


def get_chunking_parameters() -> Dict[str, Any]:
    """
    Get text chunking parameters for RAG processing.
    
    Returns:
        Dict[str, Any]: Chunking configuration parameters
    """
    return {
        'chunk_size': 1500,  # Maximum characters per chunk
        'chunk_overlap': 150,  # Overlap between consecutive chunks
        'separators': ["\n\n", "\n", ". ", "; ", ", ", " "],  # Text splitting separators
        'length_function': len,  # Function to measure text length
        'keep_separator': True,  # Whether to keep separators in chunks
        'add_start_index': True,  # Add character start index to chunk metadata
        'strip_whitespace': True,  # Remove leading/trailing whitespace
        'min_chunk_size': 100,  # Minimum characters for a valid chunk
        'max_chunks_per_claim': 10,  # Maximum chunks per claim
        'prioritize_sentences': True,  # Try to keep sentences intact
        'metadata_fields': [  # Fields to preserve in chunk metadata
            'CLAIMNO',
            'LOBCD', 
            'LOSSDT',
            'source'
        ]
    }


def get_default_rag_query() -> str:
    """
    Get the default RAG query for building coverage analysis.
    
    Returns:
        str: Default RAG query string
    """
    return "Analyze this insurance claim text to determine if it requires building coverage based on structural damage, building materials, and permanent fixtures mentioned."


def get_gpt_configuration() -> Dict[str, Any]:
    """
    Get GPT model configuration parameters.
    
    Returns:
        Dict[str, Any]: GPT configuration settings
    """
    return {
        # Model settings
        'model': 'gpt-4',  # Model version to use
        'max_tokens': 4000,  # Maximum tokens in response
        'temperature': 0.1,  # Response randomness (0.0 = deterministic)
        'top_p': 0.95,  # Nucleus sampling parameter
        'frequency_penalty': 0.0,  # Penalty for frequent tokens
        'presence_penalty': 0.0,  # Penalty for new tokens
        
        # Response formatting
        'stop_sequences': [  # Sequences that stop generation
            "\n\n---",
            "END_ANALYSIS",
            "<END>"
        ],
        
        # API settings
        'timeout_seconds': 30,  # Request timeout
        'retry_attempts': 3,  # Number of retries on failure
        'retry_delay_seconds': 2,  # Delay between retries
        'rate_limit_requests_per_minute': 60,  # Rate limiting
        
        # Response validation
        'validate_response_format': True,  # Validate response structure
        'required_response_fields': [  # Required fields in response
            'DETERMINATION',
            'CONFIDENCE',
            'SUMMARY'
        ],
        
        # Backup models
        'fallback_models': [
            'gpt-3.5-turbo',
            'gpt-4-turbo-preview'
        ],
        
        # Cost optimization
        'use_cheaper_model_for_simple_cases': True,
        'simple_case_threshold': 500,  # Character threshold for simple cases
        'cost_tracking_enabled': True
    }


def get_processing_options() -> Dict[str, Any]:
    """
    Get RAG processing options and configurations.
    
    Returns:
        Dict[str, Any]: Processing options
    """
    return {
        # Parallel processing
        'enable_parallel_processing': True,
        'max_concurrent_requests': 4,
        'batch_size': 20,  # Claims per processing batch
        'processing_timeout_seconds': 300,  # 5 minutes per batch
        
        # Caching
        'enable_response_caching': True,
        'cache_ttl_hours': 24,  # Cache time-to-live
        'cache_key_fields': ['clean_FN_TEXT', 'model', 'temperature'],
        
        # Quality controls
        'enable_confidence_thresholding': True,
        'min_confidence_threshold': 0.5,
        'high_confidence_threshold': 0.8,
        'require_manual_review_below': 0.6,
        
        # Error handling
        'continue_on_error': True,
        'max_errors_per_batch': 5,
        'log_failed_requests': True,
        'save_failed_requests_for_retry': True,
        
        # Output formatting
        'standardize_output_format': True,
        'include_processing_metadata': True,
        'add_timestamp': True,
        'preserve_original_text': True,
        
        # Performance monitoring
        'track_processing_time': True,
        'track_token_usage': True,
        'track_cost_per_request': True,
        'enable_performance_alerts': True
    }


def get_validation_settings() -> Dict[str, Any]:
    """
    Get response validation settings.
    
    Returns:
        Dict[str, Any]: Validation configuration
    """
    return {
        # Response format validation
        'validate_determination_format': True,
        'valid_determinations': [
            'BUILDING COVERAGE',
            'NO BUILDING COVERAGE',
            'UNCLEAR'
        ],
        
        # Confidence validation
        'validate_confidence_range': True,
        'confidence_min': 0.0,
        'confidence_max': 1.0,
        
        # Content validation
        'min_summary_length': 20,
        'max_summary_length': 1000,
        'require_key_factors': True,
        'min_key_factors': 1,
        
        # Consistency checks
        'check_determination_confidence_alignment': True,
        'flag_low_confidence_positive_determinations': True,
        'flag_high_confidence_unclear_determinations': True,
        
        # Business rule validation
        'validate_against_lob_codes': True,
        'lob_code_rules': {
            '15': {'typically_building': True, 'min_confidence': 0.7},
            '17': {'typically_building': True, 'min_confidence': 0.7},
            '18': {'typically_building': False, 'min_confidence': 0.8}
        },
        
        # Quality flags
        'flag_unusual_responses': True,
        'unusual_response_indicators': [
            'response_too_short',
            'missing_required_fields',
            'confidence_determination_mismatch',
            'generic_or_templated_response'
        ]
    }


def get_environment_specific_params(environment: str) -> Dict[str, Any]:
    """
    Get environment-specific RAG parameters.
    
    Args:
        environment (str): Environment name ('development', 'staging', 'production')
        
    Returns:
        Dict[str, Any]: Environment-specific parameters
    """
    base_params = get_rag_parameters()
    
    if environment == 'development':
        # Development settings - more verbose, faster processing
        base_params['gpt_config_params'].update({
            'model': 'gpt-3.5-turbo',  # Faster, cheaper model
            'max_tokens': 2000,
            'temperature': 0.2,  # Slightly more creative for testing
            'timeout_seconds': 15
        })
        base_params['processing_options'].update({
            'max_concurrent_requests': 2,
            'batch_size': 10,
            'enable_response_caching': False,  # Disable for testing
            'log_failed_requests': True
        })
    
    elif environment == 'staging':
        # Staging settings - production-like but more monitoring
        base_params['gpt_config_params'].update({
            'model': 'gpt-4',
            'max_tokens': 3000,
            'temperature': 0.05,
            'retry_attempts': 2
        })
        base_params['processing_options'].update({
            'max_concurrent_requests': 3,
            'batch_size': 15,
            'enable_performance_alerts': True
        })
    
    elif environment == 'production':
        # Production settings - optimized for reliability and performance
        base_params['gpt_config_params'].update({
            'model': 'gpt-4',
            'max_tokens': 4000,
            'temperature': 0.0,  # Deterministic
            'retry_attempts': 5,
            'timeout_seconds': 45
        })
        base_params['processing_options'].update({
            'max_concurrent_requests': 8,
            'batch_size': 50,
            'enable_response_caching': True,
            'cache_ttl_hours': 72
        })
    
    return base_params


def create_custom_rag_params(
    base_environment: str = 'production',
    custom_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create custom RAG parameters with overrides.
    
    Args:
        base_environment (str): Base environment to start from
        custom_overrides (Optional[Dict[str, Any]]): Custom parameter overrides
        
    Returns:
        Dict[str, Any]: Custom RAG parameters
    """
    params = get_environment_specific_params(base_environment)
    
    if custom_overrides:
        # Deep merge custom overrides
        params = _deep_merge_params(params, custom_overrides)
    
    return params


def _deep_merge_params(base_params: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge parameter dictionaries.
    
    Args:
        base_params (Dict[str, Any]): Base parameters
        overrides (Dict[str, Any]): Override parameters
        
    Returns:
        Dict[str, Any]: Merged parameters
    """
    result = base_params.copy()
    
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_params(result[key], value)
        else:
            result[key] = value
    
    return result


def validate_rag_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate RAG parameters configuration.
    
    Args:
        params (Dict[str, Any]): RAG parameters to validate
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required top-level keys
    required_keys = ['get_prompt', 'params_for_chunking', 'rag_query', 'gpt_config_params']
    for key in required_keys:
        if key not in params:
            validation_result['errors'].append(f"Missing required parameter: {key}")
            validation_result['is_valid'] = False
    
    # Validate GPT config
    if 'gpt_config_params' in params:
        gpt_config = params['gpt_config_params']
        
        if gpt_config.get('max_tokens', 0) <= 0:
            validation_result['errors'].append("max_tokens must be positive")
            validation_result['is_valid'] = False
        
        if not 0.0 <= gpt_config.get('temperature', 0.1) <= 2.0:
            validation_result['errors'].append("temperature must be between 0.0 and 2.0")
            validation_result['is_valid'] = False
    
    # Validate chunking params
    if 'params_for_chunking' in params:
        chunk_config = params['params_for_chunking']
        
        if chunk_config.get('chunk_size', 0) <= 0:
            validation_result['errors'].append("chunk_size must be positive")
            validation_result['is_valid'] = False
        
        if chunk_config.get('chunk_overlap', 0) < 0:
            validation_result['errors'].append("chunk_overlap cannot be negative")
            validation_result['is_valid'] = False
    
    # Warnings for suboptimal settings
    if 'gpt_config_params' in params:
        gpt_config = params['gpt_config_params']
        
        if gpt_config.get('temperature', 0.1) > 0.5:
            validation_result['warnings'].append("High temperature may reduce consistency")
        
        if gpt_config.get('max_tokens', 4000) > 8000:
            validation_result['warnings'].append("Very high max_tokens may increase costs")
    
    return validation_result


def get_rag_params_summary(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of RAG parameters configuration.
    
    Args:
        params (Dict[str, Any]): RAG parameters
        
    Returns:
        Dict[str, Any]: Parameters summary
    """
    summary = {
        'has_prompt_function': callable(params.get('get_prompt')),
        'chunking_configured': bool(params.get('params_for_chunking')),
        'gpt_model': params.get('gpt_config_params', {}).get('model', 'unknown'),
        'max_tokens': params.get('gpt_config_params', {}).get('max_tokens', 0),
        'temperature': params.get('gpt_config_params', {}).get('temperature', 0),
        'chunk_size': params.get('params_for_chunking', {}).get('chunk_size', 0),
        'parallel_processing': params.get('processing_options', {}).get('enable_parallel_processing', False),
        'validation_enabled': bool(params.get('validation_settings')),
        'estimated_cost_per_1k_tokens': _estimate_cost_per_1k_tokens(params)
    }
    
    return summary


def _estimate_cost_per_1k_tokens(params: Dict[str, Any]) -> float:
    """
    Estimate cost per 1000 tokens based on model configuration.
    
    Args:
        params (Dict[str, Any]): RAG parameters
        
    Returns:
        float: Estimated cost per 1000 tokens in USD
    """
    model = params.get('gpt_config_params', {}).get('model', 'gpt-3.5-turbo')
    
    # Approximate pricing (as of 2024)
    model_pricing = {
        'gpt-3.5-turbo': 0.002,
        'gpt-4': 0.03,
        'gpt-4-turbo': 0.01,
        'gpt-4-turbo-preview': 0.01
    }
    
    return model_pricing.get(model, 0.01)  # Default estimate