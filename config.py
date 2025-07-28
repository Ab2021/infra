"""
Configuration file for Agentic Building Coverage Analysis
GPT-4o-mini optimized settings
"""

import os
import re
import logging
from typing import Dict, Any, Optional


class GPTConfig:
    """GPT-4o-mini configuration settings"""
    
    # Model settings
    MODEL_NAME = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 2000
    
    # API settings
    API_KEY = None  # Will be set securely via get_secure_api_key()
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    TIMEOUT = 30.0
    
    # Token limits by task type
    TOKEN_LIMITS = {
        "context_analysis": 1500,
        "indicators_extraction": 3000,
        "candidates_extraction": 2500,
        "validation_reflection": 2000,
        "quality_reflection": 1500
    }
    
    # Temperature settings by task type
    TEMPERATURE_SETTINGS = {
        "context_analysis": 0.2,
        "indicators_extraction": 0.1,
        "candidates_extraction": 0.1,
        "validation_reflection": 0.1,
        "quality_reflection": 0.1
    }
    
    # System messages for different tasks
    SYSTEM_MESSAGES = {
        "context_analysis": "You are an expert building coverage analyst. Respond with valid JSON containing analysis and extraction strategy recommendations.",
        "indicators_extraction": "You are an expert building damage assessor. Extract all 21 indicators with Y/N values, confidence scores, and evidence. Respond with valid JSON only.",
        "candidates_extraction": "You are an expert financial analyst for insurance claims. Extract monetary candidates with hierarchical prioritization. Respond with valid JSON only.",
        "validation_reflection": "You are an expert validation specialist for building coverage analysis. Apply all validation rules and provide final assessment. Respond with valid JSON only.",
        "quality_reflection": "You are an expert quality assessment specialist for financial calculations. Provide detailed quality reflection and assessment. Respond with valid JSON only."
    }


class AgenticConfig:
    """Agentic system configuration"""
    
    # Memory settings
    MEMORY_LIMIT = 500
    SIMILARITY_THRESHOLD = 0.7
    SIMILAR_CLAIMS_LIMIT = 5
    
    # Validation settings
    MIN_CONFIDENCE = 0.6
    MAX_CONFIDENCE = 0.95
    QUALITY_THRESHOLD = 0.7
    
    # Feature analysis settings
    DAMAGE_MULTIPLIERS = {
        "extensive": 1.5,
        "moderate": 1.2,
        "limited": 1.0,
        "minimal": 0.8
    }
    
    OPERATIONAL_MULTIPLIERS = {
        "total_loss": 2.0,
        "major_impact": 1.5,
        "significant_impact": 1.3,
        "functional": 1.0
    }
    
    # Expected loss ranges (base values)
    BASE_LOSS_RANGE = (1000, 50000)
    
    # Processing timeouts
    STAGE1_TIMEOUT = 120  # seconds
    STAGE2_TIMEOUT = 90   # seconds
    TOTAL_TIMEOUT = 300   # seconds


class LoggingConfig:
    """Logging configuration"""
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "agentic_coverage_analysis.log"
    
    # What to log
    LOG_API_CALLS = True
    LOG_PROCESSING_TIMES = True
    LOG_CONFIDENCE_SCORES = True
    LOG_VALIDATION_RESULTS = True


def get_gpt_config() -> Dict[str, Any]:
    """Get GPT configuration dictionary"""
    return {
        "model": GPTConfig.MODEL_NAME,
        "default_temperature": GPTConfig.DEFAULT_TEMPERATURE,
        "default_max_tokens": GPTConfig.DEFAULT_MAX_TOKENS,
        "api_key": GPTConfig.API_KEY,
        "max_retries": GPTConfig.MAX_RETRIES,
        "timeout": GPTConfig.TIMEOUT,
        "token_limits": GPTConfig.TOKEN_LIMITS,
        "temperature_settings": GPTConfig.TEMPERATURE_SETTINGS,
        "system_messages": GPTConfig.SYSTEM_MESSAGES
    }


def get_agentic_config() -> Dict[str, Any]:
    """Get agentic system configuration dictionary"""
    return {
        "memory_limit": AgenticConfig.MEMORY_LIMIT,
        "similarity_threshold": AgenticConfig.SIMILARITY_THRESHOLD,
        "min_confidence": AgenticConfig.MIN_CONFIDENCE,
        "max_confidence": AgenticConfig.MAX_CONFIDENCE,
        "quality_threshold": AgenticConfig.QUALITY_THRESHOLD,
        "damage_multipliers": AgenticConfig.DAMAGE_MULTIPLIERS,
        "operational_multipliers": AgenticConfig.OPERATIONAL_MULTIPLIERS,
        "base_loss_range": AgenticConfig.BASE_LOSS_RANGE,
        "timeouts": {
            "stage1": AgenticConfig.STAGE1_TIMEOUT,
            "stage2": AgenticConfig.STAGE2_TIMEOUT,
            "total": AgenticConfig.TOTAL_TIMEOUT
        }
    }


def get_secure_api_key() -> str:
    """Securely retrieve and validate API key"""
    logger = logging.getLogger(__name__)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Validation checks
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("API key not found in environment variables")
    
    if not isinstance(api_key, str):
        logger.error("API key must be a string")
        raise ValueError("Invalid API key format")
    
    # Basic format validation (OpenAI keys start with 'sk-' and are ~51 chars)
    if not re.match(r'^sk-[A-Za-z0-9]{48}$', api_key):
        logger.error("API key format validation failed")
        raise ValueError("API key format is invalid")
    
    # Check for common test/placeholder keys
    placeholder_patterns = [
        'your-api-key', 'test-key', 'demo-key', 'placeholder',
        'sk-0000', 'sk-1111', 'sk-xxxx'
    ]
    
    if any(pattern in api_key.lower() for pattern in placeholder_patterns):
        logger.error("Placeholder API key detected")
        raise ValueError("Placeholder API key is not valid")
    
    logger.info("API key validation successful")
    return api_key


def validate_config() -> bool:
    """Validate configuration settings with enhanced security checks"""
    logger = logging.getLogger(__name__)
    
    try:
        # Validate API key
        api_key = get_secure_api_key()
        GPTConfig.API_KEY = api_key
        logger.info("API key validation passed")
        
    except ValueError as e:
        logger.error(f"API key validation failed: {e}")
        return False
    
    # Validate temperature settings
    if not (0 <= GPTConfig.DEFAULT_TEMPERATURE <= 2):
        logger.error("Invalid default temperature setting")
        return False
    
    for task, temp in GPTConfig.TEMPERATURE_SETTINGS.items():
        if not (0 <= temp <= 2):
            logger.error(f"Invalid temperature setting for task '{task}': {temp}")
            return False
    
    # Validate token limits
    for task, tokens in GPTConfig.TOKEN_LIMITS.items():
        if not (1 <= tokens <= 8000):
            logger.error(f"Invalid token limit for task '{task}': {tokens}")
            return False
    
    # Validate confidence thresholds
    if AgenticConfig.MIN_CONFIDENCE >= AgenticConfig.MAX_CONFIDENCE:
        logger.error("Invalid confidence threshold settings")
        return False
    
    if not (0 <= AgenticConfig.MIN_CONFIDENCE <= 1):
        logger.error("Min confidence must be between 0 and 1")
        return False
    
    if not (0 <= AgenticConfig.MAX_CONFIDENCE <= 1):
        logger.error("Max confidence must be between 0 and 1")
        return False
    
    # Validate timeouts
    timeouts = [
        AgenticConfig.STAGE1_TIMEOUT,
        AgenticConfig.STAGE2_TIMEOUT, 
        AgenticConfig.TOTAL_TIMEOUT
    ]
    
    for timeout in timeouts:
        if not (1 <= timeout <= 3600):  # 1 second to 1 hour
            logger.error(f"Invalid timeout setting: {timeout}")
            return False
    
    logger.info("Configuration validation passed")
    return True


# Export main configurations
__all__ = [
    "GPTConfig",
    "AgenticConfig", 
    "LoggingConfig",
    "get_gpt_config",
    "get_agentic_config",
    "validate_config"
]