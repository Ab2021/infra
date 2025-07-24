"""
Pre-processing hooks for building coverage claims.

This module provides pre-processing functions that can be used as hooks
in the building coverage pipeline, similar to Codebase 2's pre_process hook.
These functions are called before the main RAG processing to clean and
prepare the data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
import re
import logging

# Set up logging for the hook
logger = logging.getLogger(__name__)


def pre_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main pre-processing hook for building coverage claims.
    
    This function performs comprehensive data cleaning and preparation
    similar to Codebase 2's pre_process function. It includes data quality
    improvements, filtering, and feature engineering.
    
    Args:
        df (pd.DataFrame): Input dataframe containing raw claims data
        
    Returns:
        pd.DataFrame: Processed dataframe ready for RAG processing
    """
    logger.info(f"Starting pre-processing for {len(df)} claims")
    
    # Make a copy to avoid modifying the original data
    df_processed = df.copy()
    
    # Step 1: Data type conversions and date processing
    df_processed = _process_dates(df_processed)
    
    # Step 2: Calculate derived fields
    df_processed = _calculate_derived_fields(df_processed)
    
    # Step 3: Text quality improvements
    df_processed = _improve_text_quality(df_processed)
    
    # Step 4: Data quality filtering
    df_processed = _apply_quality_filters(df_processed)
    
    # Step 5: Add processing metadata
    df_processed = _add_processing_metadata(df_processed)
    
    logger.info(f"Pre-processing completed: {len(df)} → {len(df_processed)} claims")
    
    return df_processed


def _process_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and validate date columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with processed dates
    """
    date_columns = ['LOSSDT', 'REPORTEDDT', 'D_FINAL']
    
    for col in date_columns:
        if col in df.columns:
            # Convert to datetime with error handling
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Log conversion issues
            null_dates = df[col].isnull().sum()
            if null_dates > 0:
                logger.warning(f"Found {null_dates} invalid dates in column {col}")
    
    return df


def _calculate_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived fields and metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with calculated fields
    """
    # Calculate reporting lag (similar to Codebase 2)
    if 'REPORTEDDT' in df.columns and 'LOSSDT' in df.columns:
        df['rpt_lag'] = (df['REPORTEDDT'] - df['LOSSDT']).dt.days
        logger.debug("Calculated reporting lag")
    
    # Calculate claim age
    if 'LOSSDT' in df.columns:
        today = datetime.now()
        df['claim_age_days'] = (today - df['LOSSDT']).dt.days
        logger.debug("Calculated claim age")
    
    # Add claim complexity score based on text length and other factors
    if 'clean_FN_TEXT' in df.columns:
        df['text_length'] = df['clean_FN_TEXT'].str.len()
        
        # Complexity score based on multiple factors
        df['complexity_score'] = _calculate_complexity_score(df)
        logger.debug("Calculated complexity scores")
    
    return df


def _calculate_complexity_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate complexity score for claims.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Complexity scores
    """
    # Initialize complexity score
    complexity = pd.Series(1.0, index=df.index)
    
    # Text length factor
    if 'text_length' in df.columns:
        # Normalize text length (longer text = higher complexity)
        max_length = df['text_length'].max() if df['text_length'].max() > 0 else 1
        text_factor = df['text_length'] / max_length
        complexity += text_factor * 0.5
    
    # Reporting lag factor (older claims may be more complex)
    if 'rpt_lag' in df.columns:
        # Claims with unusual reporting patterns may be more complex
        unusual_lag = (df['rpt_lag'] < 0) | (df['rpt_lag'] > 365)
        complexity += unusual_lag.astype(float) * 0.3
    
    # Cap complexity score at reasonable maximum
    complexity = complexity.clip(0.1, 3.0)
    
    return complexity


def _improve_text_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Improve text quality in the clean_FN_TEXT column.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with improved text quality
    """
    if 'clean_FN_TEXT' not in df.columns:
        logger.warning("clean_FN_TEXT column not found, skipping text improvements")
        return df
    
    # Remove very short or empty text
    original_count = len(df)
    df = df[df['clean_FN_TEXT'].str.len() >= 10]
    removed_short = original_count - len(df)
    
    if removed_short > 0:
        logger.info(f"Removed {removed_short} claims with very short text")
    
    # Advanced text cleaning
    df['clean_FN_TEXT'] = df['clean_FN_TEXT'].apply(_clean_individual_text)
    
    # Remove duplicates based on claim number and similar text
    df = _remove_text_duplicates(df)
    
    return df


def _clean_individual_text(text: str) -> str:
    """
    Clean individual text entries.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    
    # Remove repeated characters (e.g., "aaaaaa" -> "aa")
    text = re.sub(r'(.)\\1{3,}', r'\\1\\1', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[^a-zA-Z0-9\\s.,;:!?()\\-$%]', ' ', text)
    
    # Normalize case for common insurance terms
    insurance_terms = {
        'bldg': 'building',
        'struc': 'structure',
        'dmg': 'damage',
        'rpr': 'repair',
        'est': 'estimate'
    }
    
    for abbrev, full_term in insurance_terms.items():
        text = re.sub(f'\\\\b{abbrev}\\\\b', full_term, text, flags=re.IGNORECASE)
    
    return text.strip()


def _remove_text_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove claims with very similar text content.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with duplicates removed
    """
    original_count = len(df)
    
    # Simple duplicate removal based on text similarity
    # This is a basic implementation - could be enhanced with more sophisticated methods
    df['text_hash'] = df['clean_FN_TEXT'].str[:100].str.lower()  # Simple hash
    df = df.drop_duplicates(subset=['CLAIMNO', 'text_hash'], keep='first')
    df = df.drop(columns=['text_hash'])
    
    removed_duplicates = original_count - len(df)
    if removed_duplicates > 0:
        logger.info(f"Removed {removed_duplicates} potential text duplicates")
    
    return df


def _apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data quality filters.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    original_count = len(df)
    
    # Filter 1: Valid reporting lag (similar to Codebase 2)
    if 'rpt_lag' in df.columns:
        valid_lag_mask = (df['rpt_lag'] >= 0) | (df['rpt_lag'].isnull())
        df = df[valid_lag_mask]
        
        negative_lag_removed = original_count - len(df)
        if negative_lag_removed > 0:
            logger.info(f"Removed {negative_lag_removed} claims with negative reporting lag")
    
    # Filter 2: Reasonable claim age
    if 'claim_age_days' in df.columns:
        reasonable_age_mask = (df['claim_age_days'] <= 3650) | (df['claim_age_days'].isnull())  # 10 years
        df = df[reasonable_age_mask]
        
        old_claims_removed = original_count - len(df)
        if old_claims_removed > 0:
            logger.info(f"Removed {old_claims_removed} very old claims")
    
    # Filter 3: Required fields present
    required_fields = ['CLAIMNO', 'clean_FN_TEXT']
    for field in required_fields:
        if field in df.columns:
            field_mask = df[field].notna() & (df[field] != '')
            df = df[field_mask]
    
    # Filter 4: Valid LOB codes (if present)
    if 'LOBCD' in df.columns:
        valid_lobs = ['15', '17', '18', '19']  # Expand as needed
        valid_lob_mask = df['LOBCD'].isin(valid_lobs) | df['LOBCD'].isnull()
        df = df[valid_lob_mask]
    
    final_count = len(df)
    total_removed = original_count - final_count
    
    if total_removed > 0:
        logger.info(f"Quality filtering removed {total_removed} claims total")
    
    return df


def _add_processing_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add processing metadata to the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with metadata
    """
    # Add processing timestamp
    df['preprocessing_timestamp'] = datetime.now()
    
    # Add processing confidence based on data quality
    df['processing_confidence'] = _calculate_processing_confidence(df)
    
    # Add data quality flags
    df['data_quality_flags'] = _generate_quality_flags(df)
    
    return df


def _calculate_processing_confidence(df: pd.DataFrame) -> pd.Series:
    """
    Calculate processing confidence based on data quality indicators.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Processing confidence scores (0.0 to 1.0)
    """
    confidence = pd.Series(1.0, index=df.index)  # Start with high confidence
    
    # Reduce confidence for claims with quality issues
    
    # Text length factor
    if 'text_length' in df.columns:
        short_text_mask = df['text_length'] < 100
        confidence.loc[short_text_mask] *= 0.8
        
        very_long_text_mask = df['text_length'] > 10000
        confidence.loc[very_long_text_mask] *= 0.9
    
    # Reporting lag factor
    if 'rpt_lag' in df.columns:
        unusual_lag_mask = (df['rpt_lag'] > 365) | (df['rpt_lag'] < 0)
        confidence.loc[unusual_lag_mask] *= 0.7
    
    # Missing data factor
    important_columns = ['LOBCD', 'LOSSDESC', 'REPORTEDDT']
    for col in important_columns:
        if col in df.columns:
            missing_mask = df[col].isnull()
            confidence.loc[missing_mask] *= 0.9
    
    # Ensure confidence is within valid range
    confidence = confidence.clip(0.1, 1.0)
    
    return confidence


def _generate_quality_flags(df: pd.DataFrame) -> pd.Series:
    """
    Generate data quality flags for each claim.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Quality flags as strings
    """
    flags = pd.Series('GOOD', index=df.index)
    
    # Flag claims with potential issues
    if 'processing_confidence' in df.columns:
        low_confidence_mask = df['processing_confidence'] < 0.7
        flags.loc[low_confidence_mask] = 'REVIEW'
        
        very_low_confidence_mask = df['processing_confidence'] < 0.5
        flags.loc[very_low_confidence_mask] = 'LOW_QUALITY'
    
    # Flag incomplete data
    if 'clean_FN_TEXT' in df.columns:
        short_text_mask = df['clean_FN_TEXT'].str.len() < 50
        flags.loc[short_text_mask] = 'INCOMPLETE'
    
    # Flag unusual claims
    if 'rpt_lag' in df.columns:
        unusual_lag_mask = df['rpt_lag'] < 0
        flags.loc[unusual_lag_mask] = 'UNUSUAL'
    
    return flags


# Additional utility functions for advanced pre-processing

def preprocess_with_config(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Pre-process data with custom configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    logger.info(f"Pre-processing with custom config for {len(df)} claims")
    
    # Apply standard pre-processing
    df_processed = pre_process(df)
    
    # Apply additional config-based processing
    if config.get('enable_advanced_filtering', False):
        df_processed = _apply_advanced_filters(df_processed, config)
    
    if config.get('enable_feature_engineering', False):
        df_processed = _add_advanced_features(df_processed, config)
    
    return df_processed


def _apply_advanced_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply advanced filtering based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    # Custom confidence threshold
    confidence_threshold = config.get('confidence_threshold', 0.5)
    if 'processing_confidence' in df.columns:
        df = df[df['processing_confidence'] >= confidence_threshold]
        logger.info(f"Applied confidence threshold filter: {confidence_threshold}")
    
    # Custom complexity filtering
    max_complexity = config.get('max_complexity', 3.0)
    if 'complexity_score' in df.columns:
        df = df[df['complexity_score'] <= max_complexity]
        logger.info(f"Applied complexity filter: max {max_complexity}")
    
    return df


def _add_advanced_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add advanced features based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Dataframe with additional features
    """
    # Add seasonal indicators
    if config.get('add_seasonal_features', False) and 'LOSSDT' in df.columns:
        df['loss_month'] = df['LOSSDT'].dt.month
        df['loss_quarter'] = df['LOSSDT'].dt.quarter
        df['loss_season'] = df['loss_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        logger.debug("Added seasonal features")
    
    # Add text complexity metrics
    if config.get('add_text_metrics', False) and 'clean_FN_TEXT' in df.columns:
        df['word_count'] = df['clean_FN_TEXT'].str.split().str.len()
        df['sentence_count'] = df['clean_FN_TEXT'].str.count('\\.') + 1
        df['avg_word_length'] = df['clean_FN_TEXT'].str.replace(' ', '').str.len() / df['word_count']
        logger.debug("Added text complexity metrics")
    
    return df