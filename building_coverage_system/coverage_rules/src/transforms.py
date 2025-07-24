"""
Data transformations for building coverage system.

This module provides data transformation functions used by the
original Codebase 1 rules engine for formatting and preparing data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def select_and_rename_bldg_predictions_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename columns for building predictions database output.
    
    This function formats the prediction results for database storage,
    maintaining compatibility with the original Codebase 1 output format.
    
    Args:
        df (pd.DataFrame): Input dataframe with predictions and rules applied
        
    Returns:
        pd.DataFrame: Formatted dataframe ready for database insertion
    """
    logger.info(f"Transforming {len(df)} predictions for database output")
    
    # Define column mapping from internal names to database schema
    column_mapping = {
        'CLAIMNO': 'CLAIM_NUMBER',
        'CLAIMKEY': 'CLAIM_KEY',
        'clean_FN_TEXT': 'CLAIM_DESCRIPTION',
        'LOBCD': 'LINE_OF_BUSINESS_CODE',
        'LOSSDESC': 'LOSS_DESCRIPTION',
        'LOSSDT': 'LOSS_DATE',
        'REPORTEDDT': 'REPORTED_DATE',
        'prediction': 'BUILDING_COVERAGE_PREDICTION',
        'confidence': 'PREDICTION_CONFIDENCE',
        'summary': 'PREDICTION_SUMMARY',
        'rule_override': 'RULE_OVERRIDE_APPLIED',
        'requires_manual_review': 'REQUIRES_MANUAL_REVIEW',
        'review_reason': 'MANUAL_REVIEW_REASON',
        'building_keyword_score': 'BUILDING_KEYWORD_SCORE',
        'processing_time': 'PROCESSING_TIME_SECONDS',
        'model_version': 'MODEL_VERSION',
        'rules_version': 'RULES_VERSION'
    }
    
    # Select available columns and create output dataframe
    output_df = pd.DataFrame()
    
    for internal_col, db_col in column_mapping.items():
        if internal_col in df.columns:
            output_df[db_col] = df[internal_col]
        else:
            # Provide default values for missing columns
            output_df[db_col] = _get_default_value(internal_col, len(df))
    
    # Add standard database columns
    output_df = _add_standard_db_columns(output_df)
    
    # Apply data type formatting
    output_df = _format_data_types(output_df)
    
    # Apply business formatting rules
    output_df = _apply_business_formatting(output_df)
    
    logger.info(f"Database transformation completed: {len(output_df)} records formatted")
    
    return output_df


def _get_default_value(column_name: str, row_count: int) -> Union[pd.Series, Any]:
    """
    Get default value for missing columns.
    
    Args:
        column_name (str): Name of the missing column
        row_count (int): Number of rows needed
        
    Returns:
        Union[pd.Series, Any]: Default values
    """
    defaults = {
        'CLAIMNO': pd.Series(['UNKNOWN'] * row_count),
        'CLAIMKEY': pd.Series(['UNKNOWN'] * row_count),
        'clean_FN_TEXT': pd.Series([''] * row_count),
        'LOBCD': pd.Series(['UNKNOWN'] * row_count),
        'LOSSDESC': pd.Series([''] * row_count),
        'LOSSDT': pd.Series([None] * row_count),
        'REPORTEDDT': pd.Series([None] * row_count),
        'prediction': pd.Series(['UNCLEAR'] * row_count),
        'confidence': pd.Series([0.5] * row_count),
        'summary': pd.Series([''] * row_count),
        'rule_override': pd.Series([None] * row_count),
        'requires_manual_review': pd.Series([False] * row_count),
        'review_reason': pd.Series([None] * row_count),
        'building_keyword_score': pd.Series([0.0] * row_count),
        'processing_time': pd.Series([0.0] * row_count),
        'model_version': pd.Series(['1.0'] * row_count),
        'rules_version': pd.Series(['1.0'] * row_count)
    }
    
    return defaults.get(column_name, pd.Series([None] * row_count))


def _add_standard_db_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard database columns required for all records.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with standard columns added
    """
    # Add processing metadata
    df['PROCESSING_TIMESTAMP'] = datetime.now()
    df['CREATED_DATE'] = datetime.now()
    df['LAST_MODIFIED_DATE'] = datetime.now()
    df['PROCESSING_STATUS'] = 'COMPLETED'
    df['DATA_SOURCE'] = 'BUILDING_COVERAGE_PIPELINE'
    df['BATCH_ID'] = _generate_batch_id()
    df['RECORD_VERSION'] = 1
    df['IS_ACTIVE'] = True
    
    return df


def _generate_batch_id() -> str:
    """
    Generate a unique batch ID for this processing run.
    
    Returns:
        str: Unique batch identifier
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"BLDG_COVERAGE_{timestamp}"


def _format_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format data types for database compatibility.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with formatted data types
    """
    # String columns - ensure proper length limits
    string_columns = {
        'CLAIM_NUMBER': 50,
        'CLAIM_KEY': 50,
        'LINE_OF_BUSINESS_CODE': 10,
        'BUILDING_COVERAGE_PREDICTION': 50,
        'PREDICTION_SUMMARY': 2000,
        'RULE_OVERRIDE_APPLIED': 100,
        'MANUAL_REVIEW_REASON': 200,
        'MODEL_VERSION': 20,
        'RULES_VERSION': 20,
        'PROCESSING_STATUS': 20,
        'DATA_SOURCE': 50,
        'BATCH_ID': 50
    }
    
    for col, max_length in string_columns.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str[:max_length]
    
    # Numeric columns
    numeric_columns = {
        'PREDICTION_CONFIDENCE': 'float64',
        'BUILDING_KEYWORD_SCORE': 'float64',
        'PROCESSING_TIME_SECONDS': 'float64',
        'RECORD_VERSION': 'int64'
    }
    
    for col, dtype in numeric_columns.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)
    
    # Boolean columns
    boolean_columns = ['REQUIRES_MANUAL_REVIEW', 'IS_ACTIVE']
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # Date columns
    date_columns = ['LOSS_DATE', 'REPORTED_DATE', 'PROCESSING_TIMESTAMP', 
                   'CREATED_DATE', 'LAST_MODIFIED_DATE']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def _apply_business_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business-specific formatting rules.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with business formatting applied
    """
    # Format prediction values to standard format
    if 'BUILDING_COVERAGE_PREDICTION' in df.columns:
        df['BUILDING_COVERAGE_PREDICTION'] = df['BUILDING_COVERAGE_PREDICTION'].map({
            'BUILDING COVERAGE': 'BUILDING_COVERAGE',
            'NO BUILDING COVERAGE': 'NO_BUILDING_COVERAGE',
            'UNCLEAR': 'UNCLEAR',
            'ERROR': 'ERROR'
        }).fillna('UNCLEAR')
    
    # Ensure confidence is in valid range
    if 'PREDICTION_CONFIDENCE' in df.columns:
        df['PREDICTION_CONFIDENCE'] = df['PREDICTION_CONFIDENCE'].clip(0.0, 1.0)
        df['PREDICTION_CONFIDENCE'] = df['PREDICTION_CONFIDENCE'].round(4)
    
    # Format LOB codes
    if 'LINE_OF_BUSINESS_CODE' in df.columns:
        df['LINE_OF_BUSINESS_CODE'] = df['LINE_OF_BUSINESS_CODE'].str.upper().str.strip()
    
    # Clean text fields
    text_fields = ['CLAIM_DESCRIPTION', 'LOSS_DESCRIPTION', 'PREDICTION_SUMMARY']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()
            df[field] = df[field].replace('nan', '')
    
    return df


def transform_for_analysis_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform data for analysis and reporting output.
    
    Args:
        df (pd.DataFrame): Input dataframe with predictions
        
    Returns:
        pd.DataFrame: Transformed dataframe for analysis
    """
    logger.info(f"Transforming {len(df)} records for analysis output")
    
    # Create analysis-focused columns
    analysis_df = df.copy()
    
    # Add derived metrics
    analysis_df = _add_analysis_metrics(analysis_df)
    
    # Create summary categories
    analysis_df = _create_summary_categories(analysis_df)
    
    # Add quality indicators
    analysis_df = _add_quality_indicators(analysis_df)
    
    # Format for readability
    analysis_df = _format_for_readability(analysis_df)
    
    return analysis_df


def _add_analysis_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add metrics useful for analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with analysis metrics
    """
    # Confidence categories
    if 'confidence' in df.columns:
        df['confidence_category'] = pd.cut(
            df['confidence'],
            bins=[0, 0.5, 0.7, 0.85, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
    
    # Text length categories
    if 'clean_FN_TEXT' in df.columns:
        text_lengths = df['clean_FN_TEXT'].str.len()
        df['text_length_category'] = pd.cut(
            text_lengths,
            bins=[0, 100, 300, 1000, float('inf')],
            labels=['Very Short', 'Short', 'Medium', 'Long'],
            include_lowest=True
        )
    
    # Processing time categories
    if 'processing_time' in df.columns:
        df['processing_speed'] = pd.cut(
            df['processing_time'],
            bins=[0, 1, 5, 15, float('inf')],
            labels=['Fast', 'Normal', 'Slow', 'Very Slow'],
            include_lowest=True
        )
    
    # Keyword strength categories
    if 'building_keyword_score' in df.columns:
        df['keyword_strength'] = pd.cut(
            df['building_keyword_score'],
            bins=[-float('inf'), 0, 2, 5, float('inf')],
            labels=['None', 'Weak', 'Moderate', 'Strong'],
            include_lowest=True
        )
    
    return df


def _create_summary_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create high-level summary categories.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with summary categories
    """
    # Overall prediction quality
    df['prediction_quality'] = 'Unknown'
    
    if 'confidence' in df.columns and 'prediction' in df.columns:
        conditions = [
            (df['confidence'] >= 0.8) & (df['prediction'].isin(['BUILDING COVERAGE', 'NO BUILDING COVERAGE'])),
            (df['confidence'] >= 0.6) & (df['prediction'].isin(['BUILDING COVERAGE', 'NO BUILDING COVERAGE'])),
            (df['confidence'] >= 0.4) & (df['prediction'].isin(['BUILDING COVERAGE', 'NO BUILDING COVERAGE'])),
            df['prediction'] == 'UNCLEAR'
        ]
        
        choices = ['High Quality', 'Good Quality', 'Fair Quality', 'Unclear']
        
        df['prediction_quality'] = np.select(conditions, choices, default='Poor Quality')
    
    # Processing complexity
    df['processing_complexity'] = 'Standard'
    
    if 'rule_override' in df.columns:
        df.loc[df['rule_override'].notna(), 'processing_complexity'] = 'Rule Override Applied'
    
    if 'requires_manual_review' in df.columns:
        df.loc[df['requires_manual_review'] == True, 'processing_complexity'] = 'Manual Review Required'
    
    return df


def _add_quality_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add data quality indicators.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with quality indicators
    """
    # Data completeness score
    important_fields = ['CLAIMNO', 'clean_FN_TEXT', 'LOBCD', 'LOSSDT']
    available_fields = [f for f in important_fields if f in df.columns]
    
    if available_fields:
        completeness_scores = []
        for _, row in df.iterrows():
            complete_fields = sum(1 for field in available_fields if pd.notna(row.get(field)))
            score = complete_fields / len(available_fields)
            completeness_scores.append(score)
        
        df['data_completeness_score'] = completeness_scores
        
        # Completeness categories
        df['data_completeness'] = pd.cut(
            df['data_completeness_score'],
            bins=[0, 0.5, 0.8, 1.0],
            labels=['Poor', 'Fair', 'Good'],
            include_lowest=True
        )
    
    # Processing flags summary
    flag_columns = ['requires_manual_review', 'high_confidence_flag', 'rule_override']
    available_flags = [f for f in flag_columns if f in df.columns]
    
    if available_flags:
        df['has_processing_flags'] = df[available_flags].notna().any(axis=1)
    
    return df


def _format_for_readability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format columns for better readability in reports.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe formatted for readability
    """
    # Format confidence as percentage
    if 'confidence' in df.columns:
        df['confidence_pct'] = (df['confidence'] * 100).round(1).astype(str) + '%'
    
    # Format processing time
    if 'processing_time' in df.columns:
        df['processing_time_formatted'] = df['processing_time'].round(2).astype(str) + 's'
    
    # Clean prediction labels
    if 'prediction' in df.columns:
        df['prediction_label'] = df['prediction'].str.replace('_', ' ').str.title()
    
    # Format dates
    date_columns = ['LOSSDT', 'REPORTEDDT']
    for col in date_columns:
        if col in df.columns:
            df[f'{col}_formatted'] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics for the processed data.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        Dict[str, Any]: Summary statistics
    """
    stats = {
        'total_claims': len(df),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    # Prediction distribution
    if 'prediction' in df.columns:
        prediction_counts = df['prediction'].value_counts().to_dict()
        stats['prediction_distribution'] = prediction_counts
        
        building_coverage_rate = prediction_counts.get('BUILDING COVERAGE', 0) / len(df)
        stats['building_coverage_rate'] = round(building_coverage_rate * 100, 1)
    
    # Confidence statistics
    if 'confidence' in df.columns:
        confidence_stats = df['confidence'].describe()
        stats['confidence_statistics'] = {
            'mean': round(confidence_stats['mean'], 3),
            'median': round(confidence_stats['50%'], 3),
            'std': round(confidence_stats['std'], 3),
            'min': round(confidence_stats['min'], 3),
            'max': round(confidence_stats['max'], 3)
        }
    
    # Quality indicators
    if 'prediction_quality' in df.columns:
        quality_dist = df['prediction_quality'].value_counts().to_dict()
        stats['quality_distribution'] = quality_dist
    
    # Processing indicators
    if 'requires_manual_review' in df.columns:
        manual_review_count = df['requires_manual_review'].sum()
        stats['manual_review_required'] = int(manual_review_count)
        stats['manual_review_rate'] = round(manual_review_count / len(df) * 100, 1)
    
    if 'rule_override' in df.columns:
        override_count = df['rule_override'].notna().sum()
        stats['rule_overrides_applied'] = int(override_count)
        stats['rule_override_rate'] = round(override_count / len(df) * 100, 1)
    
    # Performance metrics
    if 'processing_time' in df.columns:
        time_stats = df['processing_time'].describe()
        stats['processing_time_statistics'] = {
            'mean_seconds': round(time_stats['mean'], 2),
            'median_seconds': round(time_stats['50%'], 2),
            'total_seconds': round(df['processing_time'].sum(), 2)
        }
    
    return stats


def validate_output_format(df: pd.DataFrame, output_type: str = 'database') -> Dict[str, Any]:
    """
    Validate that the output dataframe meets format requirements.
    
    Args:
        df (pd.DataFrame): Output dataframe to validate
        output_type (str): Type of output ('database', 'analysis', 'report')
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'row_count': len(df),
        'column_count': len(df.columns)
    }
    
    if output_type == 'database':
        required_columns = [
            'CLAIM_NUMBER', 'BUILDING_COVERAGE_PREDICTION',
            'PREDICTION_CONFIDENCE', 'PROCESSING_TIMESTAMP'
        ]
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            validation_result['is_valid'] = False
        
        # Check data types
        if 'PREDICTION_CONFIDENCE' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['PREDICTION_CONFIDENCE']):
                validation_result['errors'].append("PREDICTION_CONFIDENCE must be numeric")
                validation_result['is_valid'] = False
        
        # Check value ranges
        if 'PREDICTION_CONFIDENCE' in df.columns:
            out_of_range = (df['PREDICTION_CONFIDENCE'] < 0) | (df['PREDICTION_CONFIDENCE'] > 1)
            if out_of_range.any():
                validation_result['warnings'].append(f"{out_of_range.sum()} confidence values out of range [0,1]")
    
    # Check for completely empty dataframe
    if len(df) == 0:
        validation_result['warnings'].append("Output dataframe is empty")
    
    # Check for null values in critical columns
    critical_columns = ['CLAIM_NUMBER', 'BUILDING_COVERAGE_PREDICTION']
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                validation_result['warnings'].append(f"{null_count} null values in {col}")
    
    return validation_result