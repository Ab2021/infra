"""
Post-processing hooks for building coverage predictions.

This module provides post-processing functions that can be used as hooks
in the building coverage pipeline, similar to Codebase 2's post_process hook.
These functions are called after RAG processing and rule application to format
output and apply business logic.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Set up logging for the hook
logger = logging.getLogger(__name__)


def post_process(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main post-processing hook for building coverage predictions.
    
    This function performs comprehensive post-processing of the prediction results
    similar to Codebase 2's post_process function. It includes business logic
    application, output formatting, and quality validation.
    
    Args:
        predictions_df (pd.DataFrame): Input dataframe containing RAG predictions and rules
        
    Returns:
        pd.DataFrame: Final processed dataframe ready for output
    """
    logger.info(f"Starting post-processing for {len(predictions_df)} predictions")
    
    # Make a copy to avoid modifying the original data
    df_processed = predictions_df.copy()
    
    # Step 1: Apply business logic transformations
    df_processed = _apply_business_logic(df_processed)
    
    # Step 2: Create enhanced summaries
    df_processed = _create_enhanced_summaries(df_processed)
    
    # Step 3: Calculate confidence adjustments
    df_processed = _adjust_confidence_scores(df_processed)
    
    # Step 4: Apply quality flags and validation
    df_processed = _apply_quality_validation(df_processed)
    
    # Step 5: Format output fields
    df_processed = _format_output_fields(df_processed)
    
    # Step 6: Add final metadata
    df_processed = _add_final_metadata(df_processed)
    
    logger.info(f"Post-processing completed for {len(df_processed)} predictions")
    
    return df_processed


def _apply_business_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business logic transformations to predictions.
    
    Args:
        df (pd.DataFrame): Input dataframe with predictions
        
    Returns:
        pd.DataFrame: Dataframe with business logic applied
    """
    # Extract structured information from summaries using regex
    if 'summary' in df.columns:
        df = _extract_structured_fields(df)
    
    # Apply business rules for prediction refinement
    df = _apply_prediction_refinements(df)
    
    # Handle special cases
    df = _handle_special_cases(df)
    
    return df


def _extract_structured_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured fields from summary text using regex patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with extracted fields
    """
    summary_col = 'summary'
    
    # Define extraction patterns (similar to Codebase 2's approach)
    extraction_patterns = {
        'damage_type': r'damage[\\s_]type[\\s]*:[\\s]*["\']([^"\']+)["\']',
        'structure_type': r'structure[\\s_]type[\\s]*:[\\s]*["\']([^"\']+)["\']',
        'cause_of_loss': r'cause[\\s_]of[\\s_]loss[\\s]*:[\\s]*["\']([^"\']+)["\']',
        'estimated_cost': r'estimated[\\s_]cost[\\s]*:[\\s]*\\$?([\\d,]+(?:\\.\\d{2})?)',
        'building_material': r'building[\\s_]material[\\s]*:[\\s]*["\']([^"\']+)["\']',
        'coverage_determination': r'coverage[\\s]*:[\\s]*["\']([^"\']+)["\']'
    }
    
    # Extract fields
    for field_name, pattern in extraction_patterns.items():
        df[field_name] = df[summary_col].str.extract(
            pattern, flags=re.IGNORECASE
        )[0]
        
        # Clean up extracted values
        df[field_name] = df[field_name].str.strip()
        df[field_name] = df[field_name].replace('', np.nan)
    
    # Log extraction statistics
    for field_name in extraction_patterns.keys():
        extracted_count = df[field_name].notna().sum()
        logger.debug(f"Extracted {field_name} for {extracted_count} claims")
    
    return df


def _apply_prediction_refinements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business rules to refine predictions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with refined predictions
    """
    if 'prediction' not in df.columns:
        logger.warning("No prediction column found for refinement")
        return df
    
    # Refinement 1: Adjust predictions based on extracted damage types
    if 'damage_type' in df.columns:
        building_damage_keywords = ['structural', 'foundation', 'roof', 'wall', 'building']
        
        # Increase building coverage likelihood for structural damage
        structural_mask = df['damage_type'].str.contains(
            '|'.join(building_damage_keywords), case=False, na=False
        )
        df.loc[structural_mask, 'prediction_adjusted'] = 'BUILDING COVERAGE'
        df.loc[~structural_mask, 'prediction_adjusted'] = df.loc[~structural_mask, 'prediction']
    else:
        df['prediction_adjusted'] = df['prediction']
    
    # Refinement 2: Handle uncertainty indicators
    uncertainty_indicators = ['uncertain', 'maybe', 'possible', 'unclear', 'ambiguous']
    uncertainty_pattern = '|'.join(uncertainty_indicators)
    
    if 'summary' in df.columns:
        uncertain_mask = df['summary'].str.contains(
            uncertainty_pattern, case=False, na=False
        )
        df.loc[uncertain_mask, 'prediction_certainty'] = 'LOW'
        df.loc[~uncertain_mask, 'prediction_certainty'] = 'HIGH'
    else:
        df['prediction_certainty'] = 'MEDIUM'
    
    return df


def _handle_special_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle special cases and edge conditions.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with special cases handled
    """
    # Special Case 1: Claims with very high confidence but conflicting information
    if 'confidence' in df.columns and 'prediction_certainty' in df.columns:
        conflict_mask = (df['confidence'] > 0.9) & (df['prediction_certainty'] == 'LOW')
        df.loc[conflict_mask, 'special_review_flag'] = 'HIGH_CONFIDENCE_LOW_CERTAINTY'
        
        logger.info(f"Flagged {conflict_mask.sum()} claims for special review")
    
    # Special Case 2: Claims with missing critical information
    critical_fields = ['CLAIMNO', 'prediction']
    for field in critical_fields:
        if field in df.columns:
            missing_mask = df[field].isnull()
            df.loc[missing_mask, 'special_review_flag'] = f'MISSING_{field.upper()}'
    
    # Special Case 3: Very old claims that might need different handling
    if 'claim_age_days' in df.columns:
        old_claims_mask = df['claim_age_days'] > 1095  # 3 years
        df.loc[old_claims_mask, 'special_review_flag'] = 'OLD_CLAIM'
    
    return df


def _create_enhanced_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create enhanced claim summaries combining multiple data sources.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with enhanced summaries
    """
    summary_components = []
    
    # Component 1: Basic claim information
    if 'CLAIMNO' in df.columns:
        summary_components.append(
            df['CLAIMNO'].apply(lambda x: f"Claim {x}" if pd.notna(x) else "")
        )
    
    # Component 2: Loss description
    if 'LOSSDESC' in df.columns:
        loss_desc_clean = df['LOSSDESC'].fillna('').str[:100]  # Limit length
        summary_components.append(
            loss_desc_clean.apply(lambda x: f"Loss: {x}" if x else "")
        )
    
    # Component 3: Prediction information
    if 'prediction_adjusted' in df.columns:
        summary_components.append(
            df['prediction_adjusted'].apply(lambda x: f"Coverage: {x}" if pd.notna(x) else "")
        )
    elif 'prediction' in df.columns:
        summary_components.append(
            df['prediction'].apply(lambda x: f"Coverage: {x}" if pd.notna(x) else "")
        )
    
    # Component 4: Confidence information
    if 'confidence' in df.columns:
        confidence_desc = df['confidence'].apply(
            lambda x: f"Confidence: {x:.1%}" if pd.notna(x) else ""
        )
        summary_components.append(confidence_desc)
    
    # Component 5: Key extracted information
    if 'damage_type' in df.columns:
        damage_info = df['damage_type'].apply(
            lambda x: f"Damage: {x}" if pd.notna(x) else ""
        )
        summary_components.append(damage_info)
    
    # Combine all components
    if summary_components:
        # Filter out empty components and join with separators
        df['enhanced_summary'] = pd.Series(summary_components).T.apply(
            lambda row: " | ".join([comp for comp in row if comp]), axis=1
        )
    else:
        df['enhanced_summary'] = "No summary available"
    
    logger.debug("Created enhanced summaries")
    
    return df


def _adjust_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust confidence scores based on post-processing analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with adjusted confidence scores
    """
    if 'confidence' not in df.columns:
        logger.warning("No confidence column found for adjustment")
        return df
    
    # Start with original confidence
    df['confidence_adjusted'] = df['confidence'].copy()
    
    # Adjustment 1: Reduce confidence for uncertain predictions
    if 'prediction_certainty' in df.columns:
        low_certainty_mask = df['prediction_certainty'] == 'LOW'
        df.loc[low_certainty_mask, 'confidence_adjusted'] *= 0.8
        
        logger.debug(f"Reduced confidence for {low_certainty_mask.sum()} uncertain predictions")
    
    # Adjustment 2: Reduce confidence for incomplete data
    if 'data_quality_flags' in df.columns:
        poor_quality_mask = df['data_quality_flags'].isin(['INCOMPLETE', 'LOW_QUALITY'])
        df.loc[poor_quality_mask, 'confidence_adjusted'] *= 0.7
        
        logger.debug(f"Reduced confidence for {poor_quality_mask.sum()} poor quality claims")
    
    # Adjustment 3: Increase confidence for consistent structured data
    structured_fields = ['damage_type', 'structure_type', 'cause_of_loss']
    if all(field in df.columns for field in structured_fields):
        complete_structure_mask = df[structured_fields].notna().all(axis=1)
        df.loc[complete_structure_mask, 'confidence_adjusted'] *= 1.1
        
        logger.debug(f"Increased confidence for {complete_structure_mask.sum()} well-structured claims")
    
    # Ensure confidence stays within valid bounds
    df['confidence_adjusted'] = df['confidence_adjusted'].clip(0.0, 1.0)
    
    return df


def _apply_quality_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality validation and create quality flags.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with quality flags
    """
    # Initialize quality flags
    df['final_quality_flag'] = 'GOOD'
    
    # Flag 1: Low confidence predictions
    if 'confidence_adjusted' in df.columns:
        low_conf_mask = df['confidence_adjusted'] < 0.5
        df.loc[low_conf_mask, 'final_quality_flag'] = 'LOW_CONFIDENCE'
        
        review_conf_mask = (df['confidence_adjusted'] >= 0.5) & (df['confidence_adjusted'] < 0.7)
        df.loc[review_conf_mask & (df['final_quality_flag'] == 'GOOD'), 'final_quality_flag'] = 'REVIEW'
    
    # Flag 2: Incomplete summaries
    if 'enhanced_summary' in df.columns:
        short_summary_mask = df['enhanced_summary'].str.len() < 50
        df.loc[short_summary_mask & (df['final_quality_flag'] == 'GOOD'), 'final_quality_flag'] = 'INCOMPLETE_SUMMARY'
    
    # Flag 3: Special review cases
    if 'special_review_flag' in df.columns:
        special_review_mask = df['special_review_flag'].notna()
        df.loc[special_review_mask, 'final_quality_flag'] = 'SPECIAL_REVIEW'
    
    # Flag 4: Missing critical predictions
    if 'prediction_adjusted' in df.columns:
        missing_pred_mask = df['prediction_adjusted'].isnull()
        df.loc[missing_pred_mask, 'final_quality_flag'] = 'NO_PREDICTION'
    
    # Log quality distribution
    quality_counts = df['final_quality_flag'].value_counts()
    logger.info("Final quality distribution:")
    for quality, count in quality_counts.items():
        logger.info(f"  {quality}: {count}")
    
    return df


def _format_output_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format output fields for final consumption.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with formatted output fields
    """
    # Format prediction field
    if 'prediction_adjusted' in df.columns:
        df['final_prediction'] = df['prediction_adjusted'].fillna('NO_PREDICTION')
    elif 'prediction' in df.columns:
        df['final_prediction'] = df['prediction'].fillna('NO_PREDICTION')
    
    # Format confidence field
    if 'confidence_adjusted' in df.columns:
        df['final_confidence'] = df['confidence_adjusted'].round(3)
    elif 'confidence' in df.columns:
        df['final_confidence'] = df['confidence'].round(3)
    else:
        df['final_confidence'] = 0.0
    
    # Create recommendation field
    df['recommendation'] = df.apply(_create_recommendation, axis=1)
    
    # Format currency fields
    currency_fields = ['estimated_cost']
    for field in currency_fields:
        if field in df.columns:
            df[f'{field}_formatted'] = df[field].apply(_format_currency)
    
    return df


def _create_recommendation(row) -> str:
    """
    Create recommendation based on prediction and confidence.
    
    Args:
        row: DataFrame row
        
    Returns:
        str: Recommendation text
    """
    prediction = row.get('final_prediction', 'NO_PREDICTION')
    confidence = row.get('final_confidence', 0.0)
    quality = row.get('final_quality_flag', 'UNKNOWN')
    
    if quality in ['NO_PREDICTION', 'LOW_CONFIDENCE']:
        return "MANUAL_REVIEW_REQUIRED"
    elif quality == 'SPECIAL_REVIEW':
        return "SPECIAL_REVIEW_REQUIRED"
    elif confidence >= 0.8:
        return f"AUTO_PROCESS_{prediction.replace(' ', '_')}"
    elif confidence >= 0.6:
        return f"REVIEW_RECOMMENDED_{prediction.replace(' ', '_')}"
    else:
        return "MANUAL_REVIEW_REQUIRED"


def _format_currency(value) -> str:
    """
    Format currency values.
    
    Args:
        value: Currency value to format
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value):
        return "N/A"
    
    try:
        # Convert to float and format
        amount = float(str(value).replace(',', '').replace('$', ''))
        return f"${amount:,.2f}"
    except (ValueError, TypeError):
        return str(value)


def _add_final_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add final processing metadata.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with final metadata
    """
    # Add processing timestamp
    df['postprocessing_timestamp'] = datetime.now()
    
    # Add processing version
    df['processing_version'] = '1.0.0'
    
    # Add record hash for tracking
    df['record_hash'] = df.apply(_calculate_record_hash, axis=1)
    
    # Add final validation status
    df['validation_status'] = df.apply(_get_validation_status, axis=1)
    
    return df


def _calculate_record_hash(row) -> str:
    """
    Calculate a hash for the record for tracking purposes.
    
    Args:
        row: DataFrame row
        
    Returns:
        str: Record hash
    """
    import hashlib
    
    # Create hash from key fields
    key_fields = ['CLAIMNO', 'final_prediction', 'final_confidence']
    hash_input = ''
    
    for field in key_fields:
        if field in row:
            hash_input += str(row[field])
    
    # Calculate MD5 hash
    hash_object = hashlib.md5(hash_input.encode())
    return hash_object.hexdigest()[:8]  # Short hash


def _get_validation_status(row) -> str:
    """
    Get overall validation status for the record.
    
    Args:
        row: DataFrame row
        
    Returns:
        str: Validation status
    """
    quality = row.get('final_quality_flag', 'UNKNOWN')
    confidence = row.get('final_confidence', 0.0)
    
    if quality == 'GOOD' and confidence >= 0.8:
        return 'VALIDATED'
    elif quality in ['REVIEW', 'INCOMPLETE_SUMMARY'] and confidence >= 0.6:
        return 'NEEDS_REVIEW'
    elif quality in ['SPECIAL_REVIEW']:
        return 'SPECIAL_CASE'
    else:
        return 'VALIDATION_FAILED'


# Additional utility functions for custom post-processing

def post_process_with_config(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Post-process data with custom configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Processed dataframe
    """
    logger.info(f"Post-processing with custom config for {len(df)} predictions")
    
    # Apply standard post-processing
    df_processed = post_process(df)
    
    # Apply additional config-based processing
    if config.get('enable_advanced_validation', False):
        df_processed = _apply_advanced_validation(df_processed, config)
    
    if config.get('enable_custom_formatting', False):
        df_processed = _apply_custom_formatting(df_processed, config)
    
    return df_processed


def _apply_advanced_validation(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply advanced validation rules based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Validated dataframe
    """
    # Custom confidence thresholds
    high_confidence_threshold = config.get('high_confidence_threshold', 0.8)
    low_confidence_threshold = config.get('low_confidence_threshold', 0.5)
    
    if 'final_confidence' in df.columns:
        # Reclassify based on custom thresholds
        high_conf_mask = df['final_confidence'] >= high_confidence_threshold
        low_conf_mask = df['final_confidence'] < low_confidence_threshold
        
        df.loc[high_conf_mask, 'confidence_category'] = 'HIGH'
        df.loc[~high_conf_mask & ~low_conf_mask, 'confidence_category'] = 'MEDIUM'
        df.loc[low_conf_mask, 'confidence_category'] = 'LOW'
        
        logger.info(f"Applied custom confidence thresholds: {high_confidence_threshold}, {low_confidence_threshold}")
    
    return df


def _apply_custom_formatting(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply custom formatting based on configuration.
    
    Args:
        df (pd.DataFrame): Input dataframe
        config (Dict[str, Any]): Configuration parameters
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    # Custom date formatting
    date_format = config.get('date_format', '%Y-%m-%d %H:%M:%S')
    timestamp_columns = ['postprocessing_timestamp', 'preprocessing_timestamp']
    
    for col in timestamp_columns:
        if col in df.columns:
            df[f'{col}_formatted'] = df[col].dt.strftime(date_format)
    
    # Custom prediction formatting
    prediction_mapping = config.get('prediction_mapping', {})
    if prediction_mapping and 'final_prediction' in df.columns:
        df['final_prediction_mapped'] = df['final_prediction'].map(prediction_mapping).fillna(df['final_prediction'])
        logger.info("Applied custom prediction mapping")
    
    return df


def create_summary_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a summary report of post-processing results.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        
    Returns:
        Dict[str, Any]: Summary report
    """
    report = {
        'total_claims': len(df),
        'processing_timestamp': datetime.now().isoformat(),
        'quality_distribution': {},
        'confidence_statistics': {},
        'recommendation_distribution': {},
        'validation_summary': {}
    }
    
    # Quality distribution
    if 'final_quality_flag' in df.columns:
        report['quality_distribution'] = df['final_quality_flag'].value_counts().to_dict()
    
    # Confidence statistics
    if 'final_confidence' in df.columns:
        conf_stats = df['final_confidence'].describe()
        report['confidence_statistics'] = {
            'mean': conf_stats['mean'],
            'median': conf_stats['50%'],
            'std': conf_stats['std'],
            'min': conf_stats['min'],
            'max': conf_stats['max']
        }
    
    # Recommendation distribution
    if 'recommendation' in df.columns:
        report['recommendation_distribution'] = df['recommendation'].value_counts().to_dict()
    
    # Validation summary
    if 'validation_status' in df.columns:
        report['validation_summary'] = df['validation_status'].value_counts().to_dict()
    
    return report