"""
Data validation for the building coverage pipeline.

This module provides the PipelineValidator class for validating input and output
data throughout the pipeline execution, ensuring data quality and consistency.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime, timedelta


class PipelineValidator:
    """
    Pipeline data validation and error handling.
    
    This class provides comprehensive data validation capabilities for the
    building coverage pipeline, including input validation, output validation,
    and data quality checks.
    
    Attributes:
        logger: Logging instance for validation operations
        validation_config (Dict[str, Any]): Configuration for validation rules
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                 validation_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline validator.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance for validation operations
            validation_config (Optional[Dict[str, Any]]): Configuration for validation rules
        """
        self.logger = logger if logger else logging.getLogger(__name__)
        self.validation_config = validation_config or self._get_default_validation_config()
    
    def _get_default_validation_config(self) -> Dict[str, Any]:
        """
        Get default validation configuration.
        
        Returns:
            Dict[str, Any]: Default validation configuration
        """
        return {
            'required_columns': {
                'input': ['CLAIMNO', 'CLAIMKEY', 'clean_FN_TEXT', 'LOBCD'],
                'output': ['CLAIMNO', 'prediction', 'confidence']
            },
            'text_validation': {
                'min_length': 10,
                'max_length': 50000,
                'required_encoding': 'utf-8'
            },
            'claim_validation': {
                'claimno_pattern': r'^[A-Z0-9]{8,12}$',
                'valid_lob_codes': ['15', '17'],
                'date_range_years': 10
            },
            'confidence_validation': {
                'min_confidence': 0.0,
                'max_confidence': 1.0,
                'low_confidence_threshold': 0.5
            }
        }
    
    def validate_input_data(self, df: pd.DataFrame, 
                           required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate input data quality and structure.
        
        This method performs comprehensive validation of input data including
        structure validation, data quality checks, and business rule validation.
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            required_columns (Optional[List[str]]): List of required columns
            
        Returns:
            Dict[str, Any]: Validation results with errors, warnings, and statistics
        """
        if required_columns is None:
            required_columns = self.validation_config['required_columns']['input']
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'data_quality_score': 0.0
        }
        
        try:
            # Basic structure validation
            structure_results = self._validate_dataframe_structure(df, required_columns)
            validation_results['errors'].extend(structure_results['errors'])
            validation_results['warnings'].extend(structure_results['warnings'])
            
            if structure_results['errors']:
                validation_results['is_valid'] = False
                return validation_results
            
            # Data quality validation
            quality_results = self._validate_data_quality(df)
            validation_results['warnings'].extend(quality_results['warnings'])
            validation_results['statistics'].update(quality_results['statistics'])
            
            # Business rule validation
            business_results = self._validate_business_rules(df)
            validation_results['warnings'].extend(business_results['warnings'])
            validation_results['statistics'].update(business_results['statistics'])
            
            # Calculate overall data quality score
            validation_results['data_quality_score'] = self._calculate_quality_score(
                df, validation_results
            )
            
            self.logger.info(f"Input validation completed. Quality score: {validation_results['data_quality_score']:.2f}")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Input validation failed: {e}")
        
        return validation_results
    
    def validate_output_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate output data quality and completeness.
        
        Args:
            df (pd.DataFrame): Output dataframe to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        required_columns = self.validation_config['required_columns']['output']
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Structure validation
            structure_results = self._validate_dataframe_structure(df, required_columns)
            validation_results['errors'].extend(structure_results['errors'])
            validation_results['warnings'].extend(structure_results['warnings'])
            
            if structure_results['errors']:
                validation_results['is_valid'] = False
                return validation_results
            
            # Output-specific validation
            output_results = self._validate_output_specific(df)
            validation_results['warnings'].extend(output_results['warnings'])
            validation_results['statistics'].update(output_results['statistics'])
            
            self.logger.info("Output validation completed successfully")
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Output validation error: {str(e)}")
            self.logger.error(f"Output validation failed: {e}")
        
        return validation_results
    
    def _validate_dataframe_structure(self, df: pd.DataFrame, 
                                    required_columns: List[str]) -> Dict[str, Any]:
        """
        Validate basic dataframe structure.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            required_columns (List[str]): Required columns
            
        Returns:
            Dict[str, Any]: Structure validation results
        """
        results = {'errors': [], 'warnings': []}
        
        # Check if dataframe is empty
        if df.empty:
            results['errors'].append("Dataframe is empty")
            return results
        
        # Check for required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['errors'].append(f"Missing required columns: {list(missing_columns)}")
        
        # Check for duplicate columns
        duplicate_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicate_columns:
            results['warnings'].append(f"Duplicate column names: {duplicate_columns}")
        
        # Check dataframe size
        if len(df) > 100000:
            results['warnings'].append(f"Large dataset detected: {len(df)} rows")
        
        return results
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality metrics.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            Dict[str, Any]: Data quality validation results
        """
        results = {'warnings': [], 'statistics': {}}
        
        # Check for duplicates
        if 'CLAIMNO' in df.columns:
            duplicates = df['CLAIMNO'].duplicated().sum()
            results['statistics']['duplicate_claims'] = duplicates
            if duplicates > 0:
                results['warnings'].append(f"Found {duplicates} duplicate claim numbers")
        
        # Check for missing values
        missing_stats = df.isnull().sum()
        results['statistics']['missing_values'] = missing_stats.to_dict()
        
        for column, missing_count in missing_stats.items():
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                if missing_pct > 10:  # More than 10% missing
                    results['warnings'].append(
                        f"Column '{column}' has {missing_pct:.1f}% missing values"
                    )
        
        # Check text column quality
        if 'clean_FN_TEXT' in df.columns:
            text_stats = self._validate_text_column(df['clean_FN_TEXT'])
            results['statistics']['text_quality'] = text_stats
            
            if text_stats['empty_text_count'] > 0:
                results['warnings'].append(
                    f"Found {text_stats['empty_text_count']} claims with empty text"
                )
            
            if text_stats['short_text_count'] > 0:
                results['warnings'].append(
                    f"Found {text_stats['short_text_count']} claims with very short text"
                )
        
        return results
    
    def _validate_business_rules(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate business-specific rules.
        
        Args:
            df (pd.DataFrame): Dataframe to validate
            
        Returns:
            Dict[str, Any]: Business rules validation results
        """
        results = {'warnings': [], 'statistics': {}}
        
        # Validate LOB codes
        if 'LOBCD' in df.columns:
            valid_lobs = self.validation_config['claim_validation']['valid_lob_codes']
            invalid_lobs = df[~df['LOBCD'].isin(valid_lobs)]['LOBCD'].value_counts()
            results['statistics']['invalid_lob_codes'] = invalid_lobs.to_dict()
            
            if len(invalid_lobs) > 0:
                results['warnings'].append(
                    f"Found invalid LOB codes: {list(invalid_lobs.index)}"
                )
        
        # Validate claim number format
        if 'CLAIMNO' in df.columns:
            pattern = self.validation_config['claim_validation']['claimno_pattern']
            invalid_claimnos = df[~df['CLAIMNO'].str.match(pattern, na=False)]
            results['statistics']['invalid_claimno_count'] = len(invalid_claimnos)
            
            if len(invalid_claimnos) > 0:
                results['warnings'].append(
                    f"Found {len(invalid_claimnos)} claims with invalid claim number format"
                )
        
        # Validate date ranges
        date_columns = ['LOSSDT', 'REPORTEDDT']
        for col in date_columns:
            if col in df.columns:
                date_stats = self._validate_date_column(df[col])
                results['statistics'][f'{col}_validation'] = date_stats
                
                if date_stats['future_dates'] > 0:
                    results['warnings'].append(
                        f"Found {date_stats['future_dates']} future dates in {col}"
                    )
                
                if date_stats['very_old_dates'] > 0:
                    results['warnings'].append(
                        f"Found {date_stats['very_old_dates']} very old dates in {col}"
                    )
        
        return results
    
    def _validate_output_specific(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate output-specific requirements.
        
        Args:
            df (pd.DataFrame): Output dataframe to validate
            
        Returns:
            Dict[str, Any]: Output validation results
        """
        results = {'warnings': [], 'statistics': {}}
        
        # Validate confidence scores
        if 'confidence' in df.columns:
            conf_config = self.validation_config['confidence_validation']
            
            # Check confidence range
            out_of_range = df[
                (df['confidence'] < conf_config['min_confidence']) |
                (df['confidence'] > conf_config['max_confidence'])
            ]
            results['statistics']['out_of_range_confidence'] = len(out_of_range)
            
            if len(out_of_range) > 0:
                results['warnings'].append(
                    f"Found {len(out_of_range)} predictions with confidence out of valid range"
                )
            
            # Check low confidence predictions
            low_confidence = df[df['confidence'] < conf_config['low_confidence_threshold']]
            results['statistics']['low_confidence_predictions'] = len(low_confidence)
            
            if len(low_confidence) > 0:
                low_pct = (len(low_confidence) / len(df)) * 100
                results['warnings'].append(
                    f"{low_pct:.1f}% of predictions have low confidence"
                )
        
        # Validate prediction completeness
        if 'prediction' in df.columns:
            empty_predictions = df['prediction'].isnull().sum()
            results['statistics']['empty_predictions'] = empty_predictions
            
            if empty_predictions > 0:
                results['warnings'].append(
                    f"Found {empty_predictions} claims with empty predictions"
                )
        
        return results
    
    def _validate_text_column(self, text_series: pd.Series) -> Dict[str, Any]:
        """
        Validate text column quality.
        
        Args:
            text_series (pd.Series): Text series to validate
            
        Returns:
            Dict[str, Any]: Text validation statistics
        """
        text_config = self.validation_config['text_validation']
        
        stats = {
            'total_records': len(text_series),
            'empty_text_count': text_series.isnull().sum(),
            'short_text_count': 0,
            'long_text_count': 0,
            'average_length': 0,
            'encoding_issues': 0
        }
        
        # Filter out null values for length calculations
        valid_text = text_series.dropna()
        
        if len(valid_text) > 0:
            lengths = valid_text.str.len()
            stats['average_length'] = lengths.mean()
            stats['short_text_count'] = (lengths < text_config['min_length']).sum()
            stats['long_text_count'] = (lengths > text_config['max_length']).sum()
            
            # Check for encoding issues (basic check)
            try:
                for text in valid_text.head(100):  # Sample check
                    if isinstance(text, str):
                        text.encode('utf-8')
            except UnicodeEncodeError:
                stats['encoding_issues'] += 1
        
        return stats
    
    def _validate_date_column(self, date_series: pd.Series) -> Dict[str, Any]:
        """
        Validate date column quality.
        
        Args:
            date_series (pd.Series): Date series to validate
            
        Returns:
            Dict[str, Any]: Date validation statistics
        """
        stats = {
            'total_records': len(date_series),
            'null_dates': date_series.isnull().sum(),
            'future_dates': 0,
            'very_old_dates': 0,
            'invalid_dates': 0
        }
        
        # Convert to datetime if not already
        try:
            date_series = pd.to_datetime(date_series, errors='coerce')
            stats['invalid_dates'] = date_series.isnull().sum() - stats['null_dates']
        except Exception:
            stats['invalid_dates'] = len(date_series)
            return stats
        
        if len(date_series.dropna()) > 0:
            today = datetime.now()
            cutoff_date = today - timedelta(days=365 * self.validation_config['claim_validation']['date_range_years'])
            
            stats['future_dates'] = (date_series > today).sum()
            stats['very_old_dates'] = (date_series < cutoff_date).sum()
        
        return stats
    
    def _calculate_quality_score(self, df: pd.DataFrame, 
                                validation_results: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.
        
        Args:
            df (pd.DataFrame): Dataframe being validated
            validation_results (Dict[str, Any]): Validation results
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if df.empty:
            return 0.0
        
        score = 1.0
        
        # Penalize for errors (major impact)
        error_count = len(validation_results['errors'])
        score -= error_count * 0.2
        
        # Penalize for warnings (minor impact)
        warning_count = len(validation_results['warnings'])
        score -= warning_count * 0.05
        
        # Penalize for missing data
        if 'missing_values' in validation_results['statistics']:
            missing_stats = validation_results['statistics']['missing_values']
            total_cells = len(df) * len(df.columns)
            missing_cells = sum(missing_stats.values())
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
            score -= missing_ratio * 0.3
        
        # Penalize for duplicates
        if 'duplicate_claims' in validation_results['statistics']:
            duplicate_ratio = validation_results['statistics']['duplicate_claims'] / len(df)
            score -= duplicate_ratio * 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def validate_pipeline_flow(self, input_df: pd.DataFrame, 
                              output_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the complete pipeline flow from input to output.
        
        Args:
            input_df (pd.DataFrame): Pipeline input data
            output_df (pd.DataFrame): Pipeline output data
            
        Returns:
            Dict[str, Any]: Complete pipeline validation results
        """
        flow_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'input_validation': {},
            'output_validation': {},
            'flow_statistics': {}
        }
        
        try:
            # Validate input
            input_results = self.validate_input_data(input_df)
            flow_results['input_validation'] = input_results
            
            if not input_results['is_valid']:
                flow_results['is_valid'] = False
                flow_results['errors'].extend(input_results['errors'])
            
            # Validate output
            output_results = self.validate_output_data(output_df)
            flow_results['output_validation'] = output_results
            
            if not output_results['is_valid']:
                flow_results['is_valid'] = False
                flow_results['errors'].extend(output_results['errors'])
            
            # Flow-specific validation
            flow_stats = self._validate_flow_consistency(input_df, output_df)
            flow_results['flow_statistics'] = flow_stats
            
            # Check data loss
            if 'claims_processed' in flow_stats:
                processing_rate = flow_stats['claims_processed'] / len(input_df) if len(input_df) > 0 else 0
                if processing_rate < 0.8:  # Less than 80% processed
                    flow_results['warnings'].append(
                        f"Low processing rate: {processing_rate:.1%} of input claims processed"
                    )
            
            self.logger.info("Pipeline flow validation completed")
            
        except Exception as e:
            flow_results['is_valid'] = False
            flow_results['errors'].append(f"Pipeline flow validation error: {str(e)}")
            self.logger.error(f"Pipeline flow validation failed: {e}")
        
        return flow_results
    
    def _validate_flow_consistency(self, input_df: pd.DataFrame, 
                                 output_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate consistency between input and output data.
        
        Args:
            input_df (pd.DataFrame): Input dataframe
            output_df (pd.DataFrame): Output dataframe
            
        Returns:
            Dict[str, Any]: Flow consistency statistics
        """
        stats = {
            'input_claims': len(input_df),
            'output_claims': len(output_df),
            'claims_processed': 0,
            'claims_lost': 0,
            'processing_rate': 0.0
        }
        
        if 'CLAIMNO' in input_df.columns and 'CLAIMNO' in output_df.columns:
            input_claims = set(input_df['CLAIMNO'])
            output_claims = set(output_df['CLAIMNO'])
            
            processed_claims = input_claims.intersection(output_claims)
            lost_claims = input_claims - output_claims
            
            stats['claims_processed'] = len(processed_claims)
            stats['claims_lost'] = len(lost_claims)
            stats['processing_rate'] = len(processed_claims) / len(input_claims) if len(input_claims) > 0 else 0.0
        
        return stats