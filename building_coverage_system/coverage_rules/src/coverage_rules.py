"""
Coverage rules engine for building coverage determination.

This module contains the original Codebase 1 rules engine that applies
business rules and classification logic for building coverage claims.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


class CoverageRules:
    """
    Rules engine for building coverage classification.
    
    This class implements the original Codebase 1 business rules
    for determining building coverage requirements based on claim
    characteristics and business logic.
    """
    
    def __init__(self, rule_config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the coverage rules engine.
        
        Args:
            rule_config (Optional[Dict[str, Any]]): Rules configuration
            logger (Optional[logging.Logger]): Logger instance
        """
        self.rule_config = rule_config or self._get_default_config()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize rule sets
        self.building_keywords = self._load_building_keywords()
        self.exclusion_keywords = self._load_exclusion_keywords()
        self.lob_rules = self._load_lob_rules()
        self.text_patterns = self._load_text_patterns()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'rules_applied': 0,
            'overrides_applied': 0,
            'confidence_adjustments': 0
        }
        
        self.logger.info("CoverageRules engine initialized")
    
    def classify_rule_conditions(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply classification rules to claims data.
        
        Args:
            claims_df (pd.DataFrame): Claims data with predictions
            
        Returns:
            pd.DataFrame: Claims with rules applied
        """
        self.logger.info(f"Applying coverage rules to {len(claims_df)} claims")
        
        # Make a copy to avoid modifying original data
        result_df = claims_df.copy()
        
        # Apply rules in sequence
        result_df = self._apply_lob_code_rules(result_df)
        result_df = self._apply_keyword_rules(result_df)
        result_df = self._apply_text_pattern_rules(result_df)
        result_df = self._apply_confidence_adjustments(result_df)
        result_df = self._apply_business_logic_overrides(result_df)
        result_df = self._apply_final_validation(result_df)
        
        # Add rule metadata
        result_df['rules_applied'] = True
        result_df['rules_timestamp'] = datetime.now()
        result_df['rules_version'] = self.rule_config.get('version', '1.0')
        
        # Update statistics
        self.stats['total_processed'] += len(claims_df)
        self.stats['rules_applied'] += 1
        
        self.logger.info(f"Rules application completed for {len(result_df)} claims")
        
        return result_df
    
    def _apply_lob_code_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply LOB (Line of Business) code specific rules.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with LOB rules applied
        """
        if 'LOBCD' not in df.columns:
            self.logger.warning("LOBCD column not found, skipping LOB rules")
            return df
        
        for lob_code, rules in self.lob_rules.items():
            lob_mask = df['LOBCD'] == lob_code
            
            if not lob_mask.any():
                continue
            
            self.logger.debug(f"Applying LOB rules for code {lob_code} to {lob_mask.sum()} claims")
            
            # Apply LOB-specific confidence adjustments
            if 'confidence_multiplier' in rules and 'confidence' in df.columns:
                df.loc[lob_mask, 'confidence'] *= rules['confidence_multiplier']
            
            # Apply LOB-specific prediction overrides
            if 'default_prediction' in rules and 'confidence' in df.columns:
                low_confidence_mask = lob_mask & (df['confidence'] < rules.get('confidence_threshold', 0.6))
                if low_confidence_mask.any():
                    df.loc[low_confidence_mask, 'prediction'] = rules['default_prediction']
                    df.loc[low_confidence_mask, 'rule_override'] = f'LOB_{lob_code}_default'
            
            # Apply LOB-specific exclusions
            if 'exclusions' in rules:
                for exclusion in rules['exclusions']:
                    if 'clean_FN_TEXT' in df.columns:
                        exclusion_mask = lob_mask & df['clean_FN_TEXT'].str.contains(
                            exclusion, case=False, na=False
                        )
                        if exclusion_mask.any():
                            df.loc[exclusion_mask, 'prediction'] = 'NO BUILDING COVERAGE'
                            df.loc[exclusion_mask, 'rule_override'] = f'LOB_{lob_code}_exclusion'
        
        return df
    
    def _apply_keyword_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply keyword-based classification rules.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with keyword rules applied
        """
        if 'clean_FN_TEXT' not in df.columns:
            self.logger.warning("clean_FN_TEXT column not found, skipping keyword rules")
            return df
        
        # Calculate keyword scores for each claim
        df['building_keyword_score'] = df['clean_FN_TEXT'].apply(self._calculate_keyword_score)
        df['exclusion_keyword_count'] = df['clean_FN_TEXT'].apply(self._count_exclusion_keywords)
        
        # Apply keyword-based rules
        
        # Rule 1: Strong building keywords override low confidence predictions
        strong_building_mask = (
            (df['building_keyword_score'] >= 5) & 
            (df.get('prediction', '') == 'NO BUILDING COVERAGE') &
            (df.get('confidence', 0) < 0.7)
        )
        if strong_building_mask.any():
            df.loc[strong_building_mask, 'prediction'] = 'BUILDING COVERAGE'
            df.loc[strong_building_mask, 'confidence'] = 0.75
            df.loc[strong_building_mask, 'rule_override'] = 'strong_building_keywords'
            self.logger.info(f"Applied strong building keyword override to {strong_building_mask.sum()} claims")
        
        # Rule 2: Multiple exclusion keywords override building predictions
        strong_exclusion_mask = (
            (df['exclusion_keyword_count'] >= 3) &
            (df.get('prediction', '') == 'BUILDING COVERAGE') &
            (df.get('confidence', 1) < 0.8)
        )
        if strong_exclusion_mask.any():
            df.loc[strong_exclusion_mask, 'prediction'] = 'NO BUILDING COVERAGE'
            df.loc[strong_exclusion_mask, 'confidence'] = 0.8
            df.loc[strong_exclusion_mask, 'rule_override'] = 'strong_exclusion_keywords'
            self.logger.info(f"Applied strong exclusion keyword override to {strong_exclusion_mask.sum()} claims")
        
        # Rule 3: Adjust confidence based on keyword strength
        if 'confidence' in df.columns:
            # Boost confidence for strong building keywords
            boost_mask = df['building_keyword_score'] >= 3
            df.loc[boost_mask, 'confidence'] = np.minimum(
                df.loc[boost_mask, 'confidence'] * 1.1, 0.95
            )
            
            # Reduce confidence for exclusion keywords
            reduce_mask = df['exclusion_keyword_count'] >= 1
            df.loc[reduce_mask, 'confidence'] = np.maximum(
                df.loc[reduce_mask, 'confidence'] * 0.9, 0.1
            )
        
        return df
    
    def _apply_text_pattern_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text pattern-based rules.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with text pattern rules applied
        """
        if 'clean_FN_TEXT' not in df.columns:
            return df
        
        for pattern_name, pattern_config in self.text_patterns.items():
            pattern = pattern_config['pattern']
            action = pattern_config['action']
            confidence_adjustment = pattern_config.get('confidence_adjustment', 1.0)
            
            # Find claims matching the pattern
            pattern_mask = df['clean_FN_TEXT'].str.contains(pattern, case=False, na=False, regex=True)
            
            if not pattern_mask.any():
                continue
            
            self.logger.debug(f"Applied pattern '{pattern_name}' to {pattern_mask.sum()} claims")
            
            # Apply the action
            if action == 'force_building_coverage':
                df.loc[pattern_mask, 'prediction'] = 'BUILDING COVERAGE'
                df.loc[pattern_mask, 'rule_override'] = f'pattern_{pattern_name}'
            elif action == 'force_no_coverage':
                df.loc[pattern_mask, 'prediction'] = 'NO BUILDING COVERAGE'
                df.loc[pattern_mask, 'rule_override'] = f'pattern_{pattern_name}'
            elif action == 'adjust_confidence' and 'confidence' in df.columns:
                df.loc[pattern_mask, 'confidence'] *= confidence_adjustment
                df.loc[pattern_mask, 'confidence'] = df.loc[pattern_mask, 'confidence'].clip(0.1, 0.95)
        
        return df
    
    def _apply_confidence_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply confidence-based adjustments and thresholds.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with confidence adjustments applied
        """
        if 'confidence' not in df.columns:
            return df
        
        # Confidence thresholds from config
        high_confidence_threshold = self.rule_config.get('high_confidence_threshold', 0.8)
        low_confidence_threshold = self.rule_config.get('low_confidence_threshold', 0.4)
        
        # Rule 1: Flag very low confidence predictions for manual review
        very_low_confidence_mask = df['confidence'] < low_confidence_threshold
        if very_low_confidence_mask.any():
            df.loc[very_low_confidence_mask, 'requires_manual_review'] = True
            df.loc[very_low_confidence_mask, 'review_reason'] = 'low_confidence'
            self.logger.info(f"Flagged {very_low_confidence_mask.sum()} claims for manual review (low confidence)")
        
        # Rule 2: High confidence predictions get additional validation
        high_confidence_mask = df['confidence'] >= high_confidence_threshold
        if high_confidence_mask.any():
            df.loc[high_confidence_mask, 'high_confidence_flag'] = True
        
        # Rule 3: Adjust confidence based on text quality
        if 'clean_FN_TEXT' in df.columns:
            text_length = df['clean_FN_TEXT'].str.len()
            
            # Reduce confidence for very short text
            short_text_mask = text_length < 50
            df.loc[short_text_mask, 'confidence'] *= 0.7
            
            # Slightly reduce confidence for moderately short text
            medium_text_mask = (text_length >= 50) & (text_length < 100)
            df.loc[medium_text_mask, 'confidence'] *= 0.9
        
        # Rule 4: Ensure confidence bounds
        df['confidence'] = df['confidence'].clip(0.05, 0.98)
        
        self.stats['confidence_adjustments'] += 1
        
        return df
    
    def _apply_business_logic_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply business logic overrides and special cases.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with business logic applied
        """
        override_count = 0
        
        # Override 1: Claims with very high amounts typically involve building coverage
        if 'RESERVE_TOTAL' in df.columns:
            high_amount_threshold = self.rule_config.get('high_amount_threshold', 100000)
            high_amount_mask = (
                (df['RESERVE_TOTAL'] > high_amount_threshold) &
                (df.get('prediction', '') == 'NO BUILDING COVERAGE') &
                (df.get('confidence', 1) < 0.9)
            )
            if high_amount_mask.any():
                df.loc[high_amount_mask, 'prediction'] = 'BUILDING COVERAGE'
                df.loc[high_amount_mask, 'confidence'] = 0.7
                df.loc[high_amount_mask, 'rule_override'] = 'high_reserve_amount'
                override_count += high_amount_mask.sum()
        
        # Override 2: Recent claims with building indicators
        if 'LOSSDT' in df.columns:
            df['claim_age_days'] = (datetime.now() - pd.to_datetime(df['LOSSDT'], errors='coerce')).dt.days
            recent_threshold = self.rule_config.get('recent_claim_days', 30)
            
            recent_building_mask = (
                (df['claim_age_days'] <= recent_threshold) &
                (df.get('building_keyword_score', 0) >= 2) &
                (df.get('prediction', '') == 'NO BUILDING COVERAGE')
            )
            if recent_building_mask.any():
                df.loc[recent_building_mask, 'requires_manual_review'] = True
                df.loc[recent_building_mask, 'review_reason'] = 'recent_with_building_indicators'
                override_count += recent_building_mask.sum()
        
        # Override 3: Consistent patterns across similar claims
        if len(df) > 1 and 'clean_FN_TEXT' in df.columns:
            # Group similar claims and check for consistency
            df = self._apply_consistency_checks(df)
        
        if override_count > 0:
            self.stats['overrides_applied'] += override_count
            self.logger.info(f"Applied business logic overrides to {override_count} claims")
        
        return df
    
    def _apply_consistency_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply consistency checks across similar claims.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with consistency checks applied
        """
        # Simple consistency check based on text similarity
        if len(df) < 2:
            return df
        
        # Group claims by similar characteristics
        df['text_length_bucket'] = pd.cut(df['clean_FN_TEXT'].str.len(), bins=5, labels=False)
        df['keyword_score_bucket'] = pd.cut(df.get('building_keyword_score', 0), bins=3, labels=False)
        
        # Check for inconsistencies within groups
        for (length_bucket, keyword_bucket), group in df.groupby(['text_length_bucket', 'keyword_score_bucket']):
            if len(group) < 2:
                continue
            
            predictions = group['prediction'].value_counts()
            if len(predictions) > 1:  # Mixed predictions in similar group
                # Flag minority predictions for review
                minority_prediction = predictions.index[-1]  # Least common
                minority_mask = (group['prediction'] == minority_prediction) & (group.get('confidence', 1) < 0.8)
                
                if minority_mask.any():
                    minority_indices = group[minority_mask].index
                    df.loc[minority_indices, 'requires_manual_review'] = True
                    df.loc[minority_indices, 'review_reason'] = 'inconsistent_with_similar_claims'
        
        # Clean up temporary columns
        df = df.drop(['text_length_bucket', 'keyword_score_bucket'], axis=1, errors='ignore')
        
        return df
    
    def _apply_final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply final validation and quality checks.
        
        Args:
            df (pd.DataFrame): Input claims data
            
        Returns:
            pd.DataFrame: Claims with final validation applied
        """
        # Ensure all required fields are present
        if 'prediction' not in df.columns:
            df['prediction'] = 'UNCLEAR'
        
        if 'confidence' not in df.columns:
            df['confidence'] = 0.5
        
        # Final confidence bounds
        df['confidence'] = df['confidence'].clip(0.05, 0.98)
        
        # Validate prediction values
        valid_predictions = ['BUILDING COVERAGE', 'NO BUILDING COVERAGE', 'UNCLEAR']
        invalid_mask = ~df['prediction'].isin(valid_predictions)
        if invalid_mask.any():
            df.loc[invalid_mask, 'prediction'] = 'UNCLEAR'
            df.loc[invalid_mask, 'confidence'] = 0.3
            self.logger.warning(f"Fixed {invalid_mask.sum()} invalid predictions")
        
        # Add rule processing summary
        df['rule_processing_complete'] = True
        df['rule_quality_score'] = self._calculate_rule_quality_score(df)
        
        return df
    
    def _calculate_keyword_score(self, text: str) -> float:
        """
        Calculate building keyword score for text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Keyword score
        """
        if not isinstance(text, str):
            return 0.0
        
        text_lower = text.lower()
        total_score = 0.0
        
        for keyword, weight in self.building_keywords.items():
            # Count occurrences with word boundaries
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = len(re.findall(pattern, text_lower))
            total_score += count * weight
        
        return total_score
    
    def _count_exclusion_keywords(self, text: str) -> int:
        """
        Count exclusion keywords in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            int: Count of exclusion keywords
        """
        if not isinstance(text, str):
            return 0
        
        text_lower = text.lower()
        count = 0
        
        for keyword in self.exclusion_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                count += 1
        
        return count
    
    def _calculate_rule_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate quality score based on rule processing.
        
        Args:
            df (pd.DataFrame): Processed claims data
            
        Returns:
            pd.Series: Quality scores
        """
        quality_score = pd.Series(0.8, index=df.index)  # Start with base score
        
        # Adjust based on confidence
        if 'confidence' in df.columns:
            high_conf_mask = df['confidence'] >= 0.8
            quality_score.loc[high_conf_mask] += 0.1
            
            low_conf_mask = df['confidence'] < 0.5
            quality_score.loc[low_conf_mask] -= 0.2
        
        # Adjust based on rule overrides
        if 'rule_override' in df.columns:
            override_mask = df['rule_override'].notna()
            quality_score.loc[override_mask] += 0.05
        
        # Adjust based on keyword strength
        if 'building_keyword_score' in df.columns:
            strong_keywords_mask = df['building_keyword_score'] >= 3
            quality_score.loc[strong_keywords_mask] += 0.05
        
        return quality_score.clip(0.1, 1.0)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default rule configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'version': '1.0',
            'high_confidence_threshold': 0.8,
            'low_confidence_threshold': 0.4,
            'high_amount_threshold': 100000,
            'recent_claim_days': 30,
            'enable_consistency_checks': True,
            'enable_business_overrides': True
        }
    
    def _load_building_keywords(self) -> Dict[str, float]:
        """
        Load building-related keywords with weights.
        
        Returns:
            Dict[str, float]: Keyword to weight mapping
        """
        return {
            # Strong indicators (weight 3.0)
            'building': 3.0, 'structure': 3.0, 'structural': 3.0, 'foundation': 3.0,
            'roof': 3.0, 'wall': 3.0, 'floor': 3.0, 'ceiling': 3.0,
            
            # Moderate indicators (weight 2.0)
            'construction': 2.0, 'architectural': 2.0, 'framework': 2.0,
            'masonry': 2.0, 'concrete': 2.0, 'framing': 2.0, 'drywall': 2.0,
            
            # Weak indicators (weight 1.0)
            'window': 1.0, 'door': 1.0, 'fixture': 1.0, 'cabinet': 1.0,
            'flooring': 1.0, 'insulation': 1.0, 'built-in': 1.0, 'permanent': 1.0
        }
    
    def _load_exclusion_keywords(self) -> List[str]:
        """
        Load exclusion keywords that indicate non-building coverage.
        
        Returns:
            List[str]: List of exclusion keywords
        """
        return [
            'furniture', 'equipment', 'machinery', 'vehicle', 'contents',
            'personal property', 'inventory', 'stock', 'supplies',
            'landscaping', 'lawn', 'garden', 'fence', 'driveway',
            'liability', 'business interruption', 'lost income',
            'automobile', 'truck', 'car', 'motorcycle'
        ]
    
    def _load_lob_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Load LOB-specific rules.
        
        Returns:
            Dict[str, Dict[str, Any]]: LOB code to rules mapping
        """
        return {
            '15': {  # Property
                'confidence_multiplier': 1.1,
                'confidence_threshold': 0.6,
                'default_prediction': 'BUILDING COVERAGE'
            },
            '17': {  # Casualty
                'confidence_multiplier': 1.05,
                'confidence_threshold': 0.7,
                'exclusions': ['liability', 'bodily injury']
            },
            '18': {  # Auto
                'confidence_multiplier': 0.8,
                'confidence_threshold': 0.8,
                'default_prediction': 'NO BUILDING COVERAGE',
                'exclusions': ['vehicle', 'automobile', 'collision']
            }
        }
    
    def _load_text_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Load text pattern rules.
        
        Returns:
            Dict[str, Dict[str, Any]]: Pattern name to configuration mapping
        """
        return {
            'foundation_damage': {
                'pattern': r'foundation.{0,20}(crack|damage|settle|shift)',
                'action': 'force_building_coverage',
                'confidence_adjustment': 1.2
            },
            'structural_collapse': {
                'pattern': r'(collapse|structural.{0,20}fail)',
                'action': 'force_building_coverage',
                'confidence_adjustment': 1.3
            },
            'vehicle_only': {
                'pattern': r'vehicle.{0,50}damage.{0,50}no.{0,20}building',
                'action': 'force_no_coverage',
                'confidence_adjustment': 1.2
            },
            'contents_only': {
                'pattern': r'contents.{0,20}only|personal.{0,20}property.{0,20}only',
                'action': 'force_no_coverage',
                'confidence_adjustment': 1.1
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get rule processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """
        Reset processing statistics.
        """
        self.stats = {
            'total_processed': 0,
            'rules_applied': 0,
            'overrides_applied': 0,
            'confidence_adjustments': 0
        }
        
        self.logger.info("Rules engine statistics reset")