"""
Tests for coverage_rules module.

This module contains unit tests for the business rules engine
and data transformation functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from building_coverage_system.coverage_rules.src.business_rules_engine import (
    BusinessRulesEngine,
    Rule,
    RuleCondition,
    RuleAction,
    create_rules_engine
)
from building_coverage_system.coverage_rules.src.transforms import (
    select_and_rename_bldg_predictions_for_db,
    format_currency_values,
    format_date_values,
    format_text_values,
    validate_required_fields,
    create_database_transformer
)
from building_coverage_system.tests.fixtures.sample_data import (
    create_sample_claims_data,
    create_sample_rules_data,
    create_sample_classifications_data
)


class TestBusinessRulesEngine:
    """Test cases for BusinessRulesEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_rules = create_sample_rules_data(10)
        self.rules_engine = BusinessRulesEngine()
        
        # Load sample rules
        for rule_data in self.sample_rules:
            rule = Rule.from_dict(rule_data)
            self.rules_engine.add_rule(rule)
    
    def test_add_rule(self):
        """Test adding a rule to the engine."""
        new_rule = Rule(
            rule_id='TEST_001',
            name='Test Rule',
            conditions=[RuleCondition('test_field', '==', 'test_value')],
            action=RuleAction('classify', 'TEST_COVERAGE'),
            priority=100
        )
        
        engine = BusinessRulesEngine()
        engine.add_rule(new_rule)
        
        assert len(engine.rules) == 1
        assert engine.rules[0].rule_id == 'TEST_001'
    
    def test_remove_rule(self):
        """Test removing a rule from the engine."""
        initial_count = len(self.rules_engine.rules)
        
        # Remove first rule
        first_rule_id = self.rules_engine.rules[0].rule_id
        removed = self.rules_engine.remove_rule(first_rule_id)
        
        assert removed is True
        assert len(self.rules_engine.rules) == initial_count - 1
        
        # Try to remove non-existent rule
        removed = self.rules_engine.remove_rule('NON_EXISTENT')
        assert removed is False
    
    def test_evaluate_rules_single_record(self):
        """Test evaluating rules against a single record."""
        test_record = {
            'loss_amount': 15000,
            'lob_code': 'HO',
            'loss_description': 'fire damage to kitchen',
            'policy_limit': 150000
        }
        
        results = self.rules_engine.evaluate_rules(test_record)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that each result has required fields
        for result in results:
            assert 'rule_id' in result
            assert 'matched' in result
            assert 'action' in result
    
    def test_evaluate_rules_dataframe(self):
        """Test evaluating rules against a dataframe."""
        claims_df = create_sample_claims_data(50)
        
        # Add required fields for rules evaluation
        claims_df['loss_amount'] = claims_df['INCURRED_TOTAL']
        claims_df['lob_code'] = claims_df['LOBCD']
        claims_df['loss_description'] = claims_df['LOSSDESC']
        claims_df['policy_limit'] = 100000
        
        results_df = self.rules_engine.evaluate_rules(claims_df)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(claims_df)
        assert 'rule_results' in results_df.columns
    
    def test_get_rule_statistics(self):
        """Test getting rule execution statistics."""
        # Evaluate some rules to generate statistics
        test_record = {
            'loss_amount': 15000,
            'lob_code': 'HO',
            'loss_description': 'fire damage',
            'policy_limit': 150000
        }
        
        self.rules_engine.evaluate_rules(test_record)
        
        stats = self.rules_engine.get_rule_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_evaluations' in stats
        assert 'rule_hit_counts' in stats
        assert 'execution_times' in stats
    
    def test_rule_priority_ordering(self):
        """Test that rules are executed in priority order."""
        engine = BusinessRulesEngine()
        
        # Add rules with different priorities
        low_priority_rule = Rule(
            rule_id='LOW_PRIORITY',
            name='Low Priority Rule',
            conditions=[RuleCondition('test_field', '==', 'test_value')],
            action=RuleAction('classify', 'LOW'),
            priority=50
        )
        
        high_priority_rule = Rule(
            rule_id='HIGH_PRIORITY',
            name='High Priority Rule',
            conditions=[RuleCondition('test_field', '==', 'test_value')],
            action=RuleAction('classify', 'HIGH'),
            priority=150
        )
        
        engine.add_rule(low_priority_rule)
        engine.add_rule(high_priority_rule)
        
        # Rules should be sorted by priority (highest first)
        assert engine.rules[0].priority == 150
        assert engine.rules[1].priority == 50
    
    def test_rule_condition_evaluation(self):
        """Test individual rule condition evaluation."""
        # Test equality condition
        condition = RuleCondition('amount', '==', 1000)
        record = {'amount': 1000}
        assert condition.evaluate(record) is True
        
        record = {'amount': 500}
        assert condition.evaluate(record) is False
        
        # Test greater than condition
        condition = RuleCondition('amount', '>', 1000)
        record = {'amount': 1500}
        assert condition.evaluate(record) is True
        
        record = {'amount': 500}
        assert condition.evaluate(record) is False
        
        # Test contains condition
        condition = RuleCondition('description', 'contains', 'fire')
        record = {'description': 'house fire damage'}
        assert condition.evaluate(record) is True
        
        record = {'description': 'water damage'}
        assert condition.evaluate(record) is False
        
        # Test 'in' condition
        condition = RuleCondition('category', 'in', ['A', 'B', 'C'])
        record = {'category': 'B'}
        assert condition.evaluate(record) is True
        
        record = {'category': 'D'}
        assert condition.evaluate(record) is False


class TestRule:
    """Test cases for Rule class."""
    
    def test_rule_creation(self):
        """Test creating a rule object."""
        rule = Rule(
            rule_id='TEST_001',
            name='Test Rule',
            conditions=[RuleCondition('field1', '==', 'value1')],
            action=RuleAction('classify', 'BUILDING_COVERAGE'),
            priority=100
        )
        
        assert rule.rule_id == 'TEST_001'
        assert rule.name == 'Test Rule'
        assert len(rule.conditions) == 1
        assert rule.action.action_type == 'classify'
        assert rule.priority == 100
        assert rule.active is True  # default
    
    def test_rule_from_dict(self):
        """Test creating rule from dictionary."""
        rule_dict = {
            'rule_id': 'DICT_001',
            'rule_name': 'Dictionary Rule',
            'conditions': {
                'field': 'loss_amount',
                'operator': '>',
                'value': 5000
            },
            'action': {
                'type': 'classify',
                'value': 'HIGH_VALUE_CLAIM'
            },
            'priority': 90,
            'active': True
        }
        
        rule = Rule.from_dict(rule_dict)
        
        assert rule.rule_id == 'DICT_001'
        assert rule.name == 'Dictionary Rule'
        assert len(rule.conditions) == 1
        assert rule.conditions[0].field == 'loss_amount'
        assert rule.action.value == 'HIGH_VALUE_CLAIM'
    
    def test_rule_evaluation(self):
        """Test rule evaluation against record."""
        rule = Rule(
            rule_id='EVAL_001',
            name='Evaluation Test',
            conditions=[
                RuleCondition('amount', '>', 1000),
                RuleCondition('type', '==', 'fire')
            ],
            action=RuleAction('classify', 'FIRE_DAMAGE'),
            priority=100
        )
        
        # Record that matches all conditions
        matching_record = {'amount': 1500, 'type': 'fire'}
        result = rule.evaluate(matching_record)
        assert result['matched'] is True
        assert result['action']['value'] == 'FIRE_DAMAGE'
        
        # Record that doesn't match all conditions
        non_matching_record = {'amount': 500, 'type': 'fire'}
        result = rule.evaluate(non_matching_record)
        assert result['matched'] is False


class TestRuleCondition:
    """Test cases for RuleCondition class."""
    
    def test_condition_operators(self):
        """Test different condition operators."""
        test_record = {
            'numeric_field': 100,
            'string_field': 'test_value',
            'list_field': ['item1', 'item2'],
            'null_field': None
        }
        
        # Test equality
        condition = RuleCondition('numeric_field', '==', 100)
        assert condition.evaluate(test_record) is True
        
        # Test inequality
        condition = RuleCondition('numeric_field', '!=', 200)
        assert condition.evaluate(test_record) is True
        
        # Test greater than
        condition = RuleCondition('numeric_field', '>', 50)
        assert condition.evaluate(test_record) is True
        
        # Test less than
        condition = RuleCondition('numeric_field', '<', 200)
        assert condition.evaluate(test_record) is True
        
        # Test contains
        condition = RuleCondition('string_field', 'contains', 'test')
        assert condition.evaluate(test_record) is True
        
        # Test is_null
        condition = RuleCondition('null_field', 'is_null', None)
        assert condition.evaluate(test_record) is True
        
        # Test is_not_null
        condition = RuleCondition('string_field', 'is_not_null', None)
        assert condition.evaluate(test_record) is True


class TestTransforms:
    """Test cases for transform functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = create_sample_classifications_data(50)
    
    def test_select_and_rename_bldg_predictions_for_db(self):
        """Test database column selection and renaming."""
        # Create test dataframe with prediction data
        predictions_df = pd.DataFrame({
            'claim_id': ['CLM001', 'CLM002', 'CLM003'],
            'coverage_type': ['DWELLING_COVERAGE', 'PERSONAL_PROPERTY', 'LIABILITY'],
            'confidence_score': [0.95, 0.87, 0.92],
            'model_version': ['v2.1.0', 'v2.1.0', 'v2.1.0'],
            'prediction_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'extra_column': ['value1', 'value2', 'value3']  # Should be excluded
        })
        
        result_df = select_and_rename_bldg_predictions_for_db(predictions_df)
        
        # Check that correct columns are present
        expected_columns = ['CLAIMNO', 'COVERAGE_TYPE', 'CONFIDENCE_SCORE', 'MODEL_VERSION', 'PREDICTION_DT']
        assert all(col in result_df.columns for col in expected_columns)
        
        # Check that extra column is excluded
        assert 'extra_column' not in result_df.columns
        
        # Check data integrity
        assert len(result_df) == len(predictions_df)
        assert result_df['CLAIMNO'].iloc[0] == 'CLM001'
    
    def test_format_currency_values(self):
        """Test currency value formatting."""
        test_df = pd.DataFrame({
            'amount': [1000.50, 2500.75, None, 0],
            'reserve': [5000.00, 10000.25, 7500.50, None]
        })
        
        formatted_df = format_currency_values(test_df, ['amount', 'reserve'])
        
        # Check formatting
        assert formatted_df['amount'].iloc[0] == 1000.50
        assert formatted_df['reserve'].iloc[0] == 5000.00
        
        # Check null handling
        assert pd.isna(formatted_df['amount'].iloc[2])
        assert pd.isna(formatted_df['reserve'].iloc[3])
    
    def test_format_date_values(self):
        """Test date value formatting."""
        test_df = pd.DataFrame({
            'loss_date': ['2023-01-15', '2023-02-20', None],
            'report_date': ['01/15/2023', '02/20/2023', '03/25/2023']
        })
        
        formatted_df = format_date_values(test_df, ['loss_date', 'report_date'])
        
        # Check that dates are converted to datetime
        assert pd.api.types.is_datetime64_any_dtype(formatted_df['loss_date'])
        assert pd.api.types.is_datetime64_any_dtype(formatted_df['report_date'])
        
        # Check null handling
        assert pd.isna(formatted_df['loss_date'].iloc[2])
    
    def test_format_text_values(self):
        """Test text value formatting."""
        test_df = pd.DataFrame({
            'description': ['  Leading spaces', 'Trailing spaces  ', '  Both  ', None],
            'category': ['lowercase', 'UPPERCASE', 'MixedCase', '']
        })
        
        formatted_df = format_text_values(test_df, ['description', 'category'])
        
        # Check trimming
        assert formatted_df['description'].iloc[0] == 'Leading spaces'
        assert formatted_df['description'].iloc[1] == 'Trailing spaces'
        assert formatted_df['description'].iloc[2] == 'Both'
        
        # Check null and empty string handling
        assert pd.isna(formatted_df['description'].iloc[3])
        assert formatted_df['category'].iloc[3] == ''
    
    def test_validate_required_fields(self):
        """Test required field validation."""
        test_df = pd.DataFrame({
            'claim_id': ['CLM001', 'CLM002', None, 'CLM004'],
            'amount': [1000, None, 2000, 3000],
            'description': ['Fire damage', '', 'Water damage', 'Wind damage']
        })
        
        required_fields = ['claim_id', 'amount']
        validation_result = validate_required_fields(test_df, required_fields)
        
        assert 'is_valid' in validation_result
        assert 'missing_fields' in validation_result
        assert 'rows_with_missing' in validation_result
        
        # Should find missing values
        assert validation_result['is_valid'] is False
        assert len(validation_result['rows_with_missing']) > 0
    
    def test_create_database_transformer(self):
        """Test database transformer factory function."""
        transformer = create_database_transformer()
        
        # Should return a callable transformer function
        assert callable(transformer)
        
        # Test with sample data
        test_df = pd.DataFrame({
            'claim_id': ['CLM001', 'CLM002'],
            'coverage_type': ['DWELLING', 'PROPERTY'],
            'confidence_score': [0.95, 0.87]
        })
        
        transformed_df = transformer(test_df)
        
        # Should return a dataframe
        assert isinstance(transformed_df, pd.DataFrame)
        assert len(transformed_df) == len(test_df)


class TestRulesEngineIntegration:
    """Integration tests for the rules engine."""
    
    def test_full_rules_workflow(self):
        """Test complete rules evaluation workflow."""
        # Create rules engine
        engine = create_rules_engine()
        
        # Load sample rules
        sample_rules = create_sample_rules_data(5)
        for rule_data in sample_rules:
            rule = Rule.from_dict(rule_data)
            engine.add_rule(rule)
        
        # Get sample claims data
        claims_df = create_sample_claims_data(20)
        
        # Prepare data for rules evaluation
        claims_df['loss_amount'] = claims_df['INCURRED_TOTAL']
        claims_df['lob_code'] = claims_df['LOBCD']
        claims_df['loss_description'] = claims_df['LOSSDESC']
        claims_df['policy_limit'] = 100000
        
        # Evaluate rules
        results_df = engine.evaluate_rules(claims_df)
        
        # Verify results
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == len(claims_df)
        assert 'rule_results' in results_df.columns
        
        # Get statistics
        stats = engine.get_rule_statistics()
        assert stats['total_evaluations'] > 0
    
    @patch('building_coverage_system.coverage_rules.src.business_rules_engine.logger')
    def test_rules_engine_logging(self, mock_logger):
        """Test rules engine logging functionality."""
        engine = BusinessRulesEngine()
        
        rule = Rule(
            rule_id='LOG_TEST',
            name='Logging Test Rule',
            conditions=[RuleCondition('test_field', '==', 'test_value')],
            action=RuleAction('classify', 'TEST_RESULT'),
            priority=100
        )
        
        engine.add_rule(rule)
        
        # Verify logging was called
        mock_logger.info.assert_called()
    
    def test_transform_pipeline(self):
        """Test complete data transformation pipeline."""
        # Create sample prediction data
        predictions_df = pd.DataFrame({
            'claim_id': ['CLM001', 'CLM002', 'CLM003'],
            'coverage_type': ['DWELLING_COVERAGE', 'PERSONAL_PROPERTY', 'LIABILITY'],
            'confidence_score': [0.95, 0.87, 0.92],
            'model_version': ['v2.1.0', 'v2.1.0', 'v2.1.0'],
            'prediction_date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'amount_field': [1000.50, 2500.75, 3000.00]
        })
        
        # Apply transformations
        transformed_df = select_and_rename_bldg_predictions_for_db(predictions_df)
        transformed_df = format_currency_values(transformed_df, ['amount_field'])
        transformed_df = format_date_values(transformed_df, ['PREDICTION_DT'])
        
        # Validate final result
        validation_result = validate_required_fields(
            transformed_df, 
            ['CLAIMNO', 'COVERAGE_TYPE', 'CONFIDENCE_SCORE']
        )
        
        assert validation_result['is_valid'] is True
        assert len(validation_result['missing_fields']) == 0