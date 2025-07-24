"""
Test suite for the main coverage pipeline.

This module contains comprehensive tests for the CoveragePipeline class
and its core functionality.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from datetime import datetime

# Import the modules to test
from modules.core.pipeline import CoveragePipeline
from modules.core.loader import ConfigLoader
from modules.core.monitor import PerformanceMonitor


class TestCoveragePipeline(unittest.TestCase):
    """Test cases for the CoveragePipeline class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_credentials = {
            'server': 'test_server',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass'
        }
        
        self.test_sql_queries = {
            'feature_query': 'SELECT * FROM test_table',
            'claims_query': 'SELECT * FROM claims'
        }
        
        self.test_rag_params = {
            'get_prompt': lambda: "Test prompt",
            'params_for_chunking': {'chunk_size': 1000},
            'rag_query': 'Test RAG query',
            'gpt_config_params': {'max_tokens': 4000}
        }
        
        self.mock_logger = Mock()
        self.test_sql_query_configs = {}
        
        # Create test dataframes
        self.test_feature_df = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002', 'CLM003'],
            'CLAIMKEY': ['KEY001', 'KEY002', 'KEY003'],
            'clean_FN_TEXT': ['Building damage claim text'] * 3,
            'LOBCD': ['15', '17', '15'],
            'LOSSDESC': ['Building damage', 'Structure damage', 'Roof damage']
        })
        
        self.test_summary_df = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002'],
            'summary': ['Building coverage summary', 'Structure coverage summary'],
            'confidence': [0.85, 0.92]
        })
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with valid parameters."""
        with patch('modules.core.pipeline.SourceLoader'), \\
             patch('modules.core.pipeline.ParallelRAGProcessor'), \\
             patch('modules.core.pipeline.MultiWriter'):
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS=self.test_sql_query_configs
            )
            
            self.assertIsNotNone(pipeline)
            self.assertEqual(pipeline.max_workers, 4)
            self.assertEqual(pipeline.credentials_dict, self.test_credentials)
            self.assertIsNone(pipeline.pre_hook_fn)
            self.assertIsNone(pipeline.post_hook_fn)
    
    def test_pipeline_initialization_with_hooks(self):
        """Test pipeline initialization with custom hooks."""
        # Create temporary hook files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as pre_hook:
            pre_hook.write('def pre_process(df): return df')
            pre_hook_path = pre_hook.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as post_hook:
            post_hook.write('def post_process(df): return df')
            post_hook_path = post_hook.name
        
        try:
            with patch('modules.core.pipeline.SourceLoader'), \\
                 patch('modules.core.pipeline.ParallelRAGProcessor'), \\
                 patch('modules.core.pipeline.MultiWriter'):
                
                pipeline = CoveragePipeline(
                    credentials_dict=self.test_credentials,
                    sql_queries=self.test_sql_queries,
                    rag_params=self.test_rag_params,
                    crypto_spark=None,
                    logger=self.mock_logger,
                    SQL_QUERY_CONFIGS=self.test_sql_query_configs,
                    pre_hook_path=pre_hook_path,
                    post_hook_path=post_hook_path
                )
                
                self.assertIsNotNone(pipeline.pre_hook_fn)
                self.assertIsNotNone(pipeline.post_hook_fn)
        
        finally:
            # Clean up temporary files
            os.unlink(pre_hook_path)
            os.unlink(post_hook_path)
    
    def test_load_hook_success(self):
        """Test successful hook loading."""
        # Create temporary hook file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as hook_file:
            hook_file.write('''
def test_function(data):
    """Test hook function."""
    return data
''')
            hook_path = hook_file.name
        
        try:
            with patch('modules.core.pipeline.SourceLoader'), \\
                 patch('modules.core.pipeline.ParallelRAGProcessor'), \\
                 patch('modules.core.pipeline.MultiWriter'):
                
                pipeline = CoveragePipeline(
                    credentials_dict=self.test_credentials,
                    sql_queries=self.test_sql_queries,
                    rag_params=self.test_rag_params,
                    crypto_spark=None,
                    logger=self.mock_logger,
                    SQL_QUERY_CONFIGS=self.test_sql_query_configs
                )
                
                hook_fn = pipeline.load_hook(hook_path, 'test_function')
                self.assertIsNotNone(hook_fn)
                self.assertTrue(callable(hook_fn))
        
        finally:
            os.unlink(hook_path)
    
    def test_load_hook_missing_function(self):
        """Test hook loading with missing function."""
        # Create temporary hook file without the expected function
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as hook_file:
            hook_file.write('def other_function(): pass')
            hook_path = hook_file.name
        
        try:
            with patch('modules.core.pipeline.SourceLoader'), \\
                 patch('modules.core.pipeline.ParallelRAGProcessor'), \\
                 patch('modules.core.pipeline.MultiWriter'):
                
                pipeline = CoveragePipeline(
                    credentials_dict=self.test_credentials,
                    sql_queries=self.test_sql_queries,
                    rag_params=self.test_rag_params,
                    crypto_spark=None,
                    logger=self.mock_logger,
                    SQL_QUERY_CONFIGS=self.test_sql_query_configs
                )
                
                hook_fn = pipeline.load_hook(hook_path, 'missing_function')
                self.assertIsNone(hook_fn)
        
        finally:
            os.unlink(hook_path)
    
    @patch('modules.core.pipeline.transforms')
    @patch('modules.core.pipeline.CoverageRules')
    def test_run_pipeline_success(self, mock_coverage_rules, mock_transforms):
        """Test successful pipeline execution."""
        # Mock the components
        mock_source_loader = Mock()
        mock_source_loader.load_data_parallel.return_value = self.test_feature_df
        
        mock_rag_processor = Mock()
        mock_rag_processor.process_claims.return_value = self.test_summary_df
        
        mock_storage_writer = Mock()
        
        # Mock the rules engine
        mock_rules_instance = Mock()
        mock_rules_instance.classify_rule_conditions.return_value = self.test_feature_df
        mock_coverage_rules.return_value = mock_rules_instance
        
        # Mock transforms
        mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.test_feature_df
        
        with patch('modules.core.pipeline.SourceLoader', return_value=mock_source_loader), \\
             patch('modules.core.pipeline.ParallelRAGProcessor', return_value=mock_rag_processor), \\
             patch('modules.core.pipeline.MultiWriter', return_value=mock_storage_writer):
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS=self.test_sql_query_configs
            )
            
            # Override mocked components
            pipeline.source_loader = mock_source_loader
            pipeline.rag_processor = mock_rag_processor
            pipeline.storage_writer = mock_storage_writer
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Verify calls
            mock_source_loader.load_data_parallel.assert_called_once()
            mock_rag_processor.process_claims.assert_called_once()
            mock_coverage_rules.assert_called_once()
            
            self.assertIsInstance(result, pd.DataFrame)
    
    def test_run_pipeline_empty_data(self):
        """Test pipeline execution with empty data."""
        mock_source_loader = Mock()
        mock_source_loader.load_data_parallel.return_value = pd.DataFrame()
        
        with patch('modules.core.pipeline.SourceLoader', return_value=mock_source_loader), \\
             patch('modules.core.pipeline.ParallelRAGProcessor'), \\
             patch('modules.core.pipeline.MultiWriter'):
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS=self.test_sql_query_configs
            )
            
            pipeline.source_loader = mock_source_loader
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            self.assertTrue(result.empty)
    
    def test_filter_claims_for_processing(self):
        """Test claim filtering logic."""
        with patch('modules.core.pipeline.SourceLoader'), \\
             patch('modules.core.pipeline.ParallelRAGProcessor'), \\
             patch('modules.core.pipeline.MultiWriter'):
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS=self.test_sql_query_configs
            )
            
            # Test data with various text lengths and LOB codes
            test_df = pd.DataFrame({
                'CLAIMNO': ['CLM001', 'CLM002', 'CLM003', 'CLM004'],
                'clean_FN_TEXT': ['Short', 'This is a long enough text for processing', 'Also long enough text', 'Short'],
                'LOBCD': ['15', '17', '18', '15']
            })
            
            filtered_df = pipeline._filter_claims_for_processing(test_df)
            
            # Should filter based on text length >= 100 and LOBCD in ['15', '17']
            expected_claims = ['CLM002']  # Only this one meets both criteria
            self.assertEqual(len(filtered_df), 1)
            self.assertIn('CLM002', filtered_df['CLAIMNO'].values)
    
    def test_get_pipeline_stats(self):
        """Test pipeline statistics retrieval."""
        with patch('modules.core.pipeline.SourceLoader'), \\
             patch('modules.core.pipeline.ParallelRAGProcessor'), \\
             patch('modules.core.pipeline.MultiWriter'):
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS=self.test_sql_query_configs
            )
            
            stats = pipeline.get_pipeline_stats()
            
            self.assertIn('max_workers', stats)
            self.assertIn('has_pre_hook', stats)
            self.assertIn('has_post_hook', stats)
            self.assertIn('components_initialized', stats)
            
            self.assertEqual(stats['max_workers'], 4)
            self.assertFalse(stats['has_pre_hook'])
            self.assertFalse(stats['has_post_hook'])


class TestConfigLoader(unittest.TestCase):
    """Test cases for the ConfigLoader class."""
    
    def test_load_config_with_defaults(self):
        """Test configuration loading with default settings."""
        base_config = {'existing_key': 'existing_value'}
        
        config = ConfigLoader.load_config(base_config)
        
        self.assertIn('existing_key', config)
        self.assertIn('pipeline', config)
        self.assertIn('parallel_processing', config['pipeline'])
        self.assertEqual(config['pipeline']['parallel_processing']['max_workers'], 4)
    
    def test_load_config_with_overrides(self):
        """Test configuration loading with override values."""
        base_config = {'existing_key': 'existing_value'}
        overrides = {
            'pipeline': {
                'parallel_processing': {
                    'max_workers': 8
                }
            }
        }
        
        config = ConfigLoader.load_config(base_config, overrides)
        
        self.assertEqual(config['pipeline']['parallel_processing']['max_workers'], 8)
    
    def test_merge_configs(self):
        """Test configuration merging logic."""
        base = {
            'level1': {
                'level2': {
                    'key1': 'value1',
                    'key2': 'value2'
                }
            }
        }
        
        overrides = {
            'level1': {
                'level2': {
                    'key2': 'new_value2',
                    'key3': 'value3'
                }
            }
        }
        
        result = ConfigLoader.merge_configs(base, overrides)
        
        self.assertEqual(result['level1']['level2']['key1'], 'value1')
        self.assertEqual(result['level1']['level2']['key2'], 'new_value2')
        self.assertEqual(result['level1']['level2']['key3'], 'value3')
    
    def test_validate_config(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            'pipeline': {
                'parallel_processing': {
                    'max_workers': 4
                },
                'source_loading': {
                    'enabled_sources': ['aip', 'atlas']
                },
                'hooks': {
                    'pre_processing_enabled': True,
                    'pre_hook_path': '/path/to/hook.py'
                }
            }
        }
        
        result = ConfigLoader.validate_config(valid_config)
        self.assertTrue(result['is_valid'])
        self.assertEqual(len(result['errors']), 0)
        
        # Invalid configuration
        invalid_config = {
            'pipeline': {
                'parallel_processing': {
                    'max_workers': -1  # Invalid
                },
                'source_loading': {
                    'enabled_sources': ['invalid_source']  # Invalid
                }
            }
        }
        
        result = ConfigLoader.validate_config(invalid_config)
        self.assertFalse(result['is_valid'])
        self.assertGreater(len(result['errors']), 0)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for the PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor()
    
    def test_operation_timing(self):
        """Test operation timing functionality."""
        import time
        
        self.monitor.start_operation('test_operation')
        time.sleep(0.01)  # Small delay
        duration = self.monitor.end_operation('test_operation')
        
        self.assertGreater(duration, 0)
        self.assertIn('test_operation', self.monitor.metrics)
        self.assertIsNotNone(self.monitor.metrics['test_operation'].duration)
    
    def test_custom_metrics(self):
        """Test custom metrics functionality."""
        self.monitor.add_custom_metric('test_metric', 42, 'test_category')
        
        self.assertIn('test_category', self.monitor.custom_metrics)
        self.assertIn('test_metric', self.monitor.custom_metrics['test_category'])
        self.assertEqual(self.monitor.custom_metrics['test_category']['test_metric']['value'], 42)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some test operations
        self.monitor.start_operation('op1')
        self.monitor.end_operation('op1')
        
        self.monitor.start_operation('op2')
        self.monitor.end_operation('op2')
        
        self.monitor.add_custom_metric('metric1', 100)
        
        summary = self.monitor.get_summary()
        
        self.assertEqual(summary['total_operations'], 2)
        self.assertGreater(summary['total_time'], 0)
        self.assertIn('op1', summary['operations'])
        self.assertIn('op2', summary['operations'])
        self.assertIn('general', summary['custom_metrics'])
    
    def test_active_operations(self):
        """Test active operations tracking."""
        self.monitor.start_operation('active_op')
        
        active_ops = self.monitor.get_active_operations()
        self.assertIn('active_op', active_ops)
        
        self.monitor.end_operation('active_op')
        
        active_ops = self.monitor.get_active_operations()
        self.assertNotIn('active_op', active_ops)
    
    def test_reset_monitor(self):
        """Test monitor reset functionality."""
        # Add some data
        self.monitor.start_operation('test_op')
        self.monitor.end_operation('test_op')
        self.monitor.add_custom_metric('test_metric', 42)
        
        # Verify data exists
        self.assertGreater(len(self.monitor.metrics), 0)
        self.assertGreater(len(self.monitor.custom_metrics), 0)
        
        # Reset
        self.monitor.reset()
        
        # Verify data is cleared
        self.assertEqual(len(self.monitor.metrics), 0)
        self.assertEqual(len(self.monitor.custom_metrics), 0)


if __name__ == '__main__':
    unittest.main()