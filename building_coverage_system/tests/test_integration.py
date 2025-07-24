"""
Integration tests for the building coverage system.

This module contains end-to-end integration tests that verify
the complete pipeline functionality with real data flows.
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
from modules.source.source_loader import SourceLoader
from modules.processor.parallel_rag import ParallelRAGProcessor
from modules.storage.multi_writer import MultiWriter


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
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
        
        # Create comprehensive test data
        self.test_data = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
            'CLAIMKEY': ['KEY001', 'KEY002', 'KEY003', 'KEY004', 'KEY005'],
            'clean_FN_TEXT': [
                'Building foundation damage due to water intrusion and structural issues affecting the main structure',
                'Roof damage from storm with extensive building material deterioration requiring structural repair',
                'Wall damage and building structural problems from flooding event',
                'Minor landscaping damage with no building involvement',
                'Vehicle damage with no structural building components affected'
            ],
            'LOBCD': ['15', '17', '15', '18', '17'],
            'LOSSDESC': ['Building damage', 'Structure damage', 'Roof damage', 'Landscape', 'Auto'],
            'LOSSDT': pd.to_datetime(['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']),
            'REPORTEDDT': pd.to_datetime(['2023-01-16', '2023-02-21', '2023-03-11', '2023-04-06', '2023-05-13'])
        })
        
        self.mock_logger = Mock()
    
    def test_full_pipeline_execution(self):
        """Test complete pipeline execution from start to finish."""
        # Mock expected RAG responses
        mock_rag_responses = pd.DataFrame({
            'CLAIMNO': ['CLM001', 'CLM002', 'CLM003'],
            'summary': [
                'Building coverage recommended due to structural foundation damage',
                'Building coverage required for roof and structural repairs',
                'Building coverage applicable for wall and structural damage'
            ],
            'confidence': [0.92, 0.88, 0.85]
        })
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up mocks
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = self.test_data
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.return_value = mock_rag_responses
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.return_value = self.test_data.merge(
                mock_rag_responses, on='CLAIMNO', how='left'
            )
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.test_data.merge(
                mock_rag_responses, on='CLAIMNO', how='left'
            )
            
            # Create pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            # Execute pipeline
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Verify pipeline execution
            self.assertIsInstance(result, pd.DataFrame)
            self.assertGreater(len(result), 0)
            
            # Verify all components were called
            mock_source_loader.load_data_parallel.assert_called_once()
            mock_rag_processor.process_claims.assert_called_once()
            mock_rules.classify_rule_conditions.assert_called_once()
    
    def test_pipeline_with_hooks(self):
        """Test pipeline execution with pre and post processing hooks."""
        # Create temporary hook files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as pre_hook:
            pre_hook.write('''
def pre_process(df):
    """Test pre-processing hook."""
    df['preprocessing_applied'] = True
    return df
''')
            pre_hook_path = pre_hook.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as post_hook:
            post_hook.write('''
def post_process(df):
    """Test post-processing hook."""
    df['postprocessing_applied'] = True
    return df
''')
            post_hook_path = post_hook.name
        
        try:
            with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
                 patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
                 patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
                 patch('modules.core.pipeline.transforms') as mock_transforms, \
                 patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
                
                # Set up mocks
                mock_source_loader = Mock()
                mock_source_loader.load_data_parallel.return_value = self.test_data
                mock_source_loader_class.return_value = mock_source_loader
                
                mock_rag_processor = Mock()
                mock_rag_processor.process_claims.return_value = pd.DataFrame({
                    'CLAIMNO': ['CLM001'],
                    'summary': ['Test summary'],
                    'confidence': [0.9]
                })
                mock_rag_class.return_value = mock_rag_processor
                
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                mock_rules = Mock()
                mock_rules.classify_rule_conditions.return_value = self.test_data.iloc[:1]
                mock_rules_class.return_value = mock_rules
                
                mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.test_data.iloc[:1]
                
                # Create pipeline with hooks
                pipeline = CoveragePipeline(
                    credentials_dict=self.test_credentials,
                    sql_queries=self.test_sql_queries,
                    rag_params=self.test_rag_params,
                    crypto_spark=None,
                    logger=self.mock_logger,
                    SQL_QUERY_CONFIGS={},
                    pre_hook_path=pre_hook_path,
                    post_hook_path=post_hook_path
                )
                
                # Execute pipeline
                result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
                
                # Verify hooks were loaded
                self.assertIsNotNone(pipeline.pre_hook_fn)
                self.assertIsNotNone(pipeline.post_hook_fn)
                
                # Verify execution completed
                self.assertIsInstance(result, pd.DataFrame)
        
        finally:
            # Clean up temporary files
            os.unlink(pre_hook_path)
            os.unlink(post_hook_path)
    
    def test_data_flow_consistency(self):
        """Test that data flows consistently through the pipeline stages."""
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Track data through each stage
            stage_data = {}
            
            def capture_source_data(*args, **kwargs):
                stage_data['source'] = self.test_data.copy()
                return self.test_data
            
            def capture_rag_data(*args, **kwargs):
                input_df = args[0] if args else kwargs.get('claims_df')
                stage_data['rag_input'] = input_df.copy() if input_df is not None else None
                rag_output = pd.DataFrame({
                    'CLAIMNO': ['CLM001', 'CLM002'],
                    'summary': ['Summary 1', 'Summary 2'],
                    'confidence': [0.9, 0.8]
                })
                stage_data['rag_output'] = rag_output
                return rag_output
            
            def capture_rules_data(*args, **kwargs):
                input_df = args[0] if args else None
                stage_data['rules_input'] = input_df.copy() if input_df is not None else None
                output_df = input_df.copy() if input_df is not None else pd.DataFrame()
                stage_data['rules_output'] = output_df
                return output_df
            
            # Set up mocks with data capture
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.side_effect = capture_source_data
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.side_effect = capture_rag_data
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.side_effect = capture_rules_data
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = pd.DataFrame()
            
            # Create and run pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Verify data consistency across stages
            self.assertIn('source', stage_data)
            self.assertIn('rag_input', stage_data)
            self.assertIn('rag_output', stage_data)
            
            # Verify data transformations
            if stage_data.get('rag_input') is not None:
                # RAG input should be filtered version of source data
                self.assertLessEqual(len(stage_data['rag_input']), len(stage_data['source']))
            
            # Verify required columns are preserved
            required_columns = ['CLAIMNO']
            for col in required_columns:
                if stage_data.get('source') is not None:
                    self.assertIn(col, stage_data['source'].columns)
                if stage_data.get('rag_output') is not None:
                    self.assertIn(col, stage_data['rag_output'].columns)
    
    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class:
            
            # Test source loading error
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.side_effect = Exception("Source loading failed")
            mock_source_loader_class.return_value = mock_source_loader
            
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            # Pipeline should handle the error gracefully
            with self.assertRaises(Exception):
                pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Verify error was logged
            self.mock_logger.error.assert_called()
    
    def test_performance_monitoring_integration(self):
        """Test that performance monitoring works throughout the pipeline."""
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up mocks for successful execution
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = self.test_data
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.return_value = pd.DataFrame({
                'CLAIMNO': ['CLM001'],
                'summary': ['Test'],
                'confidence': [0.9]
            })
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.return_value = self.test_data.iloc[:1]
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.test_data.iloc[:1]
            
            # Create pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries=self.test_sql_queries,
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            # Execute pipeline
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Verify performance monitor was used
            self.assertIsNotNone(pipeline.performance_monitor)
            
            # Get performance summary
            stats = pipeline.get_pipeline_stats()
            self.assertIn('max_workers', stats)
            self.assertIn('components_initialized', stats)


class TestConfigurationIntegration(unittest.TestCase):
    """Integration tests for configuration management."""
    
    def test_config_loading_and_validation(self):
        """Test complete configuration loading and validation process."""
        # Test configuration
        base_config = {
            'database': {
                'server': 'test_server',
                'database': 'test_db'
            }
        }
        
        overrides = {
            'pipeline': {
                'parallel_processing': {
                    'max_workers': 8
                }
            }
        }
        
        # Load configuration
        config = ConfigLoader.load_config(base_config, overrides)
        
        # Validate configuration
        validation_result = ConfigLoader.validate_config(config)
        
        # Verify results
        self.assertTrue(validation_result['is_valid'])
        self.assertEqual(config['pipeline']['parallel_processing']['max_workers'], 8)
        self.assertIn('database', config)
    
    def test_config_with_pipeline_integration(self):
        """Test configuration integration with pipeline creation."""
        config = {
            'pipeline': {
                'parallel_processing': {
                    'max_workers': 6
                },
                'source_loading': {
                    'enabled_sources': ['aip', 'atlas']
                }
            }
        }
        
        # Validate configuration
        validation_result = ConfigLoader.validate_config(config)
        self.assertTrue(validation_result['is_valid'])
        
        # Use configuration with pipeline (mock creation)
        with patch('modules.core.pipeline.SourceLoader'), \
             patch('modules.core.pipeline.ParallelRAGProcessor'), \
             patch('modules.core.pipeline.MultiWriter'):
            
            pipeline = CoveragePipeline(
                credentials_dict={'test': 'value'},
                sql_queries={'test': 'query'},
                rag_params={'test': 'param'},
                crypto_spark=None,
                logger=Mock(),
                SQL_QUERY_CONFIGS={},
                max_workers=config['pipeline']['parallel_processing']['max_workers']
            )
            
            self.assertEqual(pipeline.max_workers, 6)


if __name__ == '__main__':
    unittest.main()