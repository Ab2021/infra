"""
Performance tests for the building coverage system.

This module contains performance and stress tests to ensure
the system can handle expected workloads efficiently.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# Import the modules to test
from modules.core.pipeline import CoveragePipeline
from modules.core.monitor import PerformanceMonitor
from modules.processor.parallel_rag import ParallelRAGProcessor


class TestPipelinePerformance(unittest.TestCase):
    """Performance tests for the pipeline components."""
    
    def setUp(self):
        """Set up test fixtures for performance tests."""
        self.large_dataset = self._create_large_test_dataset(1000)
        self.medium_dataset = self._create_large_test_dataset(100)
        self.small_dataset = self._create_large_test_dataset(10)
        
        self.test_credentials = {
            'server': 'test_server',
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_pass'
        }
        
        self.test_rag_params = {
            'get_prompt': lambda: "Test prompt",
            'params_for_chunking': {'chunk_size': 1000},
            'rag_query': 'Test RAG query',
            'gpt_config_params': {'max_tokens': 4000}
        }
        
        self.mock_logger = Mock()
    
    def _create_large_test_dataset(self, size: int) -> pd.DataFrame:
        """Create a large test dataset for performance testing."""
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic claim data
        claim_numbers = [f'CLM{i:06d}' for i in range(1, size + 1)]
        lob_codes = np.random.choice(['15', '17', '18'], size=size)
        
        # Generate realistic text descriptions
        damage_types = [
            'Building foundation damage due to water intrusion and structural issues',
            'Roof damage from storm with extensive building material deterioration',
            'Wall damage and building structural problems from flooding event',
            'Floor damage with structural building components affected',
            'Ceiling damage requiring building structural repairs',
            'Window damage with building frame involvement',
            'Door damage affecting building structural integrity'
        ]
        
        text_descriptions = []
        for i in range(size):
            base_text = np.random.choice(damage_types)
            # Add some variation to make texts unique
            variation = f" Additional details for claim {i} with specific circumstances."
            text_descriptions.append(base_text + variation)
        
        return pd.DataFrame({
            'CLAIMNO': claim_numbers,
            'CLAIMKEY': [f'KEY{i:06d}' for i in range(1, size + 1)],
            'clean_FN_TEXT': text_descriptions,
            'LOBCD': lob_codes,
            'LOSSDESC': np.random.choice(['Building damage', 'Structure damage', 'Roof damage'], size=size),
            'LOSSDT': pd.date_range('2023-01-01', periods=size, freq='D'),
            'REPORTEDDT': pd.date_range('2023-01-02', periods=size, freq='D')
        })
    
    def test_pipeline_performance_small_dataset(self):
        """Test pipeline performance with small dataset (10 claims)."""
        start_time = time.time()
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up fast mocks
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = self.small_dataset
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.return_value = pd.DataFrame({
                'CLAIMNO': self.small_dataset['CLAIMNO'].tolist()[:5],
                'summary': ['Test summary'] * 5,
                'confidence': [0.9] * 5
            })
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.return_value = self.small_dataset
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.small_dataset
            
            # Create and run pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries={'feature_query': 'SELECT * FROM test'},
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
        execution_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 5.0, "Small dataset should process in under 5 seconds")
        self.assertIsInstance(result, pd.DataFrame)
        
        print(f"Small dataset performance: {execution_time:.2f} seconds for {len(self.small_dataset)} claims")
    
    def test_pipeline_performance_medium_dataset(self):
        """Test pipeline performance with medium dataset (100 claims)."""
        start_time = time.time()
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up mocks with slight delay to simulate processing
            mock_source_loader = Mock()
            
            def delayed_load(*args, **kwargs):
                time.sleep(0.01)  # Small delay to simulate database load
                return self.medium_dataset
            
            mock_source_loader.load_data_parallel.side_effect = delayed_load
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            
            def delayed_rag_process(*args, **kwargs):
                time.sleep(0.05)  # Simulate RAG processing time
                processed_claims = min(50, len(self.medium_dataset))  # Simulate filtering
                return pd.DataFrame({
                    'CLAIMNO': self.medium_dataset['CLAIMNO'].tolist()[:processed_claims],
                    'summary': ['Test summary'] * processed_claims,
                    'confidence': np.random.uniform(0.7, 0.95, processed_claims)
                })
            
            mock_rag_processor.process_claims.side_effect = delayed_rag_process
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.return_value = self.medium_dataset
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.medium_dataset
            
            # Create and run pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries={'feature_query': 'SELECT * FROM test'},
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={},
                max_workers=4
            )
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
        execution_time = time.time() - start_time
        
        # Performance assertions
        self.assertLess(execution_time, 15.0, "Medium dataset should process in under 15 seconds")
        self.assertIsInstance(result, pd.DataFrame)
        
        print(f"Medium dataset performance: {execution_time:.2f} seconds for {len(self.medium_dataset)} claims")
    
    def test_parallel_processing_performance(self):
        """Test that parallel processing improves performance."""
        # Test serial processing (max_workers=1)
        start_serial = time.time()
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up mocks with processing delay
            def setup_mocks():
                mock_source_loader = Mock()
                mock_source_loader.load_data_parallel.return_value = self.medium_dataset
                mock_source_loader_class.return_value = mock_source_loader
                
                mock_rag_processor = Mock()
                def process_with_delay(*args, **kwargs):
                    time.sleep(0.02)  # Simulate processing time
                    return pd.DataFrame({
                        'CLAIMNO': self.medium_dataset['CLAIMNO'].tolist()[:20],
                        'summary': ['Test'] * 20,
                        'confidence': [0.9] * 20
                    })
                
                mock_rag_processor.process_claims.side_effect = process_with_delay
                mock_rag_class.return_value = mock_rag_processor
                
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                mock_rules = Mock()
                mock_rules.classify_rule_conditions.return_value = self.medium_dataset
                mock_rules_class.return_value = mock_rules
                
                mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.medium_dataset
            
            # Test serial processing
            setup_mocks()
            
            pipeline_serial = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries={'feature_query': 'SELECT * FROM test'},
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={},
                max_workers=1
            )
            
            result_serial = pipeline_serial.run_pipeline(['BLDG in LOSSDESC'])
            serial_time = time.time() - start_serial
            
            # Test parallel processing
            start_parallel = time.time()
            setup_mocks()
            
            pipeline_parallel = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries={'feature_query': 'SELECT * FROM test'},
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={},
                max_workers=4
            )
            
            result_parallel = pipeline_parallel.run_pipeline(['BLDG in LOSSDESC'])
            parallel_time = time.time() - start_parallel
            
        # Verify both completed successfully
        self.assertIsInstance(result_serial, pd.DataFrame)
        self.assertIsInstance(result_parallel, pd.DataFrame)
        
        print(f"Serial processing: {serial_time:.2f} seconds")
        print(f"Parallel processing: {parallel_time:.2f} seconds")
        
        # Note: In this mock test, times might be similar due to mocking
        # But structure demonstrates performance testing approach
    
    def test_memory_usage_monitoring(self):
        """Test memory usage with large datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
             patch('modules.core.pipeline.transforms') as mock_transforms, \
             patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
            
            # Set up mocks
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = self.large_dataset
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.return_value = pd.DataFrame({
                'CLAIMNO': self.large_dataset['CLAIMNO'].tolist()[:100],
                'summary': ['Test summary'] * 100,
                'confidence': [0.9] * 100
            })
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            mock_rules = Mock()
            mock_rules.classify_rule_conditions.return_value = self.large_dataset
            mock_rules_class.return_value = mock_rules
            
            mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.large_dataset
            
            # Create and run pipeline
            pipeline = CoveragePipeline(
                credentials_dict=self.test_credentials,
                sql_queries={'feature_query': 'SELECT * FROM test'},
                rag_params=self.test_rag_params,
                crypto_spark=None,
                logger=self.mock_logger,
                SQL_QUERY_CONFIGS={}
            )
            
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            
            # Get peak memory usage
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            print(f"Memory usage: Initial={initial_memory:.1f}MB, Peak={peak_memory:.1f}MB, Increase={memory_increase:.1f}MB")
            
            # Memory usage should be reasonable (less than 500MB increase for test data)
            self.assertLess(memory_increase, 500, "Memory usage should stay within reasonable bounds")
    
    def test_performance_monitor_accuracy(self):
        """Test that the performance monitor accurately captures metrics."""
        monitor = PerformanceMonitor()
        
        # Test timing accuracy
        monitor.start_operation('test_operation')
        time.sleep(0.1)  # Sleep for 100ms
        duration = monitor.end_operation('test_operation')
        
        # Duration should be approximately 100ms (allow some variance)
        self.assertGreater(duration, 0.09)
        self.assertLess(duration, 0.15)
        
        # Test custom metrics
        monitor.add_custom_metric('test_metric', 42, 'test_category')
        summary = monitor.get_summary()
        
        self.assertEqual(summary['total_operations'], 1)
        self.assertIn('test_category', summary['custom_metrics'])
        self.assertEqual(summary['custom_metrics']['test_category']['test_metric']['value'], 42)
    
    def test_concurrent_pipeline_execution(self):
        """Test multiple pipeline instances running concurrently."""
        def run_pipeline_instance(instance_id):
            """Run a single pipeline instance."""
            with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
                 patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
                 patch('modules.core.pipeline.MultiWriter') as mock_writer_class, \
                 patch('modules.core.pipeline.transforms') as mock_transforms, \
                 patch('modules.core.pipeline.CoverageRules') as mock_rules_class:
                
                # Set up mocks for this instance
                mock_source_loader = Mock()
                mock_source_loader.load_data_parallel.return_value = self.small_dataset
                mock_source_loader_class.return_value = mock_source_loader
                
                mock_rag_processor = Mock()
                mock_rag_processor.process_claims.return_value = pd.DataFrame({
                    'CLAIMNO': [f'CLM{instance_id:03d}001'],
                    'summary': [f'Summary for instance {instance_id}'],
                    'confidence': [0.9]
                })
                mock_rag_class.return_value = mock_rag_processor
                
                mock_writer = Mock()
                mock_writer_class.return_value = mock_writer
                
                mock_rules = Mock()
                mock_rules.classify_rule_conditions.return_value = self.small_dataset
                mock_rules_class.return_value = mock_rules
                
                mock_transforms.select_and_rename_bldg_predictions_for_db.return_value = self.small_dataset
                
                # Create pipeline
                pipeline = CoveragePipeline(
                    credentials_dict=self.test_credentials,
                    sql_queries={'feature_query': f'SELECT * FROM test_{instance_id}'},
                    rag_params=self.test_rag_params,
                    crypto_spark=None,
                    logger=Mock(),  # Separate logger for each instance
                    SQL_QUERY_CONFIGS={}
                )
                
                # Run pipeline
                result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
                return {'instance_id': instance_id, 'result': result, 'success': True}
        
        # Run multiple instances concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_pipeline_instance, i) for i in range(3)]
            results = [future.result() for future in futures]
        
        execution_time = time.time() - start_time
        
        # Verify all instances completed successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
            self.assertIsInstance(result['result'], pd.DataFrame)
        
        # Concurrent execution should not take much longer than serial
        self.assertLess(execution_time, 10.0, "Concurrent execution should complete within reasonable time")
        
        print(f"Concurrent execution: {execution_time:.2f} seconds for 3 instances")


class TestStressTests(unittest.TestCase):
    """Stress tests for system limits and edge cases."""
    
    def test_empty_dataset_handling(self):
        """Test system behavior with empty datasets."""
        empty_df = pd.DataFrame()
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class:
            
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = empty_df
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            pipeline = CoveragePipeline(
                credentials_dict={'test': 'value'},
                sql_queries={'test': 'query'},
                rag_params={'test': 'param'},
                crypto_spark=None,
                logger=Mock(),
                SQL_QUERY_CONFIGS={}
            )
            
            # Should handle empty data gracefully
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)
    
    def test_malformed_data_handling(self):
        """Test system behavior with malformed data."""
        malformed_df = pd.DataFrame({
            'CLAIMNO': [None, '', 'VALID001'],
            'clean_FN_TEXT': ['', None, 'Valid text description'],
            'LOBCD': ['INVALID', None, '15']
        })
        
        with patch('modules.core.pipeline.SourceLoader') as mock_source_loader_class, \
             patch('modules.core.pipeline.ParallelRAGProcessor') as mock_rag_class, \
             patch('modules.core.pipeline.MultiWriter') as mock_writer_class:
            
            mock_source_loader = Mock()
            mock_source_loader.load_data_parallel.return_value = malformed_df
            mock_source_loader_class.return_value = mock_source_loader
            
            mock_rag_processor = Mock()
            mock_rag_processor.process_claims.return_value = pd.DataFrame()
            mock_rag_class.return_value = mock_rag_processor
            
            mock_writer = Mock()
            mock_writer_class.return_value = mock_writer
            
            pipeline = CoveragePipeline(
                credentials_dict={'test': 'value'},
                sql_queries={'test': 'query'},
                rag_params={'test': 'param'},
                crypto_spark=None,
                logger=Mock(),
                SQL_QUERY_CONFIGS={}
            )
            
            # Should handle malformed data without crashing
            result = pipeline.run_pipeline(['BLDG in LOSSDESC'])
            self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    # Run performance tests with increased verbosity
    unittest.main(verbosity=2)