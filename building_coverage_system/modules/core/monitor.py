"""
Performance monitoring for the building coverage pipeline.

This module provides the PerformanceMonitor class for tracking execution times,
resource usage, and other performance metrics throughout the pipeline execution.
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading


@dataclass
class OperationMetric:
    """
    Data class for storing individual operation metrics.
    
    Attributes:
        name (str): Operation name
        start_time (float): Operation start timestamp
        end_time (Optional[float]): Operation end timestamp
        duration (Optional[float]): Operation duration in seconds
        metadata (Dict[str, Any]): Additional metadata for the operation
    """
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_completed(self) -> bool:
        """Check if the operation has completed."""
        return self.end_time is not None
    
    def get_duration(self) -> float:
        """Get operation duration, calculating if not already done."""
        if self.duration is not None:
            return self.duration
        elif self.end_time is not None:
            self.duration = self.end_time - self.start_time
            return self.duration
        else:
            return time.time() - self.start_time  # Current elapsed time


class PerformanceMonitor:
    """
    Simple performance monitoring for pipeline operations.
    
    This class provides functionality to monitor and track performance metrics
    throughout the pipeline execution, including operation timings, resource usage,
    and custom metrics.
    
    Attributes:
        metrics (Dict[str, OperationMetric]): Dictionary of operation metrics
        start_times (Dict[str, float]): Dictionary of operation start times
        custom_metrics (Dict[str, Any]): Dictionary of custom metrics
        logger: Logging instance for monitoring operations
        _lock: Thread lock for concurrent access safety
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the performance monitor.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance for monitoring operations
        """
        self.metrics: Dict[str, OperationMetric] = {}
        self.start_times: Dict[str, float] = {}
        self.custom_metrics: Dict[str, Any] = {}
        self.logger = logger if logger else logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._session_start = time.time()
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Start timing an operation.
        
        This method begins timing for a named operation and stores metadata
        if provided.
        
        Args:
            operation_name (str): Name of the operation to track
            metadata (Optional[Dict[str, Any]]): Additional metadata for the operation
        """
        with self._lock:
            current_time = time.time()
            self.start_times[operation_name] = current_time
            
            self.metrics[operation_name] = OperationMetric(
                name=operation_name,
                start_time=current_time,
                metadata=metadata or {}
            )
            
            self.logger.debug(f"Started timing operation: {operation_name}")
    
    def end_operation(self, operation_name: str, 
                     additional_metadata: Optional[Dict[str, Any]] = None) -> float:
        """
        End timing and record metric for an operation.
        
        Args:
            operation_name (str): Name of the operation to stop timing
            additional_metadata (Optional[Dict[str, Any]]): Additional metadata to add
            
        Returns:
            float: Duration of the operation in seconds
            
        Raises:
            KeyError: If operation was not started
        """
        with self._lock:
            if operation_name not in self.start_times:
                raise KeyError(f"Operation '{operation_name}' was not started")
            
            end_time = time.time()
            start_time = self.start_times[operation_name]
            duration = end_time - start_time
            
            # Update the metric
            if operation_name in self.metrics:
                self.metrics[operation_name].end_time = end_time
                self.metrics[operation_name].duration = duration
                
                if additional_metadata:
                    self.metrics[operation_name].metadata.update(additional_metadata)
            
            # Clean up start times
            del self.start_times[operation_name]
            
            self.logger.debug(f"Completed operation '{operation_name}' in {duration:.3f} seconds")
            
            return duration
    
    def add_custom_metric(self, metric_name: str, value: Any, 
                         category: str = 'general'):
        """
        Add a custom metric value.
        
        Args:
            metric_name (str): Name of the custom metric
            value (Any): Value of the metric
            category (str): Category for organizing metrics (default: 'general')
        """
        with self._lock:
            if category not in self.custom_metrics:
                self.custom_metrics[category] = {}
            
            self.custom_metrics[category][metric_name] = {
                'value': value,
                'timestamp': time.time()
            }
            
            self.logger.debug(f"Added custom metric {category}.{metric_name}: {value}")
    
    def get_operation_duration(self, operation_name: str) -> Optional[float]:
        """
        Get the duration of a specific operation.
        
        Args:
            operation_name (str): Name of the operation
            
        Returns:
            Optional[float]: Duration in seconds, or None if operation not found
        """
        if operation_name in self.metrics:
            return self.metrics[operation_name].get_duration()
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict[str, Any]: Dictionary containing performance statistics
        """
        with self._lock:
            completed_operations = [m for m in self.metrics.values() if m.is_completed()]
            
            if not completed_operations:
                return {
                    'total_operations': 0,
                    'total_time': 0,
                    'session_duration': time.time() - self._session_start,
                    'operations': {},
                    'custom_metrics': self.custom_metrics
                }
            
            total_time = sum(op.get_duration() for op in completed_operations)
            
            # Create operation summary
            operations_summary = {}
            for op in completed_operations:
                operations_summary[op.name] = {
                    'duration': op.get_duration(),
                    'start_time': op.start_time,
                    'end_time': op.end_time,
                    'metadata': op.metadata
                }
            
            return {
                'total_operations': len(completed_operations),
                'total_time': total_time,
                'average_time': total_time / len(completed_operations),
                'session_duration': time.time() - self._session_start,
                'operations': operations_summary,
                'custom_metrics': self.custom_metrics,
                'longest_operation': max(completed_operations, key=lambda x: x.get_duration()).name,
                'shortest_operation': min(completed_operations, key=lambda x: x.get_duration()).name
            }
    
    def get_active_operations(self) -> List[str]:
        """
        Get list of currently active (running) operations.
        
        Returns:
            List[str]: List of operation names that are currently running
        """
        with self._lock:
            return list(self.start_times.keys())
    
    def print_report(self, detailed: bool = False):
        """
        Print formatted performance report.
        
        Args:
            detailed (bool): Whether to include detailed operation information
        """
        summary = self.get_summary()
        
        print("\\n" + "=" * 50)
        print("PERFORMANCE REPORT")
        print("=" * 50)
        print(f"Session Duration: {summary['session_duration']:.2f} seconds")
        print(f"Total Operations: {summary['total_operations']}")
        
        if summary['total_operations'] > 0:
            print(f"Total Processing Time: {summary['total_time']:.2f} seconds")
            print(f"Average Operation Time: {summary['average_time']:.2f} seconds")
            print(f"Longest Operation: {summary['longest_operation']}")
            print(f"Shortest Operation: {summary['shortest_operation']}")
        
        # Active operations
        active_ops = self.get_active_operations()
        if active_ops:
            print(f"\\nActive Operations: {', '.join(active_ops)}")
        
        # Individual operation times
        if summary['operations'] and detailed:
            print("\\nOperation Details:")
            for op_name, op_data in summary['operations'].items():
                print(f"  {op_name}: {op_data['duration']:.3f}s")
                if op_data['metadata']:
                    for key, value in op_data['metadata'].items():
                        print(f"    {key}: {value}")
        elif summary['operations']:
            print("\\nOperation Times:")
            for op_name, op_data in summary['operations'].items():
                print(f"  {op_name}: {op_data['duration']:.3f}s")
        
        # Custom metrics
        if summary['custom_metrics']:
            print("\\nCustom Metrics:")
            for category, metrics in summary['custom_metrics'].items():
                print(f"  {category}:")
                for metric_name, metric_data in metrics.items():
                    print(f"    {metric_name}: {metric_data['value']}")
        
        print("=" * 50 + "\\n")
    
    def export_metrics(self, format_type: str = 'dict') -> Any:
        """
        Export metrics in specified format.
        
        Args:
            format_type (str): Export format ('dict', 'json', 'csv')
            
        Returns:
            Any: Exported metrics in requested format
            
        Raises:
            ValueError: If format_type is not supported
        """
        summary = self.get_summary()
        
        if format_type == 'dict':
            return summary
        elif format_type == 'json':
            import json
            return json.dumps(summary, indent=2, default=str)
        elif format_type == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Operation', 'Duration', 'Start Time', 'End Time'])
            
            # Write operation data
            for op_name, op_data in summary['operations'].items():
                writer.writerow([
                    op_name,
                    op_data['duration'],
                    op_data['start_time'],
                    op_data['end_time']
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def reset(self):
        """
        Reset all metrics and timers.
        
        This method clears all stored metrics and resets the monitor to
        its initial state.
        """
        with self._lock:
            self.metrics.clear()
            self.start_times.clear()
            self.custom_metrics.clear()
            self._session_start = time.time()
            
            self.logger.info("Performance monitor reset")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - print final report."""
        if exc_type is None:  # No exception occurred
            self.print_report()


class PipelinePerformanceMonitor(PerformanceMonitor):
    """
    Specialized performance monitor for building coverage pipeline.
    
    This class extends PerformanceMonitor with pipeline-specific metrics
    and reporting capabilities.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline performance monitor.
        
        Args:
            logger (Optional[logging.Logger]): Logger instance
        """
        super().__init__(logger)
        self.pipeline_metrics = {
            'claims_processed': 0,
            'summaries_generated': 0,
            'rules_applied': 0,
            'errors_encountered': 0,
            'data_sources_loaded': 0
        }
    
    def record_claims_processed(self, count: int):
        """Record number of claims processed."""
        self.pipeline_metrics['claims_processed'] += count
        self.add_custom_metric('claims_processed', count, 'pipeline')
    
    def record_summaries_generated(self, count: int):
        """Record number of summaries generated."""
        self.pipeline_metrics['summaries_generated'] += count
        self.add_custom_metric('summaries_generated', count, 'pipeline')
    
    def record_error(self, error_type: str = 'general'):
        """Record an error occurrence."""
        self.pipeline_metrics['errors_encountered'] += 1
        self.add_custom_metric(f'errors_{error_type}', 1, 'errors')
    
    def record_data_source_loaded(self, source_name: str, record_count: int):
        """Record data source loading."""
        self.pipeline_metrics['data_sources_loaded'] += 1
        self.add_custom_metric(f'records_from_{source_name}', record_count, 'data_sources')
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get pipeline-specific performance summary.
        
        Returns:
            Dict[str, Any]: Pipeline performance summary
        """
        base_summary = self.get_summary()
        
        # Calculate processing rates
        total_time = base_summary.get('total_time', 0)
        claims_processed = self.pipeline_metrics['claims_processed']
        
        processing_rate = claims_processed / total_time if total_time > 0 else 0
        
        pipeline_summary = {
            **base_summary,
            'pipeline_metrics': self.pipeline_metrics,
            'processing_rate_claims_per_second': processing_rate,
            'average_time_per_claim': total_time / claims_processed if claims_processed > 0 else 0
        }
        
        return pipeline_summary
    
    def print_pipeline_report(self):
        """Print pipeline-specific performance report."""
        summary = self.get_pipeline_summary()
        
        print("\\n" + "=" * 60)
        print("BUILDING COVERAGE PIPELINE PERFORMANCE REPORT")
        print("=" * 60)
        
        # Pipeline metrics
        print(f"Claims Processed: {summary['pipeline_metrics']['claims_processed']}")
        print(f"Summaries Generated: {summary['pipeline_metrics']['summaries_generated']}")
        print(f"Data Sources Loaded: {summary['pipeline_metrics']['data_sources_loaded']}")
        print(f"Errors Encountered: {summary['pipeline_metrics']['errors_encountered']}")
        
        # Performance metrics
        if summary['total_time'] > 0:
            print(f"\\nProcessing Rate: {summary['processing_rate_claims_per_second']:.2f} claims/second")
            print(f"Average Time per Claim: {summary['average_time_per_claim']:.3f} seconds")
        
        # Call parent report for detailed operation times
        print("\\n" + "-" * 40)
        print("DETAILED OPERATION TIMES")
        print("-" * 40)
        
        if summary['operations']:
            for op_name, op_data in summary['operations'].items():
                print(f"{op_name}: {op_data['duration']:.3f}s")
        
        print("=" * 60 + "\\n")