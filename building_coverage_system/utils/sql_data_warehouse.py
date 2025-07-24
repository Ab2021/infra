"""
SQL data warehouse utilities for building coverage system.

This module provides data warehouse operations, batch processing,
and SQL optimization utilities for the original Codebase 1.
"""

import pandas as pd
import pyodbc
import sqlalchemy
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import time
import json
import threading
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class SQLDataWarehouse:
    """
    SQL data warehouse manager for large-scale data operations.
    
    This class provides comprehensive data warehouse functionality
    including batch processing, data partitioning, and performance
    optimization for building coverage analytics.
    """
    
    def __init__(
        self,
        credentials: Dict[str, Any],
        warehouse_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SQL data warehouse.
        
        Args:
            credentials (Dict[str, Any]): Database credentials
            warehouse_config (Optional[Dict[str, Any]]): Warehouse configuration
            logger (Optional[logging.Logger]): Logger instance
        """
        self.credentials = credentials
        self.warehouse_config = warehouse_config or self._get_default_warehouse_config()
        self.logger = logger or logging.getLogger(__name__)
        
        # Connection management
        self.engine = None
        self.connection_lock = threading.Lock()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_rows_processed': 0,
            'total_processing_time': 0.0,
            'batch_operations': 0,
            'partition_operations': 0
        }
        
        # Initialize connection
        self._initialize_connection()
        
        self.logger.info("SQLDataWarehouse initialized")
    
    def batch_process_claims(
        self,
        claims_df: pd.DataFrame,
        batch_size: int = 5000,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Process claims data in batches for optimal performance.
        
        Args:
            claims_df (pd.DataFrame): Claims data to process
            batch_size (int): Number of records per batch
            max_workers (int): Maximum number of parallel workers
            
        Returns:
            Dict[str, Any]: Processing results and statistics
        """
        self.logger.info(f"Starting batch processing of {len(claims_df)} claims")
        start_time = time.time()
        
        try:
            # Split data into batches
            batches = self._create_batches(claims_df, batch_size)
            
            # Process batches in parallel
            results = self._process_batches_parallel(batches, max_workers)
            
            # Aggregate results
            total_processed = sum(result['rows_processed'] for result in results)
            successful_batches = sum(1 for result in results if result['success'])
            
            processing_time = time.time() - start_time
            self._update_performance_stats(total_processed, processing_time, True)
            self.performance_stats['batch_operations'] += 1
            
            result_summary = {
                'total_claims': len(claims_df),
                'total_processed': total_processed,
                'successful_batches': successful_batches,
                'total_batches': len(batches),
                'processing_time': processing_time,
                'claims_per_second': total_processed / processing_time if processing_time > 0 else 0
            }
            
            self.logger.info(f"Batch processing completed: {total_processed} claims in {processing_time:.2f} seconds")
            
            return result_summary
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(0, processing_time, False)
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise
    
    def create_data_partitions(
        self,
        source_table: str,
        partition_column: str,
        partition_strategy: str = 'monthly',
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[str]:
        """
        Create data partitions for improved query performance.
        
        Args:
            source_table (str): Source table name
            partition_column (str): Column to partition on
            partition_strategy (str): Partitioning strategy ('monthly', 'yearly', 'quarterly')
            date_range (Optional[Tuple[datetime, datetime]]): Date range for partitioning
            
        Returns:
            List[str]: List of created partition table names
        """
        self.logger.info(f"Creating data partitions for {source_table}")
        
        try:
            # Get date range if not provided
            if not date_range:
                date_range = self._get_table_date_range(source_table, partition_column)
            
            # Generate partition periods
            partition_periods = self._generate_partition_periods(
                date_range[0], date_range[1], partition_strategy
            )
            
            # Create partition tables
            partition_tables = []
            
            for period in partition_periods:
                partition_table = self._create_partition_table(
                    source_table, partition_column, period, partition_strategy
                )
                partition_tables.append(partition_table)
            
            self.performance_stats['partition_operations'] += 1
            
            self.logger.info(f"Created {len(partition_tables)} partitions for {source_table}")
            
            return partition_tables
            
        except Exception as e:
            self.logger.error(f"Partition creation failed: {str(e)}")
            raise
    
    def optimize_warehouse_performance(
        self,
        tables: List[str],
        optimization_level: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Optimize warehouse performance through indexing and statistics updates.
        
        Args:
            tables (List[str]): List of table names to optimize
            optimization_level (str): Level of optimization ('light', 'standard', 'aggressive')
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        self.logger.info(f"Starting warehouse optimization for {len(tables)} tables")
        start_time = time.time()
        
        optimization_results = {
            'tables_optimized': 0,
            'indexes_created': 0,
            'statistics_updated': 0,
            'optimization_time': 0.0,
            'errors': []
        }
        
        try:
            for table in tables:
                try:
                    # Update table statistics
                    self._update_table_statistics(table)
                    optimization_results['statistics_updated'] += 1
                    
                    # Create/rebuild indexes based on optimization level
                    indexes_created = self._optimize_table_indexes(table, optimization_level)
                    optimization_results['indexes_created'] += indexes_created
                    
                    # Perform table maintenance
                    self._perform_table_maintenance(table, optimization_level)
                    
                    optimization_results['tables_optimized'] += 1
                    
                    self.logger.debug(f"Optimized table {table}")
                    
                except Exception as e:
                    error_msg = f"Failed to optimize table {table}: {str(e)}"
                    optimization_results['errors'].append(error_msg)
                    self.logger.warning(error_msg)
            
            optimization_results['optimization_time'] = time.time() - start_time
            
            self.logger.info(f"Warehouse optimization completed in {optimization_results['optimization_time']:.2f} seconds")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Warehouse optimization failed: {str(e)}")
            raise
    
    def execute_bulk_operations(
        self,
        operations: List[Dict[str, Any]],
        transaction_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Execute multiple SQL operations in bulk.
        
        Args:
            operations (List[Dict[str, Any]]): List of operations to execute
            transaction_mode (bool): Whether to use transactions
            
        Returns:
            Dict[str, Any]: Execution results
        """
        self.logger.info(f"Executing {len(operations)} bulk operations")
        start_time = time.time()
        
        results = {
            'total_operations': len(operations),
            'successful_operations': 0,
            'failed_operations': 0,
            'total_rows_affected': 0,
            'execution_time': 0.0,
            'errors': []
        }
        
        with self._get_connection() as conn:
            if transaction_mode:
                trans = conn.begin()
            
            try:
                for i, operation in enumerate(operations):
                    try:
                        result = self._execute_single_operation(conn, operation)
                        results['successful_operations'] += 1
                        results['total_rows_affected'] += result.get('rows_affected', 0)
                        
                    except Exception as e:
                        error_msg = f"Operation {i+1} failed: {str(e)}"
                        results['errors'].append(error_msg)
                        results['failed_operations'] += 1
                        
                        if transaction_mode:
                            self.logger.warning(f"Rolling back transaction due to error: {error_msg}")
                            trans.rollback()
                            raise
                
                if transaction_mode:
                    trans.commit()
                    
                results['execution_time'] = time.time() - start_time
                
                self.logger.info(f"Bulk operations completed: {results['successful_operations']}/{results['total_operations']} successful")
                
                return results
                
            except Exception as e:
                if transaction_mode and 'trans' in locals():
                    try:
                        trans.rollback()
                    except:
                        pass
                
                self.logger.error(f"Bulk operations failed: {str(e)}")
                raise
    
    def analyze_warehouse_usage(
        self,
        analysis_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze warehouse usage patterns and performance.
        
        Args:
            analysis_period_days (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Usage analysis results
        """
        self.logger.info(f"Analyzing warehouse usage for {analysis_period_days} days")
        
        try:
            analysis_results = {
                'analysis_period_days': analysis_period_days,
                'table_usage': self._analyze_table_usage(analysis_period_days),
                'query_performance': self._analyze_query_performance(analysis_period_days),
                'storage_usage': self._analyze_storage_usage(),
                'recommendations': []
            }
            
            # Generate optimization recommendations
            analysis_results['recommendations'] = self._generate_optimization_recommendations(
                analysis_results
            )
            
            self.logger.info("Warehouse usage analysis completed")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Warehouse usage analysis failed: {str(e)}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """
        Get a database connection with proper cleanup.
        
        Yields:
            Database connection
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        finally:
            if conn:
                conn.close()
    
    def _initialize_connection(self):
        """
        Initialize the database connection.
        """
        try:
            connection_string = self._build_connection_string()
            
            self.engine = sqlalchemy.create_engine(
                connection_string,
                pool_size=self.warehouse_config.get('pool_size', 10),
                max_overflow=self.warehouse_config.get('max_overflow', 20),
                pool_timeout=self.warehouse_config.get('pool_timeout', 30),
                pool_recycle=self.warehouse_config.get('pool_recycle', 3600),
                echo=self.warehouse_config.get('echo_sql', False)
            )
            
            self.logger.debug("Database connection initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {str(e)}")
            raise
    
    def _build_connection_string(self) -> str:
        """
        Build SQLAlchemy connection string.
        
        Returns:
            str: Connection string
        """
        server = self.credentials['server']
        database = self.credentials['database']
        username = self.credentials.get('username')
        password = self.credentials.get('password')
        driver = self.credentials.get('driver', 'ODBC Driver 17 for SQL Server')
        
        if username and password:
            conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
        else:
            conn_str = f"mssql+pyodbc://@{server}/{database}?driver={driver.replace(' ', '+')}&trusted_connection=yes"
        
        return conn_str
    
    def _create_batches(
        self,
        df: pd.DataFrame,
        batch_size: int
    ) -> List[pd.DataFrame]:
        """
        Split dataframe into batches.
        
        Args:
            df (pd.DataFrame): Source dataframe
            batch_size (int): Size of each batch
            
        Returns:
            List[pd.DataFrame]: List of batch dataframes
        """
        batches = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size].copy()
            batches.append(batch)
        
        return batches
    
    def _process_batches_parallel(
        self,
        batches: List[pd.DataFrame],
        max_workers: int
    ) -> List[Dict[str, Any]]:
        """
        Process batches in parallel.
        
        Args:
            batches (List[pd.DataFrame]): List of batch dataframes
            max_workers (int): Maximum number of workers
            
        Returns:
            List[Dict[str, Any]]: Results from each batch
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_single_batch, i, batch): i
                for i, batch in enumerate(batches)
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                
                try:
                    result = future.result()
                    result['batch_index'] = batch_index
                    results.append(result)
                    
                except Exception as e:
                    error_result = {
                        'batch_index': batch_index,
                        'success': False,
                        'rows_processed': 0,
                        'error': str(e)
                    }
                    results.append(error_result)
                    self.logger.warning(f"Batch {batch_index} processing failed: {str(e)}")
        
        # Sort results by batch index to maintain order
        results.sort(key=lambda x: x['batch_index'])
        
        return results
    
    def _process_single_batch(
        self,
        batch_index: int,
        batch_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Process a single batch of data.
        
        Args:
            batch_index (int): Index of the batch
            batch_df (pd.DataFrame): Batch data
            
        Returns:
            Dict[str, Any]: Processing result
        """
        self.logger.debug(f"Processing batch {batch_index} with {len(batch_df)} records")
        start_time = time.time()
        
        try:
            # Perform batch-specific processing
            processed_df = self._apply_batch_transformations(batch_df)
            
            # Store processed data (implement based on specific requirements)
            rows_processed = len(processed_df)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'rows_processed': rows_processed,
                'processing_time': processing_time,
                'error': None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            return {
                'success': False,
                'rows_processed': 0,
                'processing_time': processing_time,
                'error': str(e)
            }
    
    def _apply_batch_transformations(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to a batch of data.
        
        Args:
            batch_df (pd.DataFrame): Batch data
            
        Returns:
            pd.DataFrame: Transformed data
        """
        # Implement batch-specific transformations
        processed_df = batch_df.copy()
        
        # Add processing timestamp
        processed_df['batch_processed_at'] = datetime.now()
        
        # Apply any other transformations as needed
        
        return processed_df
    
    def _get_table_date_range(
        self,
        table_name: str,
        date_column: str
    ) -> Tuple[datetime, datetime]:
        """
        Get the date range for a table.
        
        Args:
            table_name (str): Table name
            date_column (str): Date column name
            
        Returns:
            Tuple[datetime, datetime]: Start and end dates
        """
        query = f"""
        SELECT 
            MIN({date_column}) as min_date,
            MAX({date_column}) as max_date
        FROM {table_name}
        WHERE {date_column} IS NOT NULL
        """
        
        with self._get_connection() as conn:
            result = conn.execute(sqlalchemy.text(query)).fetchone()
            
            return result.min_date, result.max_date
    
    def _generate_partition_periods(
        self,
        start_date: datetime,
        end_date: datetime,
        strategy: str
    ) -> List[Tuple[datetime, datetime]]:
        """
        Generate partition periods based on strategy.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            strategy (str): Partitioning strategy
            
        Returns:
            List[Tuple[datetime, datetime]]: List of period ranges
        """
        periods = []
        current_date = start_date
        
        while current_date <= end_date:
            if strategy == 'monthly':
                # Get the first day of next month
                if current_date.month == 12:
                    next_period = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_period = current_date.replace(month=current_date.month + 1, day=1)
                
                period_end = min(next_period - timedelta(days=1), end_date)
                
            elif strategy == 'quarterly':
                # Get the first day of next quarter
                quarter_start_months = [1, 4, 7, 10]
                current_quarter_start = max([m for m in quarter_start_months if m <= current_date.month])
                next_quarter_start = quarter_start_months[(quarter_start_months.index(current_quarter_start) + 1) % 4]
                
                if next_quarter_start == 1:
                    next_period = current_date.replace(year=current_date.year + 1, month=1, day=1)
                else:
                    next_period = current_date.replace(month=next_quarter_start, day=1)
                
                period_end = min(next_period - timedelta(days=1), end_date)
                
            elif strategy == 'yearly':
                # Get the first day of next year
                next_period = current_date.replace(year=current_date.year + 1, month=1, day=1)
                period_end = min(next_period - timedelta(days=1), end_date)
                
            else:
                raise ValueError(f"Unknown partitioning strategy: {strategy}")
            
            periods.append((current_date, period_end))
            current_date = next_period
        
        return periods
    
    def _create_partition_table(
        self,
        source_table: str,
        partition_column: str,
        period: Tuple[datetime, datetime],
        strategy: str
    ) -> str:
        """
        Create a partition table for a specific period.
        
        Args:
            source_table (str): Source table name
            partition_column (str): Partition column name
            period (Tuple[datetime, datetime]): Period range
            strategy (str): Partitioning strategy
            
        Returns:
            str: Partition table name
        """
        start_date, end_date = period
        
        # Generate partition table name
        if strategy == 'monthly':
            suffix = start_date.strftime('%Y%m')
        elif strategy == 'quarterly':
            quarter = (start_date.month - 1) // 3 + 1
            suffix = f"{start_date.year}Q{quarter}"
        elif strategy == 'yearly':
            suffix = str(start_date.year)
        else:
            suffix = start_date.strftime('%Y%m%d')
        
        partition_table = f"{source_table}_part_{suffix}"
        
        # Create partition table
        create_query = f"""
        CREATE TABLE {partition_table} AS
        SELECT * FROM {source_table}
        WHERE {partition_column} >= '{start_date.strftime('%Y-%m-%d')}'
        AND {partition_column} <= '{end_date.strftime('%Y-%m-%d')}'
        """
        
        with self._get_connection() as conn:
            conn.execute(sqlalchemy.text(create_query))
            conn.commit()
        
        self.logger.debug(f"Created partition table {partition_table}")
        
        return partition_table
    
    def _update_table_statistics(self, table_name: str):
        """
        Update table statistics for better query optimization.
        
        Args:
            table_name (str): Table name
        """
        update_stats_query = f"UPDATE STATISTICS {table_name}"
        
        with self._get_connection() as conn:
            conn.execute(sqlalchemy.text(update_stats_query))
            conn.commit()
    
    def _optimize_table_indexes(self, table_name: str, optimization_level: str) -> int:
        """
        Optimize table indexes based on usage patterns.
        
        Args:
            table_name (str): Table name
            optimization_level (str): Optimization level
            
        Returns:
            int: Number of indexes created/rebuilt
        """
        indexes_created = 0
        
        # Get table columns for index creation
        columns_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        
        with self._get_connection() as conn:
            columns_result = conn.execute(sqlalchemy.text(columns_query)).fetchall()
            
            # Create indexes based on optimization level
            if optimization_level in ['standard', 'aggressive']:
                # Create indexes on commonly queried columns
                common_index_columns = ['CLAIMNO', 'LOSSDT', 'LOBCD', 'STATUSCD']
                
                for column_name, data_type in columns_result:
                    if column_name in common_index_columns:
                        index_name = f"IX_{table_name}_{column_name}"
                        create_index_query = f"CREATE INDEX {index_name} ON {table_name} ({column_name})"
                        
                        try:
                            conn.execute(sqlalchemy.text(create_index_query))
                            indexes_created += 1
                        except Exception as e:
                            if "already exists" not in str(e):
                                self.logger.warning(f"Failed to create index {index_name}: {str(e)}")
            
            if optimization_level == 'aggressive':
                # Create composite indexes for better performance
                composite_indexes = [
                    ('LOBCD', 'LOSSDT'),
                    ('STATUSCD', 'LOSSDT'),
                    ('CLAIMNO', 'STATUSCD')
                ]
                
                for columns in composite_indexes:
                    if all(col in [c[0] for c in columns_result] for col in columns):
                        index_name = f"IX_{table_name}_{'_'.join(columns)}"
                        columns_str = ', '.join(columns)
                        create_index_query = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
                        
                        try:
                            conn.execute(sqlalchemy.text(create_index_query))
                            indexes_created += 1
                        except Exception as e:
                            if "already exists" not in str(e):
                                self.logger.warning(f"Failed to create composite index {index_name}: {str(e)}")
            
            conn.commit()
        
        return indexes_created
    
    def _perform_table_maintenance(self, table_name: str, optimization_level: str):
        """
        Perform table maintenance operations.
        
        Args:
            table_name (str): Table name
            optimization_level (str): Optimization level
        """
        with self._get_connection() as conn:
            if optimization_level in ['standard', 'aggressive']:
                # Rebuild indexes if aggressive optimization
                if optimization_level == 'aggressive':
                    rebuild_query = f"ALTER INDEX ALL ON {table_name} REBUILD"
                    try:
                        conn.execute(sqlalchemy.text(rebuild_query))
                    except Exception as e:
                        self.logger.warning(f"Failed to rebuild indexes for {table_name}: {str(e)}")
            
            conn.commit()
    
    def _execute_single_operation(
        self,
        conn,
        operation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single SQL operation.
        
        Args:
            conn: Database connection
            operation (Dict[str, Any]): Operation definition
            
        Returns:
            Dict[str, Any]: Operation result
        """
        query = operation.get('query', '')
        params = operation.get('params', {})
        operation_type = operation.get('type', 'SELECT')
        
        if not query:
            raise ValueError("No query specified in operation")
        
        result = conn.execute(sqlalchemy.text(query), params)
        
        rows_affected = result.rowcount if operation_type in ['INSERT', 'UPDATE', 'DELETE'] else 0
        
        return {
            'operation_type': operation_type,
            'rows_affected': rows_affected,
            'success': True
        }
    
    def _analyze_table_usage(self, days: int) -> List[Dict[str, Any]]:
        """
        Analyze table usage patterns.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            List[Dict[str, Any]]: Table usage statistics
        """
        # Implement table usage analysis based on your specific monitoring setup
        # This is a placeholder implementation
        return [
            {
                'table_name': 'claims_data',
                'query_count': 1500,
                'avg_query_time': 2.3,
                'total_reads': 50000,
                'total_writes': 500
            }
        ]
    
    def _analyze_query_performance(self, days: int) -> Dict[str, Any]:
        """
        Analyze query performance metrics.
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            Dict[str, Any]: Query performance statistics
        """
        # Implement query performance analysis
        return {
            'avg_query_time': 2.1,
            'slowest_queries': [],
            'most_frequent_queries': [],
            'cache_hit_ratio': 0.85
        }
    
    def _analyze_storage_usage(self) -> Dict[str, Any]:
        """
        Analyze storage usage patterns.
        
        Returns:
            Dict[str, Any]: Storage usage statistics
        """
        # Implement storage usage analysis
        return {
            'total_size_gb': 150.5,
            'data_size_gb': 120.3,
            'index_size_gb': 30.2,
            'growth_rate_gb_per_month': 5.2
        }
    
    def _generate_optimization_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate optimization recommendations based on analysis.
        
        Args:
            analysis_results (Dict[str, Any]): Analysis results
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Analyze storage growth
        storage_info = analysis_results.get('storage_usage', {})
        if storage_info.get('growth_rate_gb_per_month', 0) > 10:
            recommendations.append("Consider implementing data archiving strategy due to high growth rate")
        
        # Analyze query performance
        query_info = analysis_results.get('query_performance', {})
        if query_info.get('avg_query_time', 0) > 5:
            recommendations.append("Consider optimizing slow queries and adding more indexes")
        
        if query_info.get('cache_hit_ratio', 1) < 0.8:
            recommendations.append("Consider increasing database cache size to improve performance")
        
        return recommendations
    
    def _update_performance_stats(
        self,
        rows_processed: int,
        processing_time: float,
        success: bool
    ):
        """
        Update performance statistics.
        
        Args:
            rows_processed (int): Number of rows processed
            processing_time (float): Processing time in seconds
            success (bool): Whether operation was successful
        """
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_processing_time'] += processing_time
        
        if success:
            self.performance_stats['successful_operations'] += 1
            self.performance_stats['total_rows_processed'] += rows_processed
        else:
            self.performance_stats['failed_operations'] += 1
    
    def _get_default_warehouse_config(self) -> Dict[str, Any]:
        """
        Get default warehouse configuration.
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'echo_sql': False,
            'batch_size': 5000,
            'max_workers': 4,
            'optimization_level': 'standard'
        }
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get current performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations']
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['total_operations']
        else:
            stats['success_rate'] = 0.0
            stats['avg_processing_time'] = 0.0
        
        if stats['successful_operations'] > 0:
            stats['avg_rows_per_operation'] = stats['total_rows_processed'] / stats['successful_operations']
        else:
            stats['avg_rows_per_operation'] = 0.0
        
        return stats
    
    def close_connections(self):
        """
        Close all database connections.
        """
        if self.engine:
            try:
                self.engine.dispose()
                self.logger.debug("Database connections closed")
            except Exception as e:
                self.logger.warning(f"Error closing database connections: {str(e)}")


def create_sql_data_warehouse(
    credentials: Dict[str, Any],
    warehouse_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> SQLDataWarehouse:
    """
    Factory function to create a SQL data warehouse instance.
    
    Args:
        credentials (Dict[str, Any]): Database credentials
        warehouse_config (Optional[Dict[str, Any]]): Warehouse configuration
        logger (Optional[logging.Logger]): Logger instance
        
    Returns:
        SQLDataWarehouse: Configured data warehouse instance
    """
    return SQLDataWarehouse(
        credentials=credentials,
        warehouse_config=warehouse_config,
        logger=logger
    )


def optimize_table_performance(
    warehouse: SQLDataWarehouse,
    table_names: List[str],
    optimization_level: str = 'standard'
) -> Dict[str, Any]:
    """
    Quick function to optimize multiple tables.
    
    Args:
        warehouse (SQLDataWarehouse): Warehouse instance
        table_names (List[str]): List of table names
        optimization_level (str): Optimization level
        
    Returns:
        Dict[str, Any]: Optimization results
    """
    return warehouse.optimize_warehouse_performance(table_names, optimization_level)


def batch_process_data(
    warehouse: SQLDataWarehouse,
    data_df: pd.DataFrame,
    batch_size: int = 5000
) -> Dict[str, Any]:
    """
    Quick function to batch process data.
    
    Args:
        warehouse (SQLDataWarehouse): Warehouse instance
        data_df (pd.DataFrame): Data to process
        batch_size (int): Batch size
        
    Returns:
        Dict[str, Any]: Processing results
    """
    return warehouse.batch_process_claims(data_df, batch_size)