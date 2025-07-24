"""
Multi-destination data writer for building coverage system.

This module provides the MultiWriter class for writing processed data to
multiple destinations in parallel, including SQL warehouses, Snowflake,
and local storage systems.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
import logging
import time
import os
from datetime import datetime

# Import original utilities for backward compatibility
try:
    from utils.sql_data_warehouse import write_to_warehouse
except ImportError:
    write_to_warehouse = None

try:
    from utils.snowflake_writer import write_to_snowflake_db
except ImportError:
    write_to_snowflake_db = None


class MultiWriter:
    """
    Multi-destination data writer with parallel execution capabilities.
    
    This class provides functionality to write processed data to multiple
    destinations in parallel, including SQL Data Warehouse, Snowflake,
    and local file storage. It handles errors gracefully and provides
    detailed logging of write operations.
    
    Attributes:
        credentials_dict (Dict): Database credentials configuration
        logger: Logging instance for operations
        storage_handlers (Dict[str, Callable]): Mapping of destination names to handler functions
        default_timeout (int): Default timeout for write operations in seconds
        write_stats (Dict): Statistics tracking for write operations
    """
    
    def __init__(self, credentials_dict: Dict, logger: Optional[logging.Logger] = None,
                 default_timeout: int = 300):
        """
        Initialize the multi-destination data writer.
        
        Args:
            credentials_dict (Dict): Database credentials configuration
            logger (Optional[logging.Logger]): Logger instance for operations
            default_timeout (int): Default timeout for operations in seconds
        """
        self.credentials_dict = credentials_dict
        self.logger = logger if logger else logging.getLogger(__name__)
        self.default_timeout = default_timeout
        
        # Initialize storage handlers mapping
        self.storage_handlers = {
            'sql_warehouse': self.write_to_sql_warehouse,
            'snowflake': self.write_to_snowflake,
            'local': self.write_to_local,
            'parquet': self.write_to_parquet,
            'csv': self.write_to_csv
        }
        
        # Initialize write statistics
        self.write_stats = {
            'total_writes': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'total_records_written': 0,
            'destinations_used': set()
        }
    
    def save_data_parallel(self, data: pd.DataFrame, 
                          destinations: Optional[List[str]] = None,
                          max_workers: int = 3,
                          write_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Save data to multiple destinations in parallel.
        
        This method writes the provided data to multiple destinations using
        parallel processing for improved performance. Each destination is
        handled in a separate thread.
        
        Args:
            data (pd.DataFrame): Data to write
            destinations (Optional[List[str]]): List of destination names.
                                              Defaults to ['sql_warehouse']
            max_workers (int): Maximum number of parallel workers (default: 3)
            write_config (Optional[Dict[str, Any]]): Configuration for write operations
            
        Returns:
            Dict[str, Any]: Results of write operations for each destination
        """
        if destinations is None:
            destinations = ['sql_warehouse']
        
        if write_config is None:
            write_config = {}
        
        if data.empty:
            self.logger.warning("No data provided for writing")
            return {dest: "No data to write" for dest in destinations}
        
        self.logger.info(f"Writing {len(data)} records to {len(destinations)} destinations")
        start_time = time.time()
        
        results = {}
        write_times = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit write tasks for each destination
            future_to_dest = {
                executor.submit(self._write_to_destination_with_timing, 
                              data, dest, write_config): dest
                for dest in destinations
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_dest):
                dest = future_to_dest[future]
                
                try:
                    success, write_time, message = future.result()
                    write_times[dest] = write_time
                    
                    if success:
                        results[dest] = f"Success: {message}"
                        self.write_stats['successful_writes'] += 1
                        self.write_stats['total_records_written'] += len(data)
                        self.logger.info(f"Successfully wrote to {dest} in {write_time:.2f}s: {message}")
                    else:
                        results[dest] = f"Failed: {message}"
                        self.write_stats['failed_writes'] += 1
                        self.logger.error(f"Failed to write to {dest}: {message}")
                    
                    self.write_stats['destinations_used'].add(dest)
                    
                except Exception as e:
                    results[dest] = f"Error: {str(e)}"
                    write_times[dest] = 0
                    self.write_stats['failed_writes'] += 1
                    self.logger.error(f"Unexpected error writing to {dest}: {e}")
        
        # Update overall statistics
        self.write_stats['total_writes'] += len(destinations)
        
        total_time = time.time() - start_time
        self.logger.info(f"Parallel write completed in {total_time:.2f} seconds")
        
        # Add timing information to results
        results['_timing_info'] = {
            'total_time': total_time,
            'individual_times': write_times,
            'records_written': len(data)
        }
        
        return results
    
    def _write_to_destination_with_timing(self, data: pd.DataFrame, 
                                        destination: str,
                                        write_config: Dict[str, Any]) -> tuple:
        """
        Write data to a destination with timing information.
        
        Args:
            data (pd.DataFrame): Data to write
            destination (str): Destination name
            write_config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success, write_time, message)
        """
        start_time = time.time()
        
        try:
            success, message = self.write_to_destination(data, destination, write_config)
            write_time = time.time() - start_time
            return success, write_time, message
            
        except Exception as e:
            write_time = time.time() - start_time
            return False, write_time, str(e)
    
    def write_to_destination(self, data: pd.DataFrame, destination: str,
                            write_config: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Write data to a specific destination.
        
        Args:
            data (pd.DataFrame): Data to write
            destination (str): Destination name
            write_config (Optional[Dict[str, Any]]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        if destination not in self.storage_handlers:
            return False, f"Unsupported destination: {destination}"
        
        try:
            handler = self.storage_handlers[destination]
            return handler(data, write_config or {})
            
        except Exception as e:
            return False, f"Handler execution failed: {str(e)}"
    
    def write_to_sql_warehouse(self, data: pd.DataFrame, 
                              config: Dict[str, Any]) -> tuple:
        """
        Write data to SQL Data Warehouse.
        
        This method uses the original utilities for writing to SQL warehouse
        while providing enhanced error handling and configuration options.
        
        Args:
            data (pd.DataFrame): Data to write
            config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Configuration parameters
            table_name = config.get('table', 'building_coverage_predictions')
            schema = config.get('schema', 'dbo')
            write_mode = config.get('write_mode', 'append')
            
            self.logger.debug(f"Writing {len(data)} records to SQL warehouse table {schema}.{table_name}")
            
            if write_to_warehouse is None:
                return False, "SQL warehouse utility not available"
            
            # Use original utility function
            write_to_warehouse(
                data=data,
                credentials=self.credentials_dict,
                table_name=table_name,
                schema=schema,
                if_exists=write_mode
            )
            
            return True, f"Wrote {len(data)} records to {schema}.{table_name}"
            
        except Exception as e:
            self.logger.error(f"SQL warehouse write failed: {e}")
            return False, str(e)
    
    def write_to_snowflake(self, data: pd.DataFrame, 
                          config: Dict[str, Any]) -> tuple:
        """
        Write data to Snowflake database.
        
        Args:
            data (pd.DataFrame): Data to write
            config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Configuration parameters
            table_name = config.get('table', 'building_coverage_predictions')
            database = config.get('database', 'CLAIMS_DB')
            schema = config.get('schema', 'PREDICTIONS')
            write_mode = config.get('write_mode', 'append')
            
            self.logger.debug(f"Writing {len(data)} records to Snowflake table {database}.{schema}.{table_name}")
            
            if write_to_snowflake_db is None:
                return False, "Snowflake utility not available"
            
            # Use original utility function
            write_to_snowflake_db(
                data=data,
                credentials=self.credentials_dict,
                table_name=table_name,
                database=database,
                schema=schema,
                if_exists=write_mode
            )
            
            return True, f"Wrote {len(data)} records to {database}.{schema}.{table_name}"
            
        except Exception as e:
            self.logger.error(f"Snowflake write failed: {e}")
            return False, str(e)
    
    def write_to_local(self, data: pd.DataFrame, 
                      config: Dict[str, Any]) -> tuple:
        """
        Write data to local file system.
        
        Args:
            data (pd.DataFrame): Data to write
            config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            # Configuration parameters
            output_dir = config.get('output_dir', 'output')
            filename = config.get('filename', f'building_coverage_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            file_format = config.get('format', 'parquet')
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine file extension and write method
            if file_format.lower() == 'parquet':
                filepath = os.path.join(output_dir, f"{filename}.parquet")
                data.to_parquet(filepath, index=False)
            elif file_format.lower() == 'csv':
                filepath = os.path.join(output_dir, f"{filename}.csv")
                data.to_csv(filepath, index=False)
            elif file_format.lower() == 'excel':
                filepath = os.path.join(output_dir, f"{filename}.xlsx")
                data.to_excel(filepath, index=False)
            else:
                return False, f"Unsupported file format: {file_format}"
            
            self.logger.debug(f"Wrote {len(data)} records to {filepath}")
            return True, f"Wrote {len(data)} records to {filepath}"
            
        except Exception as e:
            self.logger.error(f"Local file write failed: {e}")
            return False, str(e)
    
    def write_to_parquet(self, data: pd.DataFrame, 
                        config: Dict[str, Any]) -> tuple:
        """
        Write data to Parquet format.
        
        Args:
            data (pd.DataFrame): Data to write
            config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        config['format'] = 'parquet'
        return self.write_to_local(data, config)
    
    def write_to_csv(self, data: pd.DataFrame, 
                    config: Dict[str, Any]) -> tuple:
        """
        Write data to CSV format.
        
        Args:
            data (pd.DataFrame): Data to write
            config (Dict[str, Any]): Write configuration
            
        Returns:
            tuple: (success: bool, message: str)
        """
        config['format'] = 'csv'
        return self.write_to_local(data, config)
    
    def validate_data_for_write(self, data: pd.DataFrame, 
                               destination: str) -> Dict[str, Any]:
        """
        Validate data before writing to destination.
        
        Args:
            data (pd.DataFrame): Data to validate
            destination (str): Target destination
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'data_info': {}
        }
        
        # Basic data validation
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Data is empty")
            return validation_results
        
        # Data info
        validation_results['data_info'] = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'columns': list(data.columns),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Check for required columns based on destination
        required_columns = self._get_required_columns_for_destination(destination)
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            validation_results['errors'].append(f"Missing required columns: {list(missing_columns)}")
            validation_results['is_valid'] = False
        
        # Check for null values in critical columns
        critical_columns = ['CLAIMNO', 'prediction'] if destination in ['sql_warehouse', 'snowflake'] else []
        for col in critical_columns:
            if col in data.columns and data[col].isnull().any():
                null_count = data[col].isnull().sum()
                validation_results['warnings'].append(f"Column '{col}' contains {null_count} null values")
        
        # Check data types
        if destination in ['sql_warehouse', 'snowflake']:
            type_issues = self._check_data_types_for_database(data)
            validation_results['warnings'].extend(type_issues)
        
        # Check data size for performance warnings
        if len(data) > 100000:
            validation_results['warnings'].append(f"Large dataset ({len(data)} rows) may impact write performance")
        
        return validation_results
    
    def _get_required_columns_for_destination(self, destination: str) -> List[str]:
        """
        Get required columns for a specific destination.
        
        Args:
            destination (str): Destination name
            
        Returns:
            List[str]: List of required column names
        """
        column_requirements = {
            'sql_warehouse': ['CLAIMNO', 'prediction', 'confidence'],
            'snowflake': ['CLAIMNO', 'prediction', 'confidence'],
            'local': [],
            'parquet': [],
            'csv': []
        }
        
        return column_requirements.get(destination, [])
    
    def _check_data_types_for_database(self, data: pd.DataFrame) -> List[str]:
        """
        Check data types for database compatibility.
        
        Args:
            data (pd.DataFrame): Data to check
            
        Returns:
            List[str]: List of type warnings
        """
        warnings = []
        
        for column, dtype in data.dtypes.items():
            if dtype == 'object':
                # Check for mixed types in object columns
                sample_values = data[column].dropna().head(100)
                if len(sample_values) > 0:
                    unique_types = set(type(val).__name__ for val in sample_values)
                    if len(unique_types) > 1:
                        warnings.append(f"Column '{column}' contains mixed types: {unique_types}")
        
        return warnings
    
    def get_write_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive write statistics.
        
        Returns:
            Dict[str, Any]: Write operation statistics
        """
        stats = self.write_stats.copy()
        
        # Convert set to list for JSON serialization
        stats['destinations_used'] = list(stats['destinations_used'])
        
        # Calculate additional metrics
        if stats['total_writes'] > 0:
            stats['success_rate'] = stats['successful_writes'] / stats['total_writes']
            stats['failure_rate'] = stats['failed_writes'] / stats['total_writes']
        else:
            stats['success_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset write statistics."""
        self.write_stats = {
            'total_writes': 0,
            'successful_writes': 0,
            'failed_writes': 0,
            'total_records_written': 0,
            'destinations_used': set()
        }
        self.logger.info("Write statistics reset")
    
    def get_available_destinations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available storage destinations.
        
        Returns:
            Dict[str, Dict[str, Any]]: Information about each destination
        """
        destinations_info = {}
        
        for dest_name, handler in self.storage_handlers.items():
            destinations_info[dest_name] = {
                'name': dest_name,
                'handler_function': handler.__name__,
                'description': self._get_destination_description(dest_name),
                'supports_parallel': True,
                'required_credentials': self._get_required_credentials(dest_name)
            }
        
        return destinations_info
    
    def _get_destination_description(self, destination: str) -> str:
        """
        Get description for a storage destination.
        
        Args:
            destination (str): Destination name
            
        Returns:
            str: Description of the destination
        """
        descriptions = {
            'sql_warehouse': 'SQL Data Warehouse - Primary database storage',
            'snowflake': 'Snowflake Database - Cloud data warehouse',
            'local': 'Local File System - Files stored locally',
            'parquet': 'Parquet Files - Columnar storage format',
            'csv': 'CSV Files - Comma-separated values format'
        }
        return descriptions.get(destination, f'Unknown destination: {destination}')
    
    def _get_required_credentials(self, destination: str) -> List[str]:
        """
        Get required credentials for a destination.
        
        Args:
            destination (str): Destination name
            
        Returns:
            List[str]: List of required credential keys
        """
        credential_requirements = {
            'sql_warehouse': ['server', 'database', 'username', 'password'],
            'snowflake': ['account', 'user', 'password', 'warehouse', 'database'],
            'local': [],
            'parquet': [],
            'csv': []
        }
        
        return credential_requirements.get(destination, [])
    
    def test_destination_connectivity(self, destination: str) -> Dict[str, Any]:
        """
        Test connectivity to a specific destination.
        
        Args:
            destination (str): Destination to test
            
        Returns:
            Dict[str, Any]: Test results
        """
        test_result = {
            'destination': destination,
            'status': 'unknown',
            'response_time': 0.0,
            'error': None,
            'timestamp': time.time()
        }
        
        if destination not in self.storage_handlers:
            test_result['status'] = 'failed'
            test_result['error'] = f'Unsupported destination: {destination}'
            return test_result
        
        start_time = time.time()
        
        try:
            # Create small test dataframe
            test_data = pd.DataFrame({
                'CLAIMNO': ['TEST001'],
                'prediction': ['Test prediction'],
                'confidence': [0.95],
                'test_timestamp': [datetime.now()]
            })
            
            # Test write with dry-run configuration
            test_config = {
                'table': 'test_connectivity',
                'output_dir': 'temp',
                'filename': 'connectivity_test'
            }
            
            success, message = self.write_to_destination(test_data, destination, test_config)
            response_time = time.time() - start_time
            
            test_result['response_time'] = response_time
            
            if success:
                test_result['status'] = 'success'
                test_result['message'] = message
            else:
                test_result['status'] = 'failed'
                test_result['error'] = message
                
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            test_result['response_time'] = time.time() - start_time
        
        return test_result