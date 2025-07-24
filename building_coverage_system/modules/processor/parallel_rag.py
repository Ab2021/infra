"""
Parallel RAG processing for building coverage claims.

This module provides the ParallelRAGProcessor class for processing claims
using multi-threaded RAG (Retrieval-Augmented Generation) capabilities,
similar to Codebase 2's parallel processing approach.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import threading

# Import original RAG components for backward compatibility
from coverage_rag_implementation.src.rag_predictor import RAGPredictor


class ParallelRAGProcessor:
    """
    Parallel RAG processing using multi-threading capabilities.
    
    This class extends the original RAGPredictor functionality with parallel
    processing capabilities similar to Codebase 2's ThreadPoolExecutor approach.
    It processes multiple claims concurrently while maintaining the original
    RAG logic and ensuring thread safety.
    
    Attributes:
        get_prompt: Function to retrieve prompt templates
        rag_params (Dict[str, Any]): RAG processing parameters
        max_workers (int): Maximum number of parallel workers
        logger: Logging instance for operations
        rag_predictor: Original RAGPredictor instance for compatibility
        processing_stats (Dict): Statistics tracking for processed claims
        _lock: Thread lock for concurrent access safety
    """
    
    def __init__(self, get_prompt, rag_params: Dict[str, Any], 
                 max_workers: int = 4, logger: Optional[logging.Logger] = None):
        """
        Initialize the parallel RAG processor.
        
        Args:
            get_prompt: Function to retrieve prompt templates
            rag_params (Dict[str, Any]): RAG processing parameters
            max_workers (int): Maximum number of parallel workers (default: 4)
            logger (Optional[logging.Logger]): Logger instance for operations
        """
        self.get_prompt = get_prompt
        self.rag_params = rag_params
        self.max_workers = max_workers
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Initialize original RAG predictor for compatibility
        self._init_rag_predictor()
        
        # Initialize processing statistics
        self.processing_stats = {
            'total_claims_processed': 0,
            'successful_summaries': 0,
            'failed_summaries': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _init_rag_predictor(self):
        """
        Initialize the original RAG predictor for backward compatibility.
        
        This method creates a RAGPredictor instance using the original
        Codebase 1 logic, ensuring compatibility with existing functionality.
        """
        try:
            self.rag_predictor = RAGPredictor(self.get_prompt, self.rag_params)
            self.logger.info("RAG predictor initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG predictor: {e}")
            self.rag_predictor = None
            raise
    
    def process_claims(self, claims_df: pd.DataFrame, 
                      parallel_threshold: int = 10) -> pd.DataFrame:
        """
        Process claims using parallel execution when beneficial.
        
        This method determines whether to use parallel or sequential processing
        based on the number of claims and processes them accordingly using the
        most appropriate approach.
        
        Args:
            claims_df (pd.DataFrame): DataFrame containing claims to process
            parallel_threshold (int): Minimum number of claims to trigger parallel processing
            
        Returns:
            pd.DataFrame: DataFrame containing processed summaries
        """
        if claims_df.empty:
            self.logger.warning("No claims provided for processing")
            return pd.DataFrame()
        
        unique_claims = claims_df['CLAIMNO'].unique()
        claim_count = len(unique_claims)
        
        self.logger.info(f"Processing {claim_count} unique claims")
        
        # Use sequential processing for small datasets or if parallel processing is disabled
        if claim_count < parallel_threshold or self.max_workers <= 1:
            self.logger.info("Using sequential processing")
            return self._process_claims_sequential(claims_df)
        else:
            self.logger.info(f"Using parallel processing with {self.max_workers} workers")
            return self._process_claims_parallel(claims_df, unique_claims)
    
    def _process_claims_sequential(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process claims sequentially using the original RAG predictor.
        
        Args:
            claims_df (pd.DataFrame): Claims dataframe to process
            
        Returns:
            pd.DataFrame: Processed summaries dataframe
        """
        try:
            start_time = time.time()
            
            # Use original RAGPredictor method for sequential processing
            result_df = self.rag_predictor.get_summary(claims_df)
            
            processing_time = time.time() - start_time
            self._update_processing_stats(len(claims_df['CLAIMNO'].unique()), 
                                        len(result_df) if not result_df.empty else 0,
                                        0, processing_time)
            
            self.logger.info(f"Sequential processing completed in {processing_time:.2f} seconds")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Sequential processing failed: {e}")
            return pd.DataFrame()
    
    def _process_claims_parallel(self, claims_df: pd.DataFrame, 
                               unique_claims: List[str]) -> pd.DataFrame:
        """
        Process claims in parallel using ThreadPoolExecutor.
        
        This method implements parallel processing similar to Codebase 2's approach,
        processing each claim in a separate thread for improved performance.
        
        Args:
            claims_df (pd.DataFrame): Claims dataframe to process
            unique_claims (List[str]): List of unique claim numbers
            
        Returns:
            pd.DataFrame: DataFrame containing processed summaries
        """
        start_time = time.time()
        results = []
        successful_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit each claim for parallel processing
            future_to_claim = {
                executor.submit(self._process_single_claim, claims_df, claimno): claimno
                for claimno in unique_claims
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_claim):
                claimno = future_to_claim[future]
                
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        successful_count += 1
                        self.logger.debug(f"Successfully processed claim: {claimno}")
                    else:
                        failed_count += 1
                        self.logger.warning(f"No result returned for claim: {claimno}")
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.error(f"Error processing claim {claimno}: {e}")
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(results) if results else pd.DataFrame()
        
        processing_time = time.time() - start_time
        self._update_processing_stats(len(unique_claims), successful_count, 
                                    failed_count, processing_time)
        
        self.logger.info(f"Parallel processing completed: {successful_count} successful, "
                        f"{failed_count} failed in {processing_time:.2f} seconds")
        
        return result_df
    
    def _process_single_claim(self, claims_df: pd.DataFrame, claimno: str) -> Optional[Dict]:
        """
        Process a single claim for parallel execution.
        
        This method processes an individual claim using the original RAG logic,
        designed to be thread-safe for parallel execution.
        
        Args:
            claims_df (pd.DataFrame): Full claims dataframe
            claimno (str): Claim number to process
            
        Returns:
            Optional[Dict]: Processing result dictionary or None if failed
        """
        try:
            # Filter data for specific claim
            claim_df = claims_df[claims_df['CLAIMNO'] == claimno]
            
            if claim_df.empty:
                self.logger.warning(f"No data found for claim: {claimno}")
                return None
            
            # Use original RAG processor logic for single claim
            result = self.rag_predictor.rg_processor.get_summary_and_loss_desc_b_code(
                claim_df,
                self.rag_params['params_for_chunking']['chunk_size'],
                self.rag_params['rag_query']
            )
            
            # Ensure result has required fields
            if isinstance(result, dict) and 'CLAIMNO' in result:
                return result
            else:
                self.logger.warning(f"Invalid result format for claim: {claimno}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in single claim processing for {claimno}: {e}")
            return None
    
    def _update_processing_stats(self, total_claims: int, successful: int, 
                               failed: int, processing_time: float):
        """
        Update processing statistics in a thread-safe manner.
        
        Args:
            total_claims (int): Total number of claims processed
            successful (int): Number of successfully processed claims
            failed (int): Number of failed claims
            processing_time (float): Total processing time in seconds
        """
        with self._lock:
            self.processing_stats['total_claims_processed'] += total_claims
            self.processing_stats['successful_summaries'] += successful
            self.processing_stats['failed_summaries'] += failed
            self.processing_stats['total_processing_time'] += processing_time
            
            # Calculate average processing time
            if self.processing_stats['total_claims_processed'] > 0:
                self.processing_stats['average_processing_time'] = (
                    self.processing_stats['total_processing_time'] / 
                    self.processing_stats['total_claims_processed']
                )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get current processing statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing processing statistics
        """
        with self._lock:
            stats = self.processing_stats.copy()
            
            # Add calculated metrics
            if stats['total_claims_processed'] > 0:
                stats['success_rate'] = (
                    stats['successful_summaries'] / stats['total_claims_processed']
                )
                stats['failure_rate'] = (
                    stats['failed_summaries'] / stats['total_claims_processed']
                )
                
                if stats['total_processing_time'] > 0:
                    stats['throughput_claims_per_second'] = (
                        stats['total_claims_processed'] / stats['total_processing_time']
                    )
            else:
                stats['success_rate'] = 0.0
                stats['failure_rate'] = 0.0
                stats['throughput_claims_per_second'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset processing statistics."""
        with self._lock:
            self.processing_stats = {
                'total_claims_processed': 0,
                'successful_summaries': 0,
                'failed_summaries': 0,
                'average_processing_time': 0.0,
                'total_processing_time': 0.0
            }
            self.logger.info("Processing statistics reset")
    
    def process_claims_batch(self, claims_df: pd.DataFrame, batch_size: int = 50) -> pd.DataFrame:
        """
        Process claims in batches for memory efficiency.
        
        This method processes large datasets in smaller batches to manage
        memory usage while still leveraging parallel processing.
        
        Args:
            claims_df (pd.DataFrame): Claims dataframe to process
            batch_size (int): Number of claims per batch (default: 50)
            
        Returns:
            pd.DataFrame: Combined results from all batches
        """
        if claims_df.empty:
            return pd.DataFrame()
        
        unique_claims = claims_df['CLAIMNO'].unique()
        total_claims = len(unique_claims)
        
        self.logger.info(f"Processing {total_claims} claims in batches of {batch_size}")
        
        all_results = []
        
        # Process claims in batches
        for i in range(0, total_claims, batch_size):
            batch_claims = unique_claims[i:i + batch_size]
            batch_df = claims_df[claims_df['CLAIMNO'].isin(batch_claims)]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}: "
                           f"claims {i+1}-{min(i+batch_size, total_claims)}")
            
            # Process batch
            batch_results = self.process_claims(batch_df)
            
            if not batch_results.empty:
                all_results.append(batch_results)
        
        # Combine all batch results
        if all_results:
            final_results = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"Batch processing completed: {len(final_results)} summaries generated")
            return final_results
        else:
            self.logger.warning("No results from batch processing")
            return pd.DataFrame()
    
    def get_optimal_worker_count(self, claim_count: int) -> int:
        """
        Calculate optimal worker count based on claim count and system resources.
        
        Args:
            claim_count (int): Number of claims to process
            
        Returns:
            int: Optimal number of workers
        """
        import os
        
        # Get available CPU cores
        cpu_cores = os.cpu_count() or 4
        
        # Calculate optimal workers based on various factors
        if claim_count < 10:
            optimal_workers = 1  # Sequential for small datasets
        elif claim_count < 100:
            optimal_workers = min(4, cpu_cores)  # Conservative for medium datasets
        else:
            optimal_workers = min(self.max_workers, cpu_cores)  # Full utilization for large datasets
        
        self.logger.debug(f"Optimal worker count for {claim_count} claims: {optimal_workers}")
        return optimal_workers
    
    def process_claims_adaptive(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process claims with adaptive worker allocation.
        
        This method automatically determines the optimal processing strategy
        based on dataset size and system resources.
        
        Args:
            claims_df (pd.DataFrame): Claims dataframe to process
            
        Returns:
            pd.DataFrame: Processed summaries dataframe
        """
        if claims_df.empty:
            return pd.DataFrame()
        
        claim_count = len(claims_df['CLAIMNO'].unique())
        optimal_workers = self.get_optimal_worker_count(claim_count)
        
        # Temporarily adjust worker count for this processing
        original_workers = self.max_workers
        self.max_workers = optimal_workers
        
        try:
            self.logger.info(f"Using adaptive processing with {optimal_workers} workers")
            result = self.process_claims(claims_df)
            return result
            
        finally:
            # Restore original worker count
            self.max_workers = original_workers
    
    def validate_processing_capability(self) -> Dict[str, Any]:
        """
        Validate the processing capability and configuration.
        
        Returns:
            Dict[str, Any]: Validation results and capability information
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'capabilities': {}
        }
        
        # Check RAG predictor initialization
        if self.rag_predictor is None:
            validation_results['is_valid'] = False
            validation_results['errors'].append("RAG predictor not initialized")
        
        # Check required parameters
        required_params = ['params_for_chunking', 'rag_query']
        for param in required_params:
            if param not in self.rag_params:
                validation_results['errors'].append(f"Missing required parameter: {param}")
                validation_results['is_valid'] = False
        
        # Check worker configuration
        if self.max_workers < 1:
            validation_results['errors'].append("max_workers must be at least 1")
            validation_results['is_valid'] = False
        elif self.max_workers > 16:
            validation_results['warnings'].append(
                f"High worker count ({self.max_workers}) may cause resource contention"
            )
        
        # Capability information
        validation_results['capabilities'] = {
            'max_workers': self.max_workers,
            'parallel_processing_available': self.max_workers > 1,
            'batch_processing_available': True,
            'adaptive_processing_available': True,
            'statistics_tracking': True
        }
        
        return validation_results
    
    def __str__(self) -> str:
        """String representation of the processor."""
        return f"ParallelRAGProcessor(max_workers={self.max_workers})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the processor."""
        stats = self.get_processing_stats()
        return (f"ParallelRAGProcessor(max_workers={self.max_workers}, "
                f"claims_processed={stats['total_claims_processed']}, "
                f"success_rate={stats['success_rate']:.2%})")