"""
Main pipeline orchestrator for building coverage processing.

This module provides the CoveragePipeline class which orchestrates the entire
building coverage match process with support for parallel processing, custom hooks,
and multi-source data loading.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
import pandas as pd
import time
import importlib.util
import logging

# Import original components for backward compatibility
from coverage_configs.src.environment import DatabricksEnv
from coverage_rules.src.coverage_rules import CoverageRules
from coverage_rules.src import transforms


class CoveragePipeline:
    """
    Main pipeline orchestrator with multi-threading and hooks support.
    
    This class combines the functionality of the original building coverage system
    with modern parallel processing capabilities and custom hook support similar
    to Codebase 2's architecture.
    
    Attributes:
        credentials_dict (Dict): Database credentials configuration
        sql_queries (Dict): SQL queries for data extraction
        rag_params (Dict): RAG processing parameters
        crypto_spark: Spark cryptography instance
        logger: Logging instance
        SQL_QUERY_CONFIGS (Dict): SQL query configuration parameters
        max_workers (int): Maximum number of parallel workers
        pre_hook_fn: Pre-processing hook function
        post_hook_fn: Post-processing hook function
        source_loader: Multi-source data loader instance
        rag_processor: Parallel RAG processor instance
        storage_writer: Multi-destination storage writer instance
    """
    
    def __init__(self, 
                 credentials_dict: Dict,
                 sql_queries: Dict, 
                 rag_params: Dict,
                 crypto_spark,
                 logger,
                 SQL_QUERY_CONFIGS: Dict,
                 pre_hook_path: Optional[str] = None,
                 post_hook_path: Optional[str] = None,
                 max_workers: int = 4):
        """
        Initialize the coverage pipeline.
        
        Args:
            credentials_dict (Dict): Database credentials configuration
            sql_queries (Dict): SQL queries for data extraction  
            rag_params (Dict): RAG processing parameters
            crypto_spark: Spark cryptography instance
            logger: Logging instance
            SQL_QUERY_CONFIGS (Dict): SQL query configuration parameters
            pre_hook_path (Optional[str]): Path to pre-processing hook file
            post_hook_path (Optional[str]): Path to post-processing hook file
            max_workers (int): Maximum number of parallel workers (default: 4)
        """
        self.credentials_dict = credentials_dict
        self.sql_queries = sql_queries
        self.rag_params = rag_params
        self.crypto_spark = crypto_spark
        self.logger = logger if logger else logging.getLogger(__name__)
        self.SQL_QUERY_CONFIGS = SQL_QUERY_CONFIGS
        self.max_workers = max_workers
        
        # Load custom hooks
        self.pre_hook_fn = self.load_hook(pre_hook_path, "pre_process") if pre_hook_path else None
        self.post_hook_fn = self.load_hook(post_hook_path, "post_process") if post_hook_path else None
        
        # Initialize pipeline components
        self.init_components()
    
    def load_hook(self, hook_path: str, function_name: str):
        """
        Load custom hook from file path.
        
        This method dynamically loads Python modules containing custom hook functions,
        similar to Codebase 2's hook loading mechanism.
        
        Args:
            hook_path (str): Path to the hook Python file
            function_name (str): Name of the function to load from the hook file
            
        Returns:
            callable or None: The hook function if successfully loaded, None otherwise
        """
        try:
            spec = importlib.util.spec_from_file_location("hook_module", hook_path)
            if spec is None or spec.loader is None:
                self.logger.warning(f"Could not load spec for hook {hook_path}")
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, function_name):
                self.logger.info(f"Successfully loaded hook function {function_name} from {hook_path}")
                return getattr(module, function_name)
            else:
                self.logger.warning(f"Hook {hook_path} missing {function_name} function")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load hook {hook_path}: {e}")
            return None
    
    def init_components(self):
        """
        Initialize pipeline components.
        
        This method initializes the source loader, RAG processor, and storage writer
        components required for the pipeline execution.
        """
        try:
            from modules.source.source_loader import SourceLoader
            from modules.processor.parallel_rag import ParallelRAGProcessor
            from modules.storage.multi_writer import MultiWriter
            
            self.source_loader = SourceLoader(
                self.credentials_dict, 
                self.sql_queries, 
                self.crypto_spark,
                self.logger,
                self.SQL_QUERY_CONFIGS
            )
            
            self.rag_processor = ParallelRAGProcessor(
                self.rag_params.get('get_prompt'),
                self.rag_params,
                self.max_workers
            )
            
            self.storage_writer = MultiWriter(self.credentials_dict, self.logger)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def run_pipeline(self, bldg_conditions: List[str]) -> pd.DataFrame:
        """
        Execute complete building coverage pipeline.
        
        This method orchestrates the entire building coverage processing pipeline
        including data loading, hook execution, RAG processing, rule application,
        and result transformation.
        
        Args:
            bldg_conditions (List[str]): Building coverage conditions/rules to apply
            
        Returns:
            pd.DataFrame: Final processed building coverage predictions
            
        Raises:
            Exception: If pipeline execution fails at any stage
        """
        self.logger.info("Starting building coverage match pipeline")
        start_time = time.time()
        
        try:
            # Step 1: Load data from multiple sources in parallel
            self.logger.info("Loading data from multiple sources")
            feature_df = self.source_loader.load_data_parallel(
                sources=['aip', 'atlas', 'snowflake'],
                max_workers=3
            )
            self.logger.info(f"Loaded {len(feature_df)} feature records")
            
            if feature_df.empty:
                self.logger.warning("No data loaded from sources")
                return pd.DataFrame()
            
            # Step 2: Apply pre-processing hook if available
            if self.pre_hook_fn:
                self.logger.info("Applying pre-processing hook")
                feature_df = self.pre_hook_fn(feature_df)
                self.logger.info(f"Pre-processing completed, {len(feature_df)} records remain")
            
            # Step 3: Filter claims for processing
            filtered_claims_df = self._filter_claims_for_processing(feature_df)
            self.logger.info(f"Filtered to {len(filtered_claims_df)} claims for processing")
            
            if filtered_claims_df.empty:
                self.logger.warning("No claims remaining after filtering")
                return pd.DataFrame()
            
            # Step 4: Process claims with parallel RAG
            self.logger.info("Starting parallel RAG processing")
            summary_df = self.rag_processor.process_claims(filtered_claims_df)
            self.logger.info(f"Generated {len(summary_df)} summaries")
            
            # Step 5: Apply business rules
            self.logger.info("Applying coverage rules")
            rule_predictions = self._apply_coverage_rules(feature_df, summary_df, bldg_conditions)
            
            # Step 6: Apply post-processing hook if available
            if self.post_hook_fn:
                self.logger.info("Applying post-processing hook")
                rule_predictions = self.post_hook_fn(rule_predictions)
                self.logger.info("Post-processing completed")
            
            # Step 7: Final transformations
            final_df = transforms.select_and_rename_bldg_predictions_for_db(rule_predictions)
            
            total_time = time.time() - start_time
            self.logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return final_df
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _filter_claims_for_processing(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter claims for RAG processing based on text length and LOB codes.
        
        Args:
            feature_df (pd.DataFrame): Feature dataframe to filter
            
        Returns:
            pd.DataFrame: Filtered claims dataframe
        """
        try:
            filtered_df = feature_df[
                (feature_df['clean_FN_TEXT'].str.len() >= 100) &
                (feature_df['LOBCD'].isin(['15', '17']))
            ]
            return filtered_df
        except KeyError as e:
            self.logger.error(f"Required column missing for filtering: {e}")
            return pd.DataFrame()
    
    def _apply_coverage_rules(self, feature_df: pd.DataFrame, summary_df: pd.DataFrame, 
                            bldg_conditions: List[str]) -> pd.DataFrame:
        """
        Apply building coverage rules to the processed data.
        
        Args:
            feature_df (pd.DataFrame): Original feature data
            summary_df (pd.DataFrame): RAG-processed summaries
            bldg_conditions (List[str]): Building coverage conditions to apply
            
        Returns:
            pd.DataFrame: Data with applied rules and predictions
        """
        try:
            # Merge feature data with summaries
            merged_df = pd.merge(feature_df, summary_df, on=['CLAIMNO'], how='right')
            
            # Apply coverage rules using original rules engine
            rules_engine = CoverageRules('BLDG INDICATOR', bldg_conditions)
            rule_predictions = rules_engine.classify_rule_conditions(merged_df)
            
            return rule_predictions
            
        except Exception as e:
            self.logger.error(f"Error applying coverage rules: {e}")
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline execution statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing pipeline statistics
        """
        return {
            'max_workers': self.max_workers,
            'has_pre_hook': self.pre_hook_fn is not None,
            'has_post_hook': self.post_hook_fn is not None,
            'components_initialized': all([
                hasattr(self, 'source_loader'),
                hasattr(self, 'rag_processor'),
                hasattr(self, 'storage_writer')
            ])
        }


def main():
    """
    Main entry point for command-line execution.
    
    This function provides a command-line interface for running the building
    coverage pipeline with default configurations.
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Building Coverage Match Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum parallel workers')
    parser.add_argument('--pre-hook', type=str, help='Pre-processing hook file path')
    parser.add_argument('--post-hook', type=str, help='Post-processing hook file path')
    
    args = parser.parse_args()
    
    try:
        # Initialize basic environment (would need actual configuration)
        databricks_dict = {}  # Load from config file in real implementation
        env = DatabricksEnv(databricks_dict)
        
        # Create and run pipeline
        pipeline = CoveragePipeline(
            credentials_dict=env.credentials_dict,
            sql_queries=env.sql_queries,
            rag_params=env.rag_params,
            crypto_spark=env.crypto_spark,
            logger=env.logger,
            SQL_QUERY_CONFIGS=env.SQL_QUERY_CONFIGS,
            pre_hook_path=args.pre_hook,
            post_hook_path=args.post_hook,
            max_workers=args.max_workers
        )
        
        bldg_conditions = [
            "BLDG in LOSSDESC",
            "BUILDING in LOSSDESC", 
            "STRUCTURE in LOSSDESC"
        ]
        
        results = pipeline.run_pipeline(bldg_conditions)
        print(f"Pipeline completed successfully. Processed {len(results)} claims.")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()