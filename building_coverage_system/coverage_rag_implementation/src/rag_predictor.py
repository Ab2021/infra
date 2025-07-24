"""
RAG predictor for building coverage analysis.

This module contains the original Codebase 1 RAG predictor implementation
that performs building coverage analysis using GPT models and text processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import time
from datetime import datetime
import json
import re

from .helpers.text_processing import preprocess_claim_text, extract_keywords
from .helpers.chunk_split_sentences import split_text_into_chunks
from .helpers.gpt_api import GPTAPIClient

logger = logging.getLogger(__name__)


class RAGPredictor:
    """
    RAG predictor for building coverage analysis.
    
    This class implements the original Codebase 1 RAG prediction logic
    with GPT-based analysis and rule-based post-processing.
    """
    
    def __init__(
        self, 
        gpt_config: Dict[str, Any],
        chunking_params: Dict[str, Any],
        prompt_template: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the RAG predictor.
        
        Args:
            gpt_config (Dict[str, Any]): GPT API configuration
            chunking_params (Dict[str, Any]): Text chunking parameters
            prompt_template (str): GPT prompt template
            logger (Optional[logging.Logger]): Logger instance
        """
        self.gpt_config = gpt_config
        self.chunking_params = chunking_params
        self.prompt_template = prompt_template
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize GPT API client
        self.gpt_client = GPTAPIClient(gpt_config, logger=self.logger)
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        self.logger.info("RAGPredictor initialized successfully")
    
    def predict_building_coverage(
        self, 
        claims_df: pd.DataFrame,
        batch_size: int = 20
    ) -> pd.DataFrame:
        """
        Predict building coverage for a batch of claims.
        
        Args:
            claims_df (pd.DataFrame): Claims data with text descriptions
            batch_size (int): Number of claims to process per batch
            
        Returns:
            pd.DataFrame: Claims with predictions and confidence scores
        """
        self.logger.info(f"Starting building coverage prediction for {len(claims_df)} claims")
        start_time = time.time()
        
        # Initialize results
        predictions = []
        
        # Process claims in batches
        for i in range(0, len(claims_df), batch_size):
            batch_df = claims_df.iloc[i:i+batch_size].copy()
            
            self.logger.debug(f"Processing batch {i//batch_size + 1}: claims {i+1}-{min(i+batch_size, len(claims_df))}")
            
            batch_predictions = self._process_batch(batch_df)
            predictions.extend(batch_predictions)
        
        # Create results dataframe
        results_df = self._create_results_dataframe(claims_df, predictions)
        
        # Update statistics
        total_time = time.time() - start_time
        self._update_statistics(len(claims_df), total_time, results_df)
        
        self.logger.info(f"Building coverage prediction completed in {total_time:.2f} seconds")
        
        return results_df
    
    def _process_batch(self, batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process a batch of claims for building coverage prediction.
        
        Args:
            batch_df (pd.DataFrame): Batch of claims to process
            
        Returns:
            List[Dict[str, Any]]: Batch prediction results
        """
        batch_predictions = []
        
        for _, claim in batch_df.iterrows():
            try:
                prediction = self._predict_single_claim(claim)
                batch_predictions.append(prediction)
                
            except Exception as e:
                self.logger.error(f"Error processing claim {claim.get('CLAIMNO', 'unknown')}: {str(e)}")
                
                # Create error prediction
                error_prediction = {
                    'CLAIMNO': claim.get('CLAIMNO'),
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'summary': f"Processing error: {str(e)}",
                    'error': str(e),
                    'processing_time': 0.0
                }
                batch_predictions.append(error_prediction)
        
        return batch_predictions
    
    def _predict_single_claim(self, claim: pd.Series) -> Dict[str, Any]:
        """
        Predict building coverage for a single claim.
        
        Args:
            claim (pd.Series): Individual claim data
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        start_time = time.time()
        
        # Extract and preprocess claim text
        claim_text = claim.get('clean_FN_TEXT', '')
        if not claim_text:
            raise ValueError("No claim text found")
        
        # Preprocess the text
        processed_text = preprocess_claim_text(claim_text)
        
        # Split into chunks if needed
        chunks = self._chunk_text_if_needed(processed_text)
        
        # Analyze with GPT
        gpt_analysis = self._analyze_with_gpt(chunks, claim)
        
        # Apply post-processing rules
        final_prediction = self._apply_post_processing_rules(gpt_analysis, claim)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        final_prediction['processing_time'] = processing_time
        
        return final_prediction
    
    def _chunk_text_if_needed(self, text: str) -> List[str]:
        """
        Split text into chunks if it exceeds the maximum size.
        
        Args:
            text (str): Text to potentially chunk
            
        Returns:
            List[str]: List of text chunks
        """
        max_chunk_size = self.chunking_params.get('chunk_size', 1500)
        
        if len(text) <= max_chunk_size:
            return [text]
        
        # Use the chunking helper
        chunks = split_text_into_chunks(text, self.chunking_params)
        
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _analyze_with_gpt(self, text_chunks: List[str], claim: pd.Series) -> Dict[str, Any]:
        """
        Analyze text chunks with GPT to determine building coverage.
        
        Args:
            text_chunks (List[str]): Text chunks to analyze
            claim (pd.Series): Original claim data
            
        Returns:
            Dict[str, Any]: GPT analysis results
        """
        # For multiple chunks, analyze each and combine results
        if len(text_chunks) == 1:
            return self._analyze_single_chunk(text_chunks[0], claim)
        else:
            return self._analyze_multiple_chunks(text_chunks, claim)
    
    def _analyze_single_chunk(self, text: str, claim: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single text chunk with GPT.
        
        Args:
            text (str): Text to analyze
            claim (pd.Series): Original claim data
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Format the prompt
        formatted_prompt = self.prompt_template.format(
            claim_text=text,
            claim_number=claim.get('CLAIMNO', 'unknown'),
            lob_code=claim.get('LOBCD', 'unknown'),
            loss_description=claim.get('LOSSDESC', '')
        )
        
        # Call GPT API
        gpt_response = self.gpt_client.generate_response(formatted_prompt)
        
        # Parse the response
        parsed_response = self._parse_gpt_response(gpt_response)
        
        return parsed_response
    
    def _analyze_multiple_chunks(self, text_chunks: List[str], claim: pd.Series) -> Dict[str, Any]:
        """
        Analyze multiple text chunks and combine results.
        
        Args:
            text_chunks (List[str]): Text chunks to analyze
            claim (pd.Series): Original claim data
            
        Returns:
            Dict[str, Any]: Combined analysis results
        """
        chunk_results = []
        
        for i, chunk in enumerate(text_chunks):
            self.logger.debug(f"Analyzing chunk {i+1}/{len(text_chunks)}")
            
            try:
                result = self._analyze_single_chunk(chunk, claim)
                chunk_results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to analyze chunk {i+1}: {str(e)}")
                continue
        
        if not chunk_results:
            raise ValueError("Failed to analyze any text chunks")
        
        # Combine results from multiple chunks
        combined_result = self._combine_chunk_results(chunk_results)
        
        return combined_result
    
    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine analysis results from multiple chunks.
        
        Args:
            chunk_results (List[Dict[str, Any]]): Results from individual chunks
            
        Returns:
            Dict[str, Any]: Combined results
        """
        # Extract predictions and confidences
        predictions = [r.get('prediction', 'UNCLEAR') for r in chunk_results]
        confidences = [r.get('confidence', 0.0) for r in chunk_results]
        summaries = [r.get('summary', '') for r in chunk_results]
        
        # Determine final prediction using majority vote
        building_coverage_count = predictions.count('BUILDING COVERAGE')
        no_coverage_count = predictions.count('NO BUILDING COVERAGE')
        
        if building_coverage_count > no_coverage_count:
            final_prediction = 'BUILDING COVERAGE'
        elif no_coverage_count > building_coverage_count:
            final_prediction = 'NO BUILDING COVERAGE'
        else:
            final_prediction = 'UNCLEAR'
        
        # Calculate weighted average confidence
        if confidences:
            final_confidence = np.mean(confidences)
        else:
            final_confidence = 0.0
        
        # Combine summaries
        final_summary = self._combine_summaries(summaries, final_prediction)
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'summary': final_summary,
            'chunk_count': len(chunk_results),
            'chunk_predictions': predictions,
            'chunk_confidences': confidences
        }
    
    def _combine_summaries(self, summaries: List[str], final_prediction: str) -> str:
        """
        Combine summaries from multiple chunks into a coherent summary.
        
        Args:
            summaries (List[str]): Individual chunk summaries
            final_prediction (str): Final prediction result
            
        Returns:
            str: Combined summary
        """
        if not summaries:
            return f"Analysis resulted in {final_prediction} determination."
        
        # Filter out empty summaries
        valid_summaries = [s for s in summaries if s and s.strip()]
        
        if len(valid_summaries) == 1:
            return valid_summaries[0]
        
        # Combine multiple summaries
        combined = f"Multi-section analysis indicates {final_prediction}. "
        
        # Add key points from each summary
        key_points = []
        for summary in valid_summaries:
            # Extract key phrases (simplified approach)
            if 'building' in summary.lower() or 'structure' in summary.lower():
                key_points.append(summary[:100] + "..." if len(summary) > 100 else summary)
        
        if key_points:
            combined += "Key findings: " + "; ".join(key_points[:3])
        
        return combined
    
    def _parse_gpt_response(self, gpt_response: str) -> Dict[str, Any]:
        """
        Parse GPT response to extract prediction, confidence, and summary.
        
        Args:
            gpt_response (str): Raw GPT response
            
        Returns:
            Dict[str, Any]: Parsed response data
        """
        result = {
            'prediction': 'UNCLEAR',
            'confidence': 0.0,
            'summary': gpt_response,
            'raw_response': gpt_response
        }
        
        try:
            # Extract determination
            determination_match = re.search(r'DETERMINATION:\s*([^\n]+)', gpt_response, re.IGNORECASE)
            if determination_match:
                determination = determination_match.group(1).strip()
                if 'BUILDING COVERAGE' in determination.upper():
                    result['prediction'] = 'BUILDING COVERAGE'
                elif 'NO BUILDING COVERAGE' in determination.upper():
                    result['prediction'] = 'NO BUILDING COVERAGE'
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', gpt_response, re.IGNORECASE)
            if confidence_match:
                confidence_value = float(confidence_match.group(1))
                result['confidence'] = max(0.0, min(1.0, confidence_value))
            
            # Extract summary
            summary_match = re.search(r'SUMMARY:\s*([^\n]+(?:\n[^\n:]+)*)', gpt_response, re.IGNORECASE)
            if summary_match:
                result['summary'] = summary_match.group(1).strip()
            
            # Extract key factors
            factors_match = re.search(r'KEY FACTORS:\s*([^\n]+(?:\n[^\n:]+)*)', gpt_response, re.IGNORECASE)
            if factors_match:
                result['key_factors'] = factors_match.group(1).strip()
            
        except Exception as e:
            self.logger.warning(f"Error parsing GPT response: {str(e)}")
            result['parse_error'] = str(e)
        
        return result
    
    def _apply_post_processing_rules(
        self, 
        gpt_analysis: Dict[str, Any], 
        claim: pd.Series
    ) -> Dict[str, Any]:
        """
        Apply post-processing rules to refine the prediction.
        
        Args:
            gpt_analysis (Dict[str, Any]): GPT analysis results
            claim (pd.Series): Original claim data
            
        Returns:
            Dict[str, Any]: Final prediction with post-processing applied
        """
        result = gpt_analysis.copy()
        result['CLAIMNO'] = claim.get('CLAIMNO')
        
        # Rule 1: LOB code validation
        lob_code = claim.get('LOBCD', '')
        if lob_code in ['18', '19']:  # Typically non-building LOBs
            if result['prediction'] == 'BUILDING COVERAGE' and result['confidence'] < 0.8:
                result['confidence'] *= 0.8  # Reduce confidence
                result['summary'] += " (Confidence adjusted for LOB code)"
        
        # Rule 2: Text length validation
        text_length = len(claim.get('clean_FN_TEXT', ''))
        if text_length < 100:
            result['confidence'] *= 0.7  # Reduce confidence for short text
            result['summary'] += " (Confidence reduced due to short description)"
        
        # Rule 3: Keyword-based validation
        claim_text = claim.get('clean_FN_TEXT', '').lower()
        building_keywords = extract_keywords(claim_text)
        
        if building_keywords['strong_building_indicators'] >= 2:
            if result['prediction'] == 'NO BUILDING COVERAGE':
                result['confidence'] *= 0.6  # Reduce confidence if strong indicators present
        
        # Rule 4: Confidence floor and ceiling
        result['confidence'] = max(0.1, min(0.95, result['confidence']))
        
        # Rule 5: Add metadata
        result['post_processing_applied'] = True
        result['lob_code'] = lob_code
        result['text_length'] = text_length
        result['building_keywords_count'] = building_keywords['total_score']
        
        return result
    
    def _create_results_dataframe(
        self, 
        original_df: pd.DataFrame, 
        predictions: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create results dataframe combining original data with predictions.
        
        Args:
            original_df (pd.DataFrame): Original claims data
            predictions (List[Dict[str, Any]]): Prediction results
            
        Returns:
            pd.DataFrame: Combined results dataframe
        """
        # Convert predictions to dataframe
        predictions_df = pd.DataFrame(predictions)
        
        # Merge with original data
        results_df = original_df.merge(
            predictions_df, 
            on='CLAIMNO', 
            how='left'
        )
        
        # Add processing metadata
        results_df['prediction_timestamp'] = datetime.now()
        results_df['model_version'] = self.gpt_config.get('model', 'unknown')
        results_df['processing_method'] = 'RAGPredictor'
        
        return results_df
    
    def _update_statistics(self, claims_count: int, total_time: float, results_df: pd.DataFrame):
        """
        Update processing statistics.
        
        Args:
            claims_count (int): Number of claims processed
            total_time (float): Total processing time
            results_df (pd.DataFrame): Results dataframe
        """
        self.stats['total_processed'] += claims_count
        self.stats['total_processing_time'] += total_time
        
        successful_predictions = len(results_df[results_df['prediction'] != 'ERROR'])
        self.stats['successful_predictions'] += successful_predictions
        self.stats['failed_predictions'] += claims_count - successful_predictions
        
        # Calculate average confidence
        valid_confidences = results_df[results_df['prediction'] != 'ERROR']['confidence']
        if not valid_confidences.empty:
            self.stats['average_confidence'] = valid_confidences.mean()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dict[str, Any]: Current processing statistics
        """
        stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_predictions'] / stats['total_processed']
            stats['average_processing_time'] = stats['total_processing_time'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['average_processing_time'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """
        Reset processing statistics.
        """
        self.stats = {
            'total_processed': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'total_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        self.logger.info("Processing statistics reset")


def create_rag_predictor(
    gpt_config: Dict[str, Any],
    chunking_params: Dict[str, Any],
    prompt_template: str,
    logger: Optional[logging.Logger] = None
) -> RAGPredictor:
    """
    Factory function to create a RAG predictor instance.
    
    Args:
        gpt_config (Dict[str, Any]): GPT configuration
        chunking_params (Dict[str, Any]): Chunking parameters
        prompt_template (str): Prompt template
        logger (Optional[logging.Logger]): Logger instance
        
    Returns:
        RAGPredictor: Configured RAG predictor instance
    """
    return RAGPredictor(
        gpt_config=gpt_config,
        chunking_params=chunking_params,
        prompt_template=prompt_template,
        logger=logger
    )