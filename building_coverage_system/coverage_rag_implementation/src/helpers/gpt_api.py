"""
GPT API client for building coverage analysis.

This module provides a client for interacting with GPT models
for building coverage determination and text analysis.
"""

import requests
import time
import json
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime


class GPTAPIClient:
    """
    Client for GPT API interactions.
    
    This class handles communication with GPT models for building
    coverage analysis, including error handling and retry logic.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the GPT API client.
        
        Args:
            config (Dict[str, Any]): GPT API configuration
            logger (Optional[logging.Logger]): Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # API settings
        self.api_url = config.get('api_url', 'https://api.openai.com/v1/chat/completions')
        self.api_key = config.get('api_key', '')
        self.model = config.get('model', 'gpt-4')
        
        # Request settings
        self.timeout = config.get('timeout_seconds', 30)
        self.max_retries = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay_seconds', 2)
        
        # Rate limiting
        self.rate_limit_rpm = config.get('rate_limit_requests_per_minute', 60)
        self.last_request_time = 0
        self.request_count = 0
        self.request_times = []
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0
        }
        
        self.logger.info(f"GPTAPIClient initialized with model {self.model}")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from GPT for the given prompt.
        
        Args:
            prompt (str): Input prompt for GPT
            
        Returns:
            str: GPT response text
            
        Raises:
            Exception: If API call fails after all retries
        """
        self.logger.debug(f"Generating GPT response for prompt (length: {len(prompt)})")
        
        # Apply rate limiting
        self._apply_rate_limiting()
        
        # Prepare request
        request_data = self._prepare_request(prompt)
        
        # Execute request with retries
        for attempt in range(self.max_retries):
            try:
                response = self._make_api_request(request_data)
                
                # Update statistics
                self._update_stats_success(response)
                
                return self._extract_response_text(response)
                
            except Exception as e:
                self.logger.warning(f"GPT API request attempt {attempt + 1} failed: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    # Final attempt failed
                    self._update_stats_failure()
                    raise Exception(f"GPT API request failed after {self.max_retries} attempts: {str(e)}")
                
                # Wait before retry
                time.sleep(self.retry_delay * (attempt + 1))
    
    def _apply_rate_limiting(self):
        """
        Apply rate limiting to API requests.
        """
        current_time = time.time()
        
        # Clean old request times (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we need to wait
        if len(self.request_times) >= self.rate_limit_rpm:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Record this request time
        self.request_times.append(current_time)
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """
        Prepare the API request payload.
        
        Args:
            prompt (str): Input prompt
            
        Returns:
            Dict[str, Any]: Request payload
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert insurance claim analyst specializing in building coverage determination."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        request_data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.config.get('max_tokens', 4000),
            "temperature": self.config.get('temperature', 0.1),
            "top_p": self.config.get('top_p', 0.95),
            "frequency_penalty": self.config.get('frequency_penalty', 0.0),
            "presence_penalty": self.config.get('presence_penalty', 0.0)
        }
        
        # Add stop sequences if configured
        stop_sequences = self.config.get('stop_sequences', [])
        if stop_sequences:
            request_data['stop'] = stop_sequences
        
        return request_data
    
    def _make_api_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make the actual API request.
        
        Args:
            request_data (Dict[str, Any]): Request payload
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            Exception: If request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.logger.debug(f"Making API request to {self.api_url}")
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=request_data,
            timeout=self.timeout
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse JSON response
        response_data = response.json()
        
        # Check for API errors
        if 'error' in response_data:
            raise Exception(f"API error: {response_data['error']}")
        
        return response_data
    
    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """
        Extract the response text from the API response.
        
        Args:
            response (Dict[str, Any]): API response
            
        Returns:
            str: Response text
            
        Raises:
            Exception: If response format is invalid
        """
        try:
            choices = response.get('choices', [])
            if not choices:
                raise Exception("No choices in API response")
            
            message = choices[0].get('message', {})
            content = message.get('content', '')
            
            if not content:
                raise Exception("Empty content in API response")
            
            return content.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting response text: {str(e)}")
    
    def _update_stats_success(self, response: Dict[str, Any]):
        """
        Update statistics for successful request.
        
        Args:
            response (Dict[str, Any]): API response
        """
        self.stats['total_requests'] += 1
        self.stats['successful_requests'] += 1
        
        # Extract token usage
        usage = response.get('usage', {})
        total_tokens = usage.get('total_tokens', 0)
        self.stats['total_tokens_used'] += total_tokens
        
        # Estimate cost (approximate pricing)
        cost_per_1k_tokens = self._get_cost_per_1k_tokens()
        request_cost = (total_tokens / 1000) * cost_per_1k_tokens
        self.stats['total_cost'] += request_cost
        
        self.logger.debug(f"Request successful. Tokens used: {total_tokens}, Cost: ${request_cost:.4f}")
    
    def _update_stats_failure(self):
        """
        Update statistics for failed request.
        """
        self.stats['total_requests'] += 1
        self.stats['failed_requests'] += 1
    
    def _get_cost_per_1k_tokens(self) -> float:
        """
        Get estimated cost per 1000 tokens for the current model.
        
        Returns:
            float: Cost per 1000 tokens in USD
        """
        # Approximate pricing as of 2024 (input + output average)
        model_pricing = {
            'gpt-3.5-turbo': 0.002,
            'gpt-4': 0.03,
            'gpt-4-turbo': 0.01,
            'gpt-4-turbo-preview': 0.01,
            'gpt-4o': 0.005
        }
        
        return model_pricing.get(self.model, 0.01)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get API usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        stats = self.stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
        
        if stats['successful_requests'] > 0:
            stats['avg_tokens_per_request'] = stats['total_tokens_used'] / stats['successful_requests']
        else:
            stats['avg_tokens_per_request'] = 0.0
        
        stats['estimated_cost_usd'] = stats['total_cost']
        
        return stats
    
    def reset_statistics(self):
        """
        Reset usage statistics.
        """
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost': 0.0
        }
        
        self.logger.info("API statistics reset")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the API connection with a simple request.
        
        Returns:
            Dict[str, Any]: Test results
        """
        test_result = {
            'success': False,
            'response_time': 0.0,
            'error': None,
            'model': self.model
        }
        
        try:
            start_time = time.time()
            
            test_prompt = "Test prompt: What is building coverage in insurance?"
            response = self.generate_response(test_prompt)
            
            test_result['success'] = True
            test_result['response_time'] = time.time() - start_time
            test_result['response_length'] = len(response)
            
            self.logger.info(f"API connection test successful (response time: {test_result['response_time']:.2f}s)")
            
        except Exception as e:
            test_result['error'] = str(e)
            self.logger.error(f"API connection test failed: {str(e)}")
        
        return test_result
    
    def batch_generate(
        self, 
        prompts: List[str], 
        max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts (List[str]): List of prompts to process
            max_concurrent (int): Maximum concurrent requests
            
        Returns:
            List[Dict[str, Any]]: List of responses with metadata
        """
        self.logger.info(f"Starting batch generation for {len(prompts)} prompts")
        
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.debug(f"Processing prompt {i+1}/{len(prompts)}")
            
            try:
                start_time = time.time()
                response = self.generate_response(prompt)
                processing_time = time.time() - start_time
                
                results.append({
                    'index': i,
                    'success': True,
                    'response': response,
                    'processing_time': processing_time,
                    'error': None
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to process prompt {i+1}: {str(e)}")
                
                results.append({
                    'index': i,
                    'success': False,
                    'response': None,
                    'processing_time': 0.0,
                    'error': str(e)
                })
        
        successful_count = sum(1 for r in results if r['success'])
        self.logger.info(f"Batch generation completed: {successful_count}/{len(prompts)} successful")
        
        return results