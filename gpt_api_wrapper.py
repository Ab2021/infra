"""
GPT-4o-mini API Wrapper for Agentic Building Coverage Analysis
Replaces Anthropic calls with OpenAI GPT-4o-mini implementation
"""

import openai
import json
import os
import re
import logging
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime


class GptApiWrapper:
    """OpenAI GPT-4o-mini API wrapper replacing original GptApi"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client with GPT-4o-mini model"""
        # Validate API key before initialization
        validated_key = self._validate_api_key(api_key or os.getenv("OPENAI_API_KEY"))
        
        self.client = openai.OpenAI(api_key=validated_key)
        self.model = "gpt-4o-mini"
        self.default_temperature = 0.1
        self.default_max_tokens = 2000
        self.logger = logging.getLogger(__name__)
        
    def _validate_api_key(self, api_key: str) -> str:
        """Validate OpenAI API key format and security"""
        if not api_key or not isinstance(api_key, str):
            raise ValueError("OpenAI API key is required")
        
        # Basic format validation for OpenAI keys
        if not re.match(r'^sk-[A-Za-z0-9]{48}$', api_key.strip()):
            raise ValueError("Invalid OpenAI API key format")
        
        # Check for placeholder keys
        placeholder_patterns = [
            'your-api-key', 'test-key', 'demo-key', 'placeholder',
            'sk-0000', 'sk-1111', 'sk-xxxx', 'fake-key'
        ]
        
        if any(pattern in api_key.lower() for pattern in placeholder_patterns):
            raise ValueError("Placeholder API key is not valid")
        
        return api_key.strip()
    
    def _validate_input_parameters(self, prompt: str, temperature: float = None, 
                                 max_tokens: int = None, system_message: str = None) -> Dict[str, Any]:
        """Validate and sanitize input parameters"""
        errors = []
        
        # Validate prompt
        if not prompt or not isinstance(prompt, str):
            errors.append("Prompt must be a non-empty string")
        elif len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty or whitespace only")
        elif len(prompt) > 50000:  # Reasonable limit
            errors.append("Prompt exceeds maximum length (50,000 characters)")
        
        # Validate temperature
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append("Temperature must be a number")
            elif not (0.0 <= temperature <= 2.0):
                errors.append("Temperature must be between 0.0 and 2.0")
        
        # Validate max_tokens
        if max_tokens is not None:
            if not isinstance(max_tokens, int):
                errors.append("max_tokens must be an integer")
            elif max_tokens <= 0 or max_tokens > 4096:
                errors.append("max_tokens must be between 1 and 4096")
        
        # Validate system_message
        if system_message is not None:
            if not isinstance(system_message, str):
                errors.append("system_message must be a string")
            elif len(system_message) > 5000:
                errors.append("system_message exceeds maximum length (5,000 characters)")
        
        if errors:
            raise ValueError("; ".join(errors))
        
        # Sanitize inputs
        sanitized_prompt = self._sanitize_text_input(prompt)
        sanitized_system = self._sanitize_text_input(system_message) if system_message else None
        
        return {
            'prompt': sanitized_prompt,
            'temperature': temperature or self.default_temperature,
            'max_tokens': max_tokens or self.default_max_tokens,
            'system_message': sanitized_system
        }
    
    def _sanitize_text_input(self, text: str) -> str:
        """Sanitize text input to prevent injection attacks"""
        if not text:
            return ""
        
        # Remove potential injection patterns
        dangerous_patterns = [
            r'ignore\s+previous\s+instructions',
            r'system\s*:',
            r'assistant\s*:',
            r'user\s*:',
            r'###\s*instruction',
            r'```.*?```',
            r'\[INST\].*?\[/INST\]',
            r'<\|.*?\|>',
            r'act\s+as\s+if',
            r'pretend\s+to\s+be'
        ]
        
        sanitized = text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '[SANITIZED]', sanitized, flags=re.IGNORECASE)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def generate_content(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        system_message: str = None
    ) -> str:
        """
        Generate content using GPT-4o-mini with input validation
        
        Args:
            prompt: The user prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
            
        Returns:
            Generated text response
        """
        try:
            # Validate and sanitize inputs
            validated_inputs = self._validate_input_parameters(prompt, temperature, max_tokens, system_message)
            
            messages = []
            
            # Add system message if provided
            if validated_inputs['system_message']:
                messages.append({"role": "system", "content": validated_inputs['system_message']})
            
            # Add user prompt
            messages.append({"role": "user", "content": validated_inputs['prompt']})
            
            # Make API call with validated parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=validated_inputs['temperature'],
                max_tokens=validated_inputs['max_tokens'],
                response_format={"type": "text"}
            )
            
            return response.choices[0].message.content
            
        except ValueError as e:
            self.logger.error(f"Input validation error: {str(e)}")
            return self._fallback_response(f"Validation Error: {str(e)}")
        except Exception as e:
            self.logger.error(f"GPT-4o-mini API Error: {str(e)}")
            return self._fallback_response(prompt)
    
    async def generate_content_async(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        system_message: str = None
    ) -> str:
        """Async version of generate_content"""
        return await asyncio.to_thread(
            self.generate_content, 
            prompt, 
            temperature, 
            max_tokens, 
            system_message
        )
    
    def generate_json_content(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        system_message: str = None
    ) -> Dict[str, Any]:
        """
        Generate JSON content using GPT-4o-mini with JSON mode and input validation
        
        Returns:
            Parsed JSON response
        """
        try:
            # Validate and sanitize inputs
            validated_inputs = self._validate_input_parameters(prompt, temperature, max_tokens, system_message)
            
            messages = []
            
            # Enhanced system message for JSON output
            json_system_message = (
                validated_inputs['system_message'] or 
                "You are a helpful assistant that responds with valid JSON only. "
                "Ensure your response is properly formatted JSON."
            )
            messages.append({"role": "system", "content": json_system_message})
            
            # Add instruction for JSON format to prompt
            json_prompt = f"{validated_inputs['prompt']}\n\nPlease respond with valid JSON only."
            messages.append({"role": "user", "content": json_prompt})
            
            # Make API call with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=validated_inputs['temperature'],
                max_tokens=validated_inputs['max_tokens'],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except ValueError as e:
            self.logger.error(f"JSON input validation error: {str(e)}")
            return self._fallback_json_response(f"Validation Error: {str(e)}")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            return self._fallback_json_response(f"JSON Parsing Error: {str(e)}")
        except Exception as e:
            self.logger.error(f"GPT-4o-mini JSON API Error: {str(e)}")
            return self._fallback_json_response(f"API Error: {str(e)}")
    
    async def generate_json_content_async(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        system_message: str = None
    ) -> Dict[str, Any]:
        """Async version of generate_json_content"""
        return await asyncio.to_thread(
            self.generate_json_content, 
            prompt, 
            temperature, 
            max_tokens, 
            system_message
        )
    
    def _fallback_response(self, error_context: str = "Unknown error") -> str:
        """Fallback response when API fails"""
        return json.dumps({
            "error": "API_FAILURE",
            "message": "GPT-4o-mini API call failed",
            "context": error_context[:200],  # Limit error context length
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        })
    
    def _fallback_json_response(self, error_context: str = "Unknown error") -> Dict[str, Any]:
        """Fallback JSON response when API fails"""
        return {
            "error": "API_FAILURE",
            "message": "GPT-4o-mini JSON API call failed",
            "context": error_context[:200],  # Limit error context length
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        }
    
    def set_model(self, model_name: str):
        """Change the model (for testing different GPT models)"""
        self.model = model_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model configuration"""
        return {
            "model": self.model,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens
        }


# Backward compatibility alias
GptApi = GptApiWrapper