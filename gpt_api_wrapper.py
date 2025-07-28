"""
GPT-4o-mini API Wrapper for Agentic Building Coverage Analysis
Replaces Anthropic calls with OpenAI GPT-4o-mini implementation
"""

import openai
import json
import os
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime


class GptApiWrapper:
    """OpenAI GPT-4o-mini API wrapper replacing original GptApi"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client with GPT-4o-mini model"""
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4o-mini"
        self.default_temperature = 0.1
        self.default_max_tokens = 2000
        
    def generate_content(
        self, 
        prompt: str, 
        temperature: float = None, 
        max_tokens: int = None,
        system_message: str = None
    ) -> str:
        """
        Generate content using GPT-4o-mini
        
        Args:
            prompt: The user prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
            
        Returns:
            Generated text response
        """
        try:
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                response_format={"type": "text"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"GPT-4o-mini API Error: {str(e)}")
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
        Generate JSON content using GPT-4o-mini with JSON mode
        
        Returns:
            Parsed JSON response
        """
        try:
            messages = []
            
            # Enhanced system message for JSON output
            json_system_message = (
                system_message or 
                "You are a helpful assistant that responds with valid JSON only. "
                "Ensure your response is properly formatted JSON."
            )
            messages.append({"role": "system", "content": json_system_message})
            
            # Add instruction for JSON format to prompt
            json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
            messages.append({"role": "user", "content": json_prompt})
            
            # Make API call with JSON mode
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            return self._fallback_json_response()
        except Exception as e:
            print(f"GPT-4o-mini JSON API Error: {str(e)}")
            return self._fallback_json_response()
    
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
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when API fails"""
        return json.dumps({
            "error": "API_FAILURE",
            "message": "GPT-4o-mini API call failed",
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        })
    
    def _fallback_json_response(self) -> Dict[str, Any]:
        """Fallback JSON response when API fails"""
        return {
            "error": "API_FAILURE",
            "message": "GPT-4o-mini JSON API call failed",
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