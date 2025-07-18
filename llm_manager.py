import requests
from typing import Dict, Any, List
from config import Config
import json

class LLMManager:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or Config.LLM_PROVIDER
        self.model = model
        self.config = Config()
        
    def get_available_models(self) -> Dict[str, str]:
        """Get available models for the current provider"""
        if self.provider == "together":
            return self.config.TOGETHER_MODELS
        elif self.provider == "groq":
            return self.config.GROQ_MODELS
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_together_api(self, messages: List[Dict], **kwargs) -> str:
        """Call Together AI API"""
        if not self.config.TOGETHER_API_KEY:
            raise ValueError("Together API key not found")
        
        headers = {
            "Authorization": f"Bearer {self.config.TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model or self.config.TOGETHER_MODELS["mixtral-8x7b"],
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None)
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")
    
    def _call_groq_api(self, messages: List[Dict], **kwargs) -> str:
        """Call Groq API"""
        if not self.config.GROQ_API_KEY:
            raise ValueError("Groq API key not found")
        
        headers = {
            "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model or self.config.GROQ_MODELS["mixtral-8x7b"],
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "stop": kwargs.get("stop", None)
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API call failed: {response.status_code}, {response.text}")
    
    def generate_response(self, messages: List[Dict], **kwargs) -> str:
        """Generate response using the configured provider"""
        if self.provider == "together":
            return self._call_together_api(messages, **kwargs)
        elif self.provider == "groq":
            return self._call_groq_api(messages, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a simple prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_response(messages, **kwargs)