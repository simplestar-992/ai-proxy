#!/usr/bin/env python3
"""
AIProxy - Unified API proxy for AI models
OpenAI, Anthropic, Google, local models
"""
import os
import sys
import json
import requests
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"

@dataclass
class ModelConfig:
    provider: ModelProvider
    model: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7

class AIProxy:
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.default_model = "gpt-4"
    
    def add_model(self, name: str, config: ModelConfig) -> None:
        self.models[name] = config
    
    def chat(self, model_name: str, messages: List[Dict], **kwargs) -> str:
        config = self.models.get(model_name)
        if not config:
            config = self.models.get(self.default_model)
        
        if not config:
            raise ValueError(f"No model configured: {model_name}")
        
        if config.provider == ModelProvider.OPENAI:
            return self._openai_chat(config, messages, **kwargs)
        elif config.provider == ModelProvider.ANTHROPIC:
            return self._anthropic_chat(config, messages, **kwargs)
        elif config.provider == ModelProvider.GOOGLE:
            return self._google_chat(config, messages, **kwargs)
        elif config.provider == ModelProvider.LOCAL:
            return self._local_chat(config, messages, **kwargs)
    
    def _openai_chat(self, config: ModelConfig, messages: List[Dict], **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens),
            "temperature": kwargs.get("temperature", config.temperature)
        }
        response = requests.post(
            f"{config.base_url or 'https://api.openai.com/v1'}/chat/completions",
            headers=headers, json=data
        )
        return response.json()["choices"][0]["message"]["content"]
    
    def _anthropic_chat(self, config: ModelConfig, messages: List[Dict], **kwargs) -> str:
        headers = {
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", config.max_tokens)
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=data
        )
        return response.json()["content"][0]["text"]
    
    def _google_chat(self, config: ModelConfig, messages: List[Dict], **kwargs) -> str:
        headers = {"Authorization": f"Bearer {config.api_key}"}
        data = {
            "contents": [{"parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {
                "maxOutputTokens": config.max_tokens,
                "temperature": config.temperature
            }
        }
        response = requests.post(
            f"{config.base_url}/v1beta/models/{config.model}:generateContent",
            headers=headers, json=data
        )
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    def _local_chat(self, config: ModelConfig, messages: List[Dict], **kwargs) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": self._messages_to_prompt(messages),
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
        response = requests.post(
            f"{config.base_url}/completion",
            headers=headers, json=data
        )
        return response.json()["choices"][0]["text"]
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

# CLI
if __name__ == "__main__":
    proxy = AIProxy()
    
    # Configure models
    if os.getenv("OPENAI_API_KEY"):
        proxy.add_model("gpt-4", ModelConfig(
            provider=ModelProvider.OPENAI,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        ))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        proxy.add_model("claude", ModelConfig(
            provider=ModelProvider.ANTHROPIC,
            model="claude-3-opus-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        ))
    
    if len(sys.argv) > 1:
        messages = [{"role": "user", "content": sys.argv[1]}]
        model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4"
        response = proxy.chat(model, messages)
        print(response)
    else:
        print("Usage: aiproxy <message> [model]")
