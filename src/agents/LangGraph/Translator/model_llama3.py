from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from pydantic_core.core_schema import none_schema
from sympy.physics.units import temperature
import os

from src.agents.LangGraph.Translator.ollama_manager import OllamaManager


class ChatLlamaManager(OllamaManager):
    def initialize_model(self):
        if self.llm is None:
            self.llm = ChatOllama(
                        base_url= self.base_url,
                        model = self.model_name,
                        temperature = self.temperature,
                        max_tokens =  self.max_tokens,
                        request_timeout = 60,
                        headers={"User-Agent": os.getenv("USER_AGENT", "LangChain-Agent")}
            )