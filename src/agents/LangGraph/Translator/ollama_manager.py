from abc import ABC, abstractmethod
from config_manager import ConfigManager


class OllamaManager(ABC):
    def __init__(self, model_name: str, config_mgr: ConfigManager):
        self.model_name = model_name
        self.config = config_mgr.get_model_config(model_name)
        self.base_url = self.config.get('base_url', 'http://localhost:11434/')
        self.temperature = self.config.get('temperature', 0.0)
        self.max_tokens = self.config.get('max_tokens', 1024)
        self.llm = None
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

