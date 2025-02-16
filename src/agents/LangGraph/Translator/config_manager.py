#import os
import yaml
#from typing import Dict, List

class ConfigManager:
    def __init__(self, config_path: str = "config_agent_langgraph_translator.yaml", project_root: str= "./"):
        self.config_path = config_path
        self.project_root = project_root
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def get_model_config(self, model_name: str) -> dict:
        return self.config['models'].get(model_name, {})



