import os
import yaml
from typing import List, Optional
from config_manager import ConfigManager


class PromptManager:
    """
    Manages prompt templates from files
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize Prompt Manager

        :param prompts_directory: Directory containing prompt template files
        """
        self.prompts_directory = config_manager.config['prompts_dir'].replace(
                                            "{PROJECT_ROOT}", config_manager.project_root)
        self.prompts = {}
        #self.prompt_metadata = {}
        self._load_prompts()

    def _load_prompts(self):
        """
        Load all prompt templates from files in the specified directory
        """
        # Ensure directory exists
        if not os.path.exists(self.prompts_directory):
            raise ValueError(f"Prompt directory not found: {self.prompts_directory}")

        # Load prompts from .txt files
        for filename in os.listdir(self.prompts_directory):
            if '_agent_langgraph_translator' in filename and filename.endswith('.txt'):
                prompt_name = os.path.splitext(filename)[0]
                print(f'prompt_name: {prompt_name}')
                file_path = os.path.join(self.prompts_directory, filename)

                with open(file_path, 'r') as f:
                    #self.prompts[prompt_name] = f.read().strip()
                    self.prompts[prompt_name] = yaml.safe_load(f)
                    print(f"_load_prompts: {self.prompts[prompt_name]}, type: {type(self.prompts[prompt_name])}")

    def get_prompt(self, prompt_name: str) -> Optional[str]:
        """
        Retrieve a specific prompt template

        :param prompt_name: Name of the prompt template
        :return: Prompt template or None
        """
        return self.prompts.get(prompt_name)