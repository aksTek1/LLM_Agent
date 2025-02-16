import os
#import json
from dotenv import load_dotenv
from langsmith import Client

#from langgraph.graph import Graph, StateGraph
#from langgraph.prebuilt import ToolMessage
#from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config_manager import ConfigManager
from model_llama3 import ChatLlamaManager
from chat_translator_application import ChatTranslationAgent


def get_project_root(current_file, levels_up):
    """Get the absolute path to the project root directory"""
    # Get the directory containing the current script
    current_dir = os.path.dirname(os.path.abspath(current_file))
    print(f'get_project_root- current_dir: {current_dir}, file:{__file__} ')
    for curLevel in range(levels_up-1):
        print(f'get_project_root- curLevel: {curLevel}, current_dir:{current_dir}')
        current_dir = os.path.dirname(current_dir)

    print(f'get_project_root- current_dir: {current_dir} ')
    # Go up one level to reach project root
    project_root = os.path.dirname(current_dir)
    return project_root


def main():
    # Get project root directory.
    # the root of the project is 2 levels up from current dir
    levels_up = 4
    project_root = get_project_root(__file__, levels_up)
    print(f'project_root:{project_root}')

    # Construct absolute paths for the configs
    config_path = os.path.join(project_root, "config", "config_agent_langgraph_translator.yaml")
    #json_input_path = os.path.join(project_root, "data", "input", "input_agent_langgraph_translator.json")
    #output_path = os.path.join(project_root, "data", "output", "results_agent_langgraph_translator.json")

    #with open(json_input_path, 'r') as f:
    #    input_data = json.load(f)

    # Enable LangSmith tracing
    load_dotenv(dotenv_path=project_root+'/.env',override=True)
    langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
    agent_translator_langsmith_client = Client(api_key=langsmith_api_key)


    config_manager = ConfigManager(config_path, project_root)
    chatllama_manager = ChatLlamaManager("llama3", config_manager)
    chatTranslate = ChatTranslationAgent(chatllama_manager, config_manager)

    #text = "My name is Forrest Gump, people call me Forrest Gump"
    #chatTranslate.process_text(text)
    chatTranslate.run_interactive()

    #print(f"Original: {text}")
    #print(f"Translation: {translation}")

if __name__ == "__main__":
    main()