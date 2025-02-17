from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.agents.LangGraph.Translator.ollama_manager import OllamaManager
from src.agents.LangGraph.Translator.prompt_manager import PromptManager
from src.agents.LangGraph.Translator.config_manager import ConfigManager
from src.agents.LangGraph.Translator.agent_state import AgentState

from langsmith import traceable


class ChatTranslationAgent:
    def __init__(self, model_manager: OllamaManager, config_manager: ConfigManager):
        self.model_manager = model_manager
        self.config_manager = config_manager
        self.prompt_manager = PromptManager(config_manager)
        self.target_language = config_manager.config["target_language"]
        self.prompt_name = config_manager.config["prompt_name"]


    @traceable(project_name='LLM_Project_agent_translator')
    def create_translator_prompt(self) -> ChatPromptTemplate:
        system_message = human_message = ""
        prompt_templates = self.prompt_manager.get_prompt(self.prompt_name).get('translator', {})
        for role, prompt_template_value in prompt_templates.items():
            if role == "system":
                system_message = prompt_template_value.replace("{target_language}", self.target_language)
            if role == "human":
                human_message = prompt_template_value

        print(f"create_translator_prompt system_message:{system_message}, human_message: {human_message}")


        return ChatPromptTemplate.from_messages([
                    ("system", system_message),
                    ("human", human_message)
        ])


    @traceable(project_name='LLM_Project_agent_translator')
    def should_continue(self, state: AgentState) -> Literal["get_input", "end"]:
        if not state["should_continue"]:
            return "end"
        return "get_input"


    @traceable(project_name='LLM_Project_agent_translator')
    def get_input(self, state: AgentState) -> AgentState:
        user_input = input("Enter text to translate (or 'quit'/'exit' to end): ")

        # Check for exit commands
        if user_input.lower() in ['quit', 'exit']:
            return {
                "messages": state["messages"],
                "next": None,
                "should_continue": False
            }

        print(f"get_input messages:{state["messages"]}, should_continue:{self.should_continue} ")
        return {
            "messages": [*state["messages"], HumanMessage(content=user_input)],
            "next": None,
            "should_continue": True
        }

    def validate_response(self, response) -> str:
        """Validate and extract content from LLM response"""
        if response is None:
            raise ValueError("Received empty response from LLM")

        if hasattr(response, 'content'):
            return str(response.content)
        elif isinstance(response, str):
            return response
        elif isinstance(response, dict) and 'content' in response:
            return str(response['content'])
        else:
            return str(response)

    @traceable(project_name='LLM_Project_agent_translator')
    def translate(self, state: AgentState) -> AgentState:
        try:
            print(f"translate state[messages]: {state["messages"]}")
            prompt = self.create_translator_prompt()
            print(f'translate prompt: {prompt}')

            # Get the last human message
            last_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]
            if not last_message.content.strip():
                raise ValueError("human message is empty")
            print(f'last_message: {last_message}, input_text: {last_message} and content:{last_message.content}')

            user_input_dict = {"user_input": last_message.content}
            # Generate translation
            chain = prompt | self.model_manager.llm
            response = chain.invoke(user_input_dict)
            print(f'translate response from LLM: {response}')
            translation_content = self.validate_response(response)
            if not translation_content.strip():
                raise ValueError("received empty response message from LLM")

            # Create a valid AIMessage
            translation_message = AIMessage(content=translation_content)

            # Return new state
            return {"messages": [*state["messages"], translation_message],
                    "next": None, "should_continue": True}
        except Exception as e:
            print(f"Translation error: {str(e)}")
            print(f"Debug - Full error: {repr(e)}")
            error_message = AIMessage(content=f"Error: {str(e)}")
            return {
                "messages": [*state["messages"], error_message],
                "next": None,
                "should_continue": True
            }

    @traceable(project_name='LLM_Project_agent_translator')
    def create_workflow(self) -> Graph:
        """Create the translation workflow"""
        workflow = StateGraph(AgentState)

        #Add get input node
        workflow.add_node("get_input", lambda state: self.get_input(state))
        # Add translation node
        workflow.add_node("translate", lambda state: self.translate(state))

        # Set conditional edges
        workflow.add_conditional_edges("get_input",
                                        self.should_continue,
                                        {
                                            "get_input": "translate",
                                            "end": END
                                        }
        )

        # Add edge from translate back to the conditional
        workflow.add_edge("translate", "get_input")

        # Set entry point
        workflow.set_entry_point("get_input")

        # Compile workflow
        return workflow.compile()


    @traceable(project_name='LLM_Project_agent_translator')
    def run_interactive(self):
        agent = self.create_workflow()

        # Initialize state
        initial_state = {
            "messages": [
                SystemMessage(content=f"Translation bot initialized. Target language: {self.target_language}")
            ],
            "next": None,
            "should_continue": True
        }

        # Run the workflow
        final_state = agent.invoke(initial_state)

        # Print farewell message
        print("end of session!")
