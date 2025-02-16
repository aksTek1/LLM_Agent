import os
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    next: str | None
    should_continue: bool
    #messages: Annotated[list[HumanMessage | AIMessage], translate]