import datetime
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import first_responder, revisor
from tool_executor import execute_tools

MAX_ITERATIONS = 2
builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

def event_loop(state: List[BaseMessage]) -> str:
    if len(state) > MAX_ITERATIONS:
        return END
    return "draft"

builder.add_conditional_edges("revise", event_loop, {END: END, "execute_tools":"execute_tools"})
builder.set_entry_point("draft")
graph = builder.compile()


if __name__ == "__main__":
    print("Hello LangGraph!")
