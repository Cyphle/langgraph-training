from typing import List, Sequence, TypedDict, Annotated

from annotated_types import Ge
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflection_chain

class MessageGraph(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return { "messages": [generate_chain.invoke({"messages": state["messages"]})] }

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return { "messages": [HumanMessage(content=res.content)] }

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={ END: END, REFLECT: REFLECT })
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello LangGraph!")
    inputs = HumanMessage(content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post

                                  """)
    response = graph.invoke(inputs)