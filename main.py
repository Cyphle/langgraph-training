from typing import List, Sequence

from annotated_types import Ge
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflection_chain

REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: Sequence[BaseMessage]):
    return generate_chain.invoke({ "messages": state })

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflection_chain.invoke({ "messages": messages })
    return [HumanMessage(content=res.content)]

builder = MessageGraph()
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# Conditional node
def should_continue(state: List[BaseMessage]) -> bool:
    if len(state) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, { END: END, REFLECT: REFLECT })
builder.add_edge(REFLECT, GENERATE) # Oriented graph REFLECT -> GENERATE

graph = builder.compile()
print(graph.get_graph().draw_mermaid())

if __name__ == "__main__":
    print("Hello LangGraph!")