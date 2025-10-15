

from typing import Dict, Any
from AgentThree.graph.chains.generation import generation_chain
from AgentThree.graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"question": question, "context": documents})
    return {"generation": generation, "question": question, "documents": documents}