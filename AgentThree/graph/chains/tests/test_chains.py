import pprint
from dotenv import load_dotenv

from AgentThree.graph.chains import hallucination_grader
from AgentThree.graph.chains.hallucination_grader import GradeHallucinations
from AgentThree.graph.chains.router import RouteQuery, question_router
load_dotenv()

from AgentThree.graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})

    pprint(generation)

def test_hallucination_grader() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"question": question, "context": docs})

    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": generation})

    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke({"documents": docs, "generation": "In order to make pizza we need to first start with the dough"})

    assert not res.binary_score

def test_question_router() -> None:
    question = "agent memory"
    
    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_question_router_answer_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"