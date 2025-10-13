from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Cirituqe of chat is missing.")
    superfluous: str = Field(description="Cirituqe of chat is superfluous.")

class AnswerQuestion(BaseModel):
    """Answer the questions."""

    answer: str = Field(description="~250 word detialed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )

    