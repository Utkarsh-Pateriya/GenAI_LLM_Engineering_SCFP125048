import os
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

class InterviewQuestion(BaseModel):
    question: str = Field(description="The technical interview question")
    difficulty: str = Field(description="Junior, Mid, or Senior")
    expected_answer_keywords: List[str] = Field(description="List of 3-5 technical keywords expected in the answer")

class InterviewPlan(BaseModel):
    role: str = Field(description="Target role title")
    questions: List[InterviewQuestion] = Field(description="Set of structured interview questions")

parser = PydanticOutputParser(pydantic_object=InterviewPlan)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Engineering Hiring Manager. Prepare structured interview rubrics.\n{formatting}"),
    ("user", "Generate an interview plan for the role of: {role_title} focusing on: {focus_area}")
]).partial(formatting=parser.get_format_instructions())

chain = prompt | llm | parser

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 3: Structured Interview Questionnaire Pipeline ===")
    rubric = chain.invoke({
        "role_title": "Backend Distributed Systems Engineer",
        "focus_area": "Raft consensus, CAP theorem, and database sharding"
    })
    print(f"Role: {rubric.role}\n")
    for idx, q in enumerate(rubric.questions, start=1):
        print(f"Q{idx} ({q.difficulty}): {q.question}")
        print(f"   Keywords: {', '.join(q.expected_answer_keywords)}\n")