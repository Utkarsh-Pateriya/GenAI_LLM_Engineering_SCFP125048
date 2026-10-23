import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

class ResearchDigest(BaseModel):
    key_breakthrough: str = Field(description="The core scientific discovery in plain terms")
    business_impact: str = Field(description="Commercial relevance for executives")
    everyday_analogy: str = Field(description="A real-world metaphor explaining the concept to a high schooler")

parser = JsonOutputParser(pydantic_object=ResearchDigest)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Chief Science Communicator. Summarize complex scientific literature into actionable formats.\n{format_instructions}"),
    ("user", "Technical Abstract:\n{abstract}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

abstract_sample = """
We introduce a hybrid quantum-classical optimization algorithm leveraging Parameterized Quantum Circuits (PQCs) 
coupled with classical stochastic gradient approximations to solve high-dimensional combinatorial scheduling problems. 
Benchmarking against standard simulated annealing reveals a 4.2x reduction in convergence iterations on graph-partitioning 
topologies containing upwards of 10^5 nodes, while maintaining noise resilience under noisy intermediate-scale quantum (NISQ) simulation.
"""

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 2: Technical Abstract Deconstruction Pipeline ===")
    parsed_output = chain.invoke({"abstract": abstract_sample})
    print("Parsed JSON Object:")
    for k, v in parsed_output.items():
        print(f"\n[{k.upper()}]:\n{v}")