import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

support_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Tier-2 Technical Support Specialist for 'ApexCloud Hosting'.
Guidelines:
1. Tone must be empathetic, calm, and technically authoritative.
2. Acknowledge customer frustration without admitting liability unnecessarily.
3. Provide step-by-step troubleshooting actions.
4. Conclude with a clear escalation pathway if the initial steps fail."""),
    ("user", "Customer Incident: {incident_report}")
])

chain = support_prompt | llm

incident = """
Ticket #8841: Our production e-commerce database is throwing 504 Gateway Timeouts 
since your platform maintenance at 02:00 AM UTC. We have lost over 40 customer checkouts in the last hour! 
Fix this immediately or we will terminate our enterprise contract.
"""

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 4: Tone-Adapted Support Generation ===")
    response = chain.invoke({"incident_report": incident})
    print(response.content)