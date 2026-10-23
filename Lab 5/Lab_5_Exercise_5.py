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
    temperature=0.0
)

class TicketTriage(BaseModel):
    urgency: str = Field(description="LOW, MEDIUM, HIGH, or SEV-1")
    department: str = Field(description="Network Ops, Security, Billing, or Tier 1")
    sla_hours: int = Field(description="Target resolution SLA in hours")
    recommended_action: str = Field(description="Initial triage step")

parser = JsonOutputParser(pydantic_object=TicketTriage)

triage_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an automated IT Service Management (ITSM) dispatcher. Triage incoming tickets according to ITIL standards.\n{formatting}"),
    ("user", "Incoming Ticket Description:\n{ticket_body}")
]).partial(formatting=parser.get_format_instructions())

triage_chain = triage_prompt | llm | parser

sample_ticket = """
Subject: Multiple employee accounts reporting MFA lockouts and unauthorized IP warnings from Eastern Europe.
Body: Over the last 20 minutes, 14 corporate VPN accounts have triggered repeated brute-force authentication alerts. 
Two domain administrator credentials were included in the targeted list.
"""

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 5: ITSM Ticket Escalation Engine ===")
    decision = triage_chain.invoke({"ticket_body": sample_ticket})
    print(decision)