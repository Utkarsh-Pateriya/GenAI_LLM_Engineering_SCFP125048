import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

parser = StrOutputParser()

# Step 1: Weather risk classification
risk_template = PromptTemplate.from_template(
    """Analyze the following weather report for an outdoor event:
Weather: {weather_conditions}

Identify the primary hazards (e.g., wind, rain, heat, lightning) and rate the risk as LOW, MODERATE, or HIGH."""
)

# Step 2: Operational mitigation recommendations
advisory_template = PromptTemplate.from_template(
    """Hazard Assessment:
{hazard_assessment}

Event Type: {event_type}

Provide 3 critical logistical safety precautions the event organizers must implement."""
)

pipeline = (
    {"hazard_assessment": risk_template | llm | parser, "event_type": lambda x: x["event_type"]}
    | advisory_template
    | llm
    | parser
)

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 4: Contextual Event Feasibility Pipeline ===")
    conditions = "Temperature 39°C (102°F), humidity 78%, wind gusts up to 45 km/h, UV index 11+."
    event = "Outdoor Half-Marathon with 2,500 participants"
    
    print(f"Event: {event}")
    print(f"Report: {conditions}\n")
    
    safety_advisory = pipeline.invoke({"weather_conditions": conditions, "event_type": event})
    print("Safety Advisory:")
    print(safety_advisory)