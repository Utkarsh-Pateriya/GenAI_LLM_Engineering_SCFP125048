import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

parser = StrOutputParser()

# Chain step 1: Sentiment Extraction
sentiment_prompt = ChatPromptTemplate.from_template(
    "Analyze this product review: \"{review}\". State whether the overall sentiment is POSITIVE, NEGATIVE, or NEUTRAL, followed by a one-line rationale."
)

# Chain step 2: Remediation Plan based on output
action_prompt = ChatPromptTemplate.from_template(
    """Based on this review evaluation:
"{sentiment_analysis}"

Generate an operational action item for the customer satisfaction team. Keep it under 3 bullet points."""
)

full_pipeline = (
    {"sentiment_analysis": sentiment_prompt | llm | parser}
    | action_prompt
    | llm
    | parser
)

if __name__ == "__main__":
    print("=== Lab 5 - Exercise 1: Multi-Stage Sentiment Pipeline ===")
    test_input = "The noise cancellation is phenomenal on flights, but the headband plastic snapped after only 3 weeks of gentle use."
    print(f"Input Review: {test_input}\n")
    result = full_pipeline.invoke({"review": test_input})
    print("Generated Operations Workflow:")
    print(result)