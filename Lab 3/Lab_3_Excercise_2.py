import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1
)

classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Quality Assurance Analyst.
Classify customer reviews strictly into the following fields:
- Sentiment: [Positive | Negative | Neutral | Mixed]
- Primary Category: [Hardware | Billing | Delivery | Software UI | Customer Service]
- Action Urgency: [Low | Medium | Critical]
- Summary: One concise sentence explaining the reason.

Few-shot Example 1:
Review: "The package arrived two days late and the box was torn open."
Classification:
Sentiment: Negative
Primary Category: Delivery
Action Urgency: Medium
Summary: Delivery delay combined with damaged external packaging.

Few-shot Example 2:
Review: "Battery swollen and sparking near the charging port while plugged in!"
Classification:
Sentiment: Negative
Primary Category: Hardware
Action Urgency: Critical
Summary: Severe safety hazard involving swollen battery sparking under charge."""),
    ("user", "Review: \"{review_text}\"\nClassification:")
])

chain = classifier_prompt | llm

test_reviews = [
    "The ergonomic keyboard design completely relieved my wrist strain after 8 hours of typing.",
    "My subscription was renewed automatically despite my cancellation request last week, please refund immediately!",
    "The camera software crashes whenever I attempt to record in 4K 60fps mode."
]

if __name__ == "__main__":
    print("=== Lab 3 - Exercise 2: Few-Shot Review Classifier ===")
    for text in test_reviews:
        print(f"\nReview: {text}")
        print("-" * 50)
        res = chain.invoke({"review_text": text})
        print(res.content)